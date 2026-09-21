# ============================================================================
# STOCHASTIC DYNAMICS - Function factory for random processes (AR1, RW, ArrivalProcess)
# ConfigurableEpi specialises to two forms of stochastic dynamics:
# 1. Gaussian processes on time-evolving scalar valued parameters. These get tracked in the state vector as "latent" states.
# 2. Marked point processes (ArrivalProcess) that are not associated with one parameter but rather with a random change in the compartments.
# For example, an ArrivalProcess could represent a sudden influx of infected individuals with a new variant.
# ============================================================================


# ============================================================================
# Time-varying parameters in Unconstrained Space
# ============================================================================
#
# DESIGN: Time-varying parameters are stored in UNCONSTRAINED space in the state vector.
# This avoids domain errors when sigma points drift (e.g., log of negative).
#
# - State vector stores: the unconstrained EKP coordinates for Rt, rho, etc.
# - update_single: simple linear dynamics
# - extract_constrained: applies the EKP constraint when reading values for use
#
# Process noise draws w_i are unit-normal; sigma scaling is done in update_single.
# ============================================================================

# Scalar fast path. EKP's `transform_unconstrained_to_constrained(pd, [x])` is written for batched
# matrices: per call it rebuilds the name => constraint dict and the batch index lists (~5 us and
# ~125 allocations for one scalar). These maps run several times per particle per filter step, so
# that generic path was ~85% of a daily PF run — and invisible to the JET guard, which only looks at
# ConfigurableEpi frames. Every prior here is one-dimensional with a single constraint (`only`
# enforces it), so resolve that constraint's two maps ONCE, at build time. EKP stores them as
# `::Function` fields; lifting them into type parameters is what makes the per-particle call static.
# They are the same functions EKP would broadcast, so results are bit-identical.
struct ScalarConstraint{F, G}
    to_constrained::F
    to_unconstrained::G
end

function ScalarConstraint(init::ParameterDistribution)
    constraint = only(get_all_constraints(init))
    return ScalarConstraint(
        constraint.unconstrained_to_constrained, constraint.constrained_to_unconstrained
    )
end

@inline _to_constrained(constraint::ScalarConstraint, unconstrained) =
    constraint.to_constrained(unconstrained)

# Cold-path conveniences (model build, initialisation): resolve and apply in one go.
_to_constrained(init::ParameterDistribution, unconstrained) =
    _to_constrained(ScalarConstraint(init), unconstrained)
_to_unconstrained(init::ParameterDistribution, constrained) =
    _to_unconstrained(ScalarConstraint(init), constrained)

@inline _named_tuple(::Val{names}, values::Tuple) where {names} =
    NamedTuple{names}(values)

@inline function _to_unconstrained(constraint::ScalarConstraint, constrained)
    unconstrained = constraint.to_unconstrained(constrained)
    isfinite(unconstrained) || throw(
        ArgumentError(
            "Constraint transform produced a non-finite unconstrained value for $constrained. " *
                "Bounded latent states cannot be initialized exactly at the boundary."
        )
    )
    return unconstrained
end

# ============================================================================
# Time-varying parameter Process Noise: w passed in, sigma scaling inside dynamics
# ============================================================================

"""
    ou_step(tau, dt) -> (rho, innovation_factor)

The exact one-step transition of an Ornstein–Uhlenbeck process with correlation time `tau` sampled
at `dt`: `rho = exp(-dt/tau)` and `innovation_factor = sqrt(1 - rho^2)`, so a process with
stationary sd `sigma` takes an innovation of `sigma * innovation_factor`.

`-expm1(-2x)` rather than `1 - exp(-2x)` is load-bearing, not cosmetic: at `tau = 1e8` days,
`1 - exp(-1.4e-7)` loses every significant digit to cancellation while `-expm1` stays exact. That
matters because a large `tau` is precisely the regime a hierarchy can wander into.
"""
function ou_step(tau::Real, dt::Real)
    @inline
    (isfinite(tau) && tau > 0) || throw(
        ArgumentError(
            "AR(1)/OU correlation time must be finite and positive, got tau = $tau"
        )
    )
    x = dt / tau
    return exp(-x), sqrt(-expm1(-2x))
end

"""
    update_single(spec::ProcessParamSpec, old_unc, w_i, params, dt[, constraint])

Advance a single latent state in UNCONSTRAINED space by one step of length `dt`, given unit noise
`w_i ~ N(0,1)`. Returns the new unconstrained value.

`dt` is threaded in rather than baked into the parameters so both process types are
**dt-invariant**: the AR(1)/OU takes a correlation time in days and a stationary sd, the random
walk a per-sqrt-day diffusion rate. Changing `weekly_dt` therefore changes the sampling of the same
continuous-time process instead of silently redefining it.

The supplied unit-noise draw is scaled by the process-noise parameter inside
the update.

`constraint` is `spec.init`'s pre-resolved `ScalarConstraint`. The filters' hot path passes it;
omitting it resolves the constraint on the spot, which is fine for a one-off call.
"""

function update_single(
        spec::RWParamSpec, old_unc, w_i,
        params::NamedTuple, dt::Real, constraint = nothing
    )
    sigma_rate = get_param_value(spec.sigma, params)
    # Driftless Brownian motion in unconstrained space. `sigma` is a per-sqrt-day RATE, so the step
    # sd is `sigma * sqrt(dt)` — the same dt-invariance the AR(1) has.
    return old_unc + sigma_rate * sqrt(dt) * w_i
end

function update_single(
        spec::AR1ParamSpec, old_unc, w_i,
        params::NamedTuple, dt::Real, constraint = ScalarConstraint(spec.init)
    )
    mu_val = get_param_value(spec.mu, params)
    tau_val = get_param_value(spec.tau, params)
    sigma_val = get_param_value(spec.sigma, params)   # STATIONARY sd, not the innovation sd

    # Exact OU transition in unconstrained space, centred on the unconstrained image of mu.
    rho, innovation = ou_step(tau_val, dt)
    unconstrained_mu = _to_unconstrained(constraint, mu_val)
    return unconstrained_mu + rho * (old_unc - unconstrained_mu) + sigma_val * innovation * w_i
end

function update_single(
        spec::IntegratedParamSpec, old_unc, w_i,
        params::NamedTuple, dt::Real, constraint = nothing
    )
    # Noise-free: `w_i` is deliberately unused, and so is `constraint` — the increment is added in
    # the unconstrained chart, so no map is needed. `params` holds every latent's constrained value at
    # the START of the step, so the rate read here is the pre-advance one (explicit Euler).
    return old_unc + params[spec.rate] * dt / spec.per_days
end


# ============================================================================
# ARRIVAL PROCESS — marked-point-process dynamics (PARTICLE FILTER ONLY)
# ============================================================================
#
# The AR1/RW machinery above propagates continuous latent states — they carry state because an OU /
# random walk is Markov in its own coordinate. An `ArrivalProcess` (see ParamSpec.jl) is the
# discrete, particle-only sibling, and it is STATELESS: its mark is absorbed straight into the
# compartments, so the model state already records that it fired — a separate flag would be
# redundant bookkeeping. `advance_arrival!` (below) is its `draw_mark → apply!` step, run by
# `build_stochastic_update`'s `advance` at the START of a step (before the flow). Self-excitation
# (e.g. single-shot) is therefore just a state-reading intensity: `rate(x, latent, hyper, t)` can
# read the very compartment its own jump seeds.
# ============================================================================

# Fire the stateless jump drivers in place, against the compartments `x_model`. Recurses over the
# (heterogeneous) spec tuple rather than looping, so nothing is boxed (keeps the closure JET-clean).
# Called only on a stochastic step (`rng !== nothing`); the empty case is a no-op. Sequential: a
# later driver's intensity sees an earlier driver's seed, which is the correct within-step ordering.
@inline _advance_jumps!(x_model, ::Tuple{}, _latent, _hyper, _dt, _t, _rng) = x_model
@inline function _advance_jumps!(x_model, jump_specs::Tuple, latent, hyper, dt, t, rng)
    advance_arrival!(first(jump_specs), x_model, latent, hyper, dt, t, rng)
    return _advance_jumps!(x_model, Base.tail(jump_specs), latent, hyper, dt, t, rng)
end

"""
    step_arrival_probability(rate, dt) -> Float64

Per-step firing probability of a constant-hazard point process of intensity `rate` (per unit time)
over a step of length `dt`: `1 - exp(-rate·dt)`. `rate = 0` gives exactly `0`.
"""
@inline function step_arrival_probability(rate::Real, dt::Real)
    isfinite(rate) || throw(ArgumentError("arrival rate must be finite, got $rate"))
    rate >= 0 || throw(ArgumentError("arrival rate must be non-negative, got $rate"))
    isfinite(dt) || throw(ArgumentError("dt must be finite, got $dt"))
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    return -expm1(-rate * dt)
end

"""
    beta_mark(; mean_key = :mark_mean, concentration = 1.0, out_key = :mark)

Default mark sampler: a `mark(p_eff, rng)` closure drawing `out_key ~ Beta(μν, (1-μ)ν)` with mean
`μ = p_eff[mean_key] ∈ (0, 1)` and concentration `ν = concentration`. `ν = 1` is the U-shaped
"fizzle or jump" prior. The mean is read from `p_eff`, so it may be a fixed shared hyperparameter
or a learnable [`LiuWest`](@ref) one.
"""
function beta_mark(;
        mean_key::Symbol = :mark_mean, concentration::Real = 1.0, out_key::Symbol = :mark
    )
    conc = float(concentration)
    conc > 0 || throw(ArgumentError("concentration must be positive, got $concentration"))
    return function sample_beta_mark(p_eff, rng)
        mean_mark = getproperty(p_eff, mean_key)
        0 < mean_mark < 1 ||
            throw(ArgumentError("$mean_key must be in (0, 1), got $mean_mark"))
        value = rand(rng, Beta(mean_mark * conc, (1 - mean_mark) * conc))
        return NamedTuple{(out_key,)}((value,))
    end
end

"""
    seed_transition(model_names, from::Symbol, into::Symbol; size_key::Symbol)

Default state transition: a `transition!(x_model, mark, p_eff)` closure moving `mark[size_key]`
individuals from compartment `from` into compartment `into`, capped at the available `from`
population (mass-conserving). `model_names` is `ode_names(layout)` (or `all_names(layout)`), used
once to resolve compartment indices in the model prefix.
"""
function seed_transition(model_names, from::Symbol, into::Symbol; size_key::Symbol)
    i_from = findfirst(==(from), model_names)
    i_into = findfirst(==(into), model_names)
    i_from === nothing && throw(ArgumentError("no compartment `$from` in model_names"))
    i_into === nothing && throw(ArgumentError("no compartment `$into` in model_names"))
    return function seed!(x_model, mark, _p_eff)
        requested = getproperty(mark, size_key)
        moved = min(max(requested, zero(requested)), x_model[i_from])
        x_model[i_from] -= moved
        x_model[i_into] += moved
        return x_model
    end
end

# --- Generic redistribution primitives -------------------------------------

"""
    pool_redistribute!(x_model, sources, targets, weights) -> x_model

Mass-conserving **relabel**: pool the mass held in the `sources` compartments, empty them, then add
`weights[k] · pool` into `targets[k]`. `sources`/`targets` are index tuples into `x_model` and
`weights` a matching tuple; for exact conservation `weights` should sum to `1` (a `1 − f` weight
inherits `f`'s rounding — ~1 ulp — as any floating-point split does). `sources` and `targets` may
**overlap**: the pool is summed and the sources cleared *before* any target is written, so a
compartment that is both ends at its `weights` share rather than accumulating. `targets` should be
distinct. The redistribution sibling of [`seed_transition`](@ref): where that moves a fixed count
one-to-one, this splits a pooled mass across many targets by a (typically mark-derived) weight
vector. Throws `DimensionMismatch` if `targets` and `weights` differ in length.
"""
@inline function pool_redistribute!(x_model, sources, targets, weights)
    # Precondition, not decoration: `weights[k]` is read under `@inbounds`, so a length mismatch
    # would be an out-of-bounds read (UB) rather than a clean error. For tuple arguments both
    # lengths are compile-time constants, so this folds away entirely — zero cost in the hot path.
    length(targets) == length(weights) || throw(
        DimensionMismatch(
            "pool_redistribute!: targets ($(length(targets))) and weights ($(length(weights))) must have equal length"
        )
    )
    Ty = eltype(x_model)
    pooled = zero(Ty)
    @inbounds for s in sources
        pooled += x_model[s]
    end
    @inbounds for s in sources
        x_model[s] = zero(Ty)
    end
    @inbounds for k in eachindex(targets)
        x_model[targets[k]] += weights[k] * pooled
    end
    return x_model
end

"""
    pro_rata_move!(x_model, sources, targets, amount) -> x_model

Move up to `amount` individuals out of the `sources` compartments into the `targets`, split **pro
rata** by each source's current occupancy and capped at the total available (mass-conserving, never
negative). `sources[k]` drains into `targets[k]`, so the tuples are paired and equal length, and
`sources` must be disjoint from `targets`. A non-positive `amount`, or empty pools, moves nothing.
The multi-source generalization of the move step in [`seed_transition`](@ref) — seed several
compartments from several pools at once while preserving their proportions.

The per-source cap `min(share, available)` is not cosmetic: when the request saturates the pool the
intended share is exactly `available`, but `moved · available / pool` can round **up** an ulp and
drive the compartment negative — which a downstream `max(x, ε)` clamp would then paper over by
creating mass. Throws `DimensionMismatch` if `sources` and `targets` differ in length. Disjointness
of `sources` and `targets` is a documented contract, not enforced (an O(n·m) runtime check with no
compile-time fold, unlike the length check).
"""
@inline function pro_rata_move!(x_model, sources, targets, amount)
    # See `pool_redistribute!`: `targets[k]` is read under `@inbounds`, so guard the paired length.
    # Folds away for tuples (compile-time lengths); a clean error for anything else.
    length(sources) == length(targets) || throw(
        DimensionMismatch(
            "pro_rata_move!: sources ($(length(sources))) and targets ($(length(targets))) must have equal length"
        )
    )
    Ty = eltype(x_model)
    requested = max(Ty(amount), zero(Ty))
    pool = zero(Ty)
    @inbounds for s in sources
        pool += x_model[s]
    end
    (requested > zero(Ty) && pool > zero(Ty)) || return x_model
    moved = min(requested, pool)
    @inbounds for k in eachindex(sources)
        s = sources[k]
        available = x_model[s]
        take = min(moved * available / pool, available)
        x_model[s] -= take
        x_model[targets[k]] += take
    end
    return x_model
end

# --- Driver traits -----------------------------------------------------------

"""
    carries_state(spec) -> Bool

Whether a driver must carry its own coordinate in the state vector to be Markov — i.e. whether it
claims slots in the [`StateLayout`](@ref). `true` for the Gaussian latent processes (the `ParamSpec`
default — an OU/random walk is Markov in its own value); `false` for [`ArrivalProcess`](@ref), whose
mark is absorbed into the compartments, so the model state already records that it fired.

This is deliberately **separate** from [`supports_gaussian_filter`](@ref): carrying state (the Markov
property) is orthogonal to *how* a process evolves (diffusion vs jump) or which filter can represent
it. The layout partitions on this trait alone.
"""
carries_state(::ParamSpec) = true
carries_state(::ArrivalProcess) = false

"""
    supports_gaussian_filter(spec) -> Bool

Whether a process spec's state can be propagated by a Gaussian (UKF) filter. `true` for the
Gaussian latent processes (the `ParamSpec` default — AR1, RW); `false` for particle-only processes
like [`ArrivalProcess`](@ref) whose discrete fired/not-fired mark-switch a single Gaussian cannot
represent.
"""
supports_gaussian_filter(::ParamSpec) = true
supports_gaussian_filter(::ArrivalProcess) = false

"""
    assert_gaussian_filter_compatible(specs)

Throw if any spec is particle-only (`supports_gaussian_filter == false`). Call from a UKF build to
reject a model that declares an [`ArrivalProcess`](@ref): a Gaussian state cannot carry a
fired/not-fired mark-switch. Accepts a single spec or an iterable of specs.
"""
assert_gaussian_filter_compatible(spec::ParamSpec) = assert_gaussian_filter_compatible((spec,))
function assert_gaussian_filter_compatible(specs)
    bad = [s for s in specs if !supports_gaussian_filter(s)]
    isempty(bad) || throw(
        ArgumentError(
            "model declares particle-only process(es) $(typeof.(bad)); a Gaussian UKF cannot " *
                "represent them — use a particle filter (PF)"
        )
    )
    return nothing
end

"""
    advance_arrival!(process::ArrivalProcess, x_model, latent, hyper, dt, t, rng) -> x_model

Fire one **stateless** arrival (jump) driver against the current state — the `draw_mark → apply!`
atom of the driver protocol. `x_model` is a view of the model compartments (both the intensity's
state argument and the jump's target); `latent` is the constrained latent NamedTuple. Fires with
probability `step_arrival_probability(process.rate(x_model, latent, hyper, t), dt)`; on firing it
draws `process.sample_mark(hyper, rng)` and applies `process.transition!` to `x_model` in place, so
the mark is absorbed straight into the compartments. A zero intensity skips the draw, so a
self-extinguished driver burns no randomness.

Because the intensity reads the state, **self-excitation needs no stored history**: a single-shot
arrival is just `rate = (x, l, p, t) -> x[i_invader] > 0 ? 0.0 : p.arrival_rate`. `hyper` carries the
effective parameters (hyperparameters + any Liu-West `learned` values), so the rate and mark may be
learned online. [`build_stochastic_update`](@ref)'s `advance` calls this *before* the deterministic
flow (the **noise → flow** operator split), so a seeded arrival grows within its own step.
"""
@inline function advance_arrival!(process::ArrivalProcess, x_model, latent, hyper, dt, t, rng)
    p_fire = step_arrival_probability(process.rate(x_model, latent, hyper, t), dt)
    if p_fire > 0 && rand(rng) < p_fire
        mark = process.sample_mark(hyper, rng)
        process.transition!(x_model, mark, hyper)     # absorbed into the compartments
    end
    return x_model
end


# ============================================================================
# StochasticUpdate: the Lévy noise driver (diffusion coefficients + jumps)
# ============================================================================

"""
    StochasticUpdate{L}

The model's complete per-step stochastic driver — the Lévy noise increment (drift + Brownian
diffusion + marked jumps). Built by [`build_stochastic_update`](@ref) from the full list of driver
specs. `L` is the number of Gaussian **coefficient** drivers (AR1/RW), whose constrained values feed
the vector field; **jump** drivers ([`ArrivalProcess`](@ref)) contribute to `advance` but not `extract`.

Coefficient states are stored in UNCONSTRAINED space (so UKF sigma points can spread without domain
errors); `extract` applies the constraint bijector when reading them for the flow/measurement.

# Fields
- `advance::Function`: `(x, hyper, w, rng, t, dt) -> x'` — the Lévy increment over `[t, t+dt]`:
  advance each coefficient driver from `w[1:L]` (unconstrained), fire each jump driver on `rng`
  (seeding compartments in place), and carry the jump blocks forward. Returns the pre-flow state
  `[compartments(seeded) | coefficients(advanced) | jump blocks]` (length `layout.total_dim`).
  `rng === nothing` ⇒ no jumps (the UKF path).
- `extract::Function`: `(x) -> NamedTuple` of the **coefficient** drivers' constrained values.
- `extract_params::Function`: `(x, hyper) -> NamedTuple` (hyper ⊕ coefficients; coefficients win).
- `to_unconstrained::Function`: `(constrained) -> SVector{L}` (coefficient init).
"""
struct StochasticUpdate{
        L, AdvanceFn <: Function, ExtractFn <: Function,
        ExtractParamsFn <: Function, ToUnconstrainedFn <: Function,
    }
    advance::AdvanceFn
    extract::ExtractFn
    extract_params::ExtractParamsFn
    to_unconstrained::ToUnconstrainedFn
end

function StochasticUpdate{L}(advance, extract, extract_params, to_unconstrained) where {L}
    return StochasticUpdate{
        L, typeof(advance), typeof(extract),
        typeof(extract_params), typeof(to_unconstrained),
    }(
        advance, extract, extract_params, to_unconstrained
    )
end

"""
    build_stochastic_update(layout::StateLayout, driver_specs::Tuple) -> StochasticUpdate

Build the model's Lévy noise driver from the full list of stochastic drivers. `driver_specs` is
partitioned by `carries_state`: the **state-carrying** drivers (AR1/RW — Markov in their own
coordinate, advanced in unconstrained space from the injected noise, feeding the vector field) own
`layout.latent_range`, and the **stateless jump** drivers ([`ArrivalProcess`](@ref) — fired on the
rng, absorbing their mark into the compartments) own no slots at all. This replaces the former
`build_latent_dynamics` + the arrival wiring in `build_pf_dynamics` (ADR-0001): an arrival is now
just another driver in the list. The state-carrying drivers must match `layout.latent_names`.

# Example
```julia
driver_specs = (AR1ParamSpec(:Rt, …), ArrivalProcess(:invader, …))
layout = StateLayout(pn, driver_specs; signal_names = (:reports,))
stochastic = build_stochastic_update(layout, driver_specs)
```
"""
function build_stochastic_update(
        layout::StateLayout{N, M, K, L, S},
        driver_specs::Tuple
    ) where {N, M, K, L, S}
    # Partition on the axis the LAYOUT cares about: does the driver carry state (is it Markov in its
    # own coordinate)? State-carrying drivers (AR1/RW) own the latent slots; stateless jump drivers
    # (ArrivalProcess) own none — their effect is absorbed into the compartments.
    coeff_specs = Tuple(s for s in driver_specs if carries_state(s))
    jump_specs = Tuple(s for s in driver_specs if !carries_state(s))
    length(coeff_specs) == L || throw(
        ArgumentError("state-carrying driver count $(length(coeff_specs)) does not match layout L=$L")
    )
    for i in 1:L
        @assert coeff_specs[i].name == layout.latent_names[i] "Spec $(coeff_specs[i].name) doesn't match layout name $(layout.latent_names[i]) at position $i"
    end

    init_priors = ntuple(i -> coeff_specs[i].init, Val(L))
    constraints = map(ScalarConstraint, init_priors)   # resolved once; see `ScalarConstraint`
    latent_names = layout.latent_names
    latent_names_val = Val(latent_names)
    latent_start = first(layout.latent_range)
    n_ode = N + M
    total_dim = layout.total_dim

    # Read coefficients as a constrained NamedTuple (for the flow / measurement / jump rates).
    @inline function extract_constrained(state::AbstractVector)
        raw_unc = ntuple(i -> state[latent_start + i - 1], Val(L))
        constrained_vals = ntuple(i -> _to_constrained(constraints[i], raw_unc[i]), Val(L))
        return _named_tuple(latent_names_val, constrained_vals)
    end

    # Advance the coefficient drivers in unconstrained space (the Brownian + drift increment).
    @inline function _coeff_new(state, hyperparams, w, dt)
        raw_unc = ntuple(i -> state[latent_start + i - 1], Val(L))
        constrained_vals = ntuple(i -> _to_constrained(constraints[i], raw_unc[i]), Val(L))
        all_params = merge(hyperparams, _named_tuple(latent_names_val, constrained_vals))
        return ntuple(Val(L)) do i
            update_single(coeff_specs[i], raw_unc[i], w[i], all_params, dt, constraints[i])
        end
    end

    # THE Lévy increment: coefficients (from w) + jumps (from rng), producing the pre-flow state.
    @inline function advance(x, hyperparams, w, rng, t, dt)
        Ty = eltype(x)
        out = Vector{Ty}(undef, total_dim)
        @inbounds for i in 1:n_ode
            out[i] = x[i]                       # compartments — jumps seed them in place
        end
        # Update time-varying parameters
        new_coeff = _coeff_new(x, hyperparams, w, dt)
        @inbounds for i in 1:L
            out[latent_start + i - 1] = new_coeff[i]
        end
        # Apply jumps. Intensities read the PRE-advance latent (`x`, not `out`) on purpose: a
        # point-process compensator must be predictable — evaluated at t⁻, never at a coefficient
        # that has already absorbed this step's Brownian increment. `_coeff_new` reads `x` too, so
        # every driver sees the state at t: the noise operator is a simultaneous update and driver
        # order cannot affect what any of them reads. (The flow then runs off the advanced
        # coefficients — that asymmetry IS the noise → flow split, not an inconsistency.)
        if rng !== nothing
            _advance_jumps!(
                view(out, 1:n_ode), jump_specs, extract_constrained(x), hyperparams, dt, t, rng
            )
        end
        return out
    end

    @inline extract_params(state::AbstractVector, hyperparams::NamedTuple) =
        merge(hyperparams, extract_constrained(state))

    @inline function to_unconstrained(constrained::NamedTuple)
        return SVector{L}(
            ntuple(i -> _to_unconstrained(constraints[i], constrained[latent_names[i]]), Val(L))
        )
    end

    return StochasticUpdate{L}(advance, extract_constrained, extract_params, to_unconstrained)
end
