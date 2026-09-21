# ============================================================================
# LEARNED HYPERPARAMETERS — particles carry static hyperparams
# ============================================================================
#
# Selected hyperparameters are learned online by appending them to the particle
# state (in UNCONSTRAINED space, via their EKP prior bijectors) as a block AFTER
# the core+obs+latent state. Each particle carries its own value, which overrides
# the shared `p` when the PF builders evaluate the dynamics / measurement
# (per-particle `merge(p, extract(x))`).
#
# One `HyperUpdater` owns the complete block of hyperparameters that it updates.
# `LiuWest(priors)` therefore carries every scalar prior in the jointly updated
# block and performs one multivariate shrink-jitter refresh over that block.
#
# Storage is the flat SVector tail (no LLPF friction); the per-param structure
# lives here in the builder. This mirrors StochasticUpdate (same unconstrained
# storage + `extract` / `to_unconstrained`, reusing `_to_constrained` /
# `_named_tuple`), but the parameters are static and refreshed by the updater.
# ============================================================================

"""
    HyperUpdater

Online-learning method for a block of learned hyperparameters. Concrete subtypes
own the priors and configuration for the complete block. Currently implemented:
[`LiuWest`](@ref).
"""
abstract type HyperUpdater <: HyperparamInferenceMethod end

"""
    LiuWest(priors; discount=0.95, jitter_floor_fraction=DEFAULT_JITTER_FLOOR_FRACTION) <: HyperUpdater

Learn a block of static hyperparameters by a joint Liu-West kernel
shrink-jitter update. `priors` may be a tuple or a `NamedTuple` of scalar
`ParameterDistribution`s; a single prior is accepted as a convenience. Each
parameter occupies one unconstrained particle-state slot.

The `NamedTuple` constructor checks that every key agrees with the name carried
by its prior. `discount` must lie in `(0, 1]`, and `jitter_floor_fraction` must be
non-negative — see [`DEFAULT_JITTER_FLOOR_FRACTION`](@ref). Setting it to `0` restores the
textbook variance-preserving kernel, and with it the collapse.

`forgetting_memory_days` (a `NamedTuple` or `Dict`, parameter name => days; empty by default) turns
on **Kulhavý exponential forgetting toward the prior** for the named parameters only. Liu-West uses
a prior ONCE, to scatter the initial cloud; after that nothing refers to it, so a weakly identified
parameter is free to drift. With forgetting, every step replaces that parameter's Gaussian
approximation by the geometric mean of itself and its prior, `post^λ · prior^(1−λ)` with
`λ = exp(−dt / memory)`. The prior becomes the fixed point: with no information in the data the
cloud relaxes to the prior's mean and variance on the memory's time scale, and with information the
steady state is one copy of the prior plus about one memory's worth of data. A memory of `Inf`
is no forgetting. Names must belong to this block. The pull on the mean is precision-weighted, so
it governs a healthy cloud; a collapsed one is re-widened by the jitter floor first (see the
comment in `_build_hyperparam_updater`).
"""
struct LiuWest{H, P <: Tuple, T <: Real} <: HyperUpdater
    priors::P
    discount::T
    jitter_floor_fraction::T
    forgetting_memory_days::Dict{Symbol, Float64}
end

# `forgetting_memory_days` as given (NamedTuple or Dict, Symbol or String keys) → a validated
# `Dict{Symbol, Float64}`. Naming a parameter outside the block is an error rather than a no-op:
# a forgetting request that silently did nothing would be read off a run config as if it had.
function _forgetting_memory(raw, names)
    memory = Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in pairs(raw))
    unknown = setdiff(keys(memory), names)
    isempty(unknown) || throw(
        ArgumentError(
            "forgetting_memory_days names parameters outside the LiuWest block: " *
                "$(sort!(collect(unknown))); the block learns $names"
        )
    )
    for (name, days) in memory
        days > 0 || throw(
            ArgumentError(
                "forgetting memory for $name must be positive (Inf disables it), got $days"
            )
        )
    end
    return memory
end

function LiuWest(
        priors::Tuple; discount::Real = 0.95,
        jitter_floor_fraction::Real = DEFAULT_JITTER_FLOOR_FRACTION,
        forgetting_memory_days = (;),
    )
    isempty(priors) && throw(ArgumentError("LiuWest requires at least one prior"))
    all(prior -> prior isa ParameterDistribution, priors) || throw(
        ArgumentError("all LiuWest priors must be ParameterDistribution objects")
    )
    names = Tuple(prior_name(prior) for prior in priors)
    length(unique(names)) == length(names) || throw(
        ArgumentError("LiuWest prior names must be unique, got $names")
    )
    0 < discount <= 1 || throw(
        ArgumentError("discount must be in (0, 1], got $discount")
    )
    jitter_floor_fraction >= 0 || throw(
        ArgumentError("jitter_floor_fraction must be non-negative, got $jitter_floor_fraction")
    )
    delta, floor_fraction = promote(float(discount), float(jitter_floor_fraction))
    return LiuWest{length(priors), typeof(priors), typeof(delta)}(
        priors, delta, floor_fraction, _forgetting_memory(forgetting_memory_days, names)
    )
end

LiuWest(
    prior::ParameterDistribution; discount::Real = 0.95,
    jitter_floor_fraction::Real = DEFAULT_JITTER_FLOOR_FRACTION,
    forgetting_memory_days = (;),
) = LiuWest((prior,); discount, jitter_floor_fraction, forgetting_memory_days)

function LiuWest(
        priors::NamedTuple; discount::Real = 0.95,
        jitter_floor_fraction::Real = DEFAULT_JITTER_FLOOR_FRACTION,
        forgetting_memory_days = (;),
    )
    all(prior -> prior isa ParameterDistribution, values(priors)) || throw(
        ArgumentError("all LiuWest priors must be ParameterDistribution objects")
    )
    key_names = Tuple(keys(priors))
    prior_names = Tuple(prior_name(prior) for prior in values(priors))
    key_names == prior_names || throw(
        ArgumentError(
            "LiuWest NamedTuple keys must match the names carried by its priors; " *
                "got keys=$key_names and prior names=$prior_names"
        )
    )
    return LiuWest(
        Tuple(values(priors)); discount, jitter_floor_fraction, forgetting_memory_days
    )
end

# Replace one slot of a particle without a full copy: functional `setindex` for an
# immutable `SVector` (stack), in-place for a mutable container (e.g. a heap
# `Vector` particle, the route taken when the state outgrows `SVector` — see
# .github/particle-filter-state-representation.md).
@inline _set_slot(p::SVector, val, idx) = setindex(p, val, idx)
@inline function _set_slot(p::AbstractVector, val, idx)
    p[idx] = val
    return p
end

"""
    LearnedHyperparams{H}

A block of `H` hyperparameters learned online, carried per-particle in the
unconstrained tail of the state vector. `offset == layout.total_dim`; the block
spans `(offset+1):(offset+total_slots)`. Each parameter's value lives at
`value_slots[i]`.

# Fields
- `method`: block-level online-learning method
- `names`: `NTuple{H, Symbol}`
- `priors`: `NTuple{H, ParameterDistribution}`
- `offset::Int`, `value_slots::NTuple{H, Int}`, `total_slots::Int`
- `extract`: `(x) -> NamedTuple` of CONSTRAINED parameter values
- `to_unconstrained`: `(constrained_nt) -> SVector{H}` unconstrained values
"""
struct LearnedHyperparams{H, Method, Names, Priors, ExtractFn, ToUncFn}
    method::Method
    names::Names
    priors::Priors
    offset::Int
    value_slots::NTuple{H, Int}
    total_slots::Int
    extract::ExtractFn
    to_unconstrained::ToUncFn
end

n_learned(::LearnedHyperparams{H}) where {H} = H
hyper_range(b::LearnedHyperparams) = (b.offset + 1):(b.offset + b.total_slots)

# The learned block of every particle as an `SVector{H}` (unconstrained), and the weighted
# mean / covariance of that block. Shared by the Liu-West kernel and the `sd_ratio` diagnostic in
# the PF hyperparameter summary, so both read the cloud the same way.
function _theta_block(particles::AbstractVector, learned::LearnedHyperparams{H}) where {H}
    slots = learned.value_slots
    return [
        SVector{H}(ntuple(k -> particles[i][slots[k]], Val(H))) for i in eachindex(particles)
    ]
end

function _weighted_theta_moments(
        θs::AbstractVector{<:SVector{H}}, weights::AbstractVector
    ) where {H}
    wsum = sum(weights)
    θ̄ = sum(weights[i] .* θs[i] for i in eachindex(θs)) ./ wsum
    V = sum(weights[i] .* ((θs[i] - θ̄) * (θs[i] - θ̄)') for i in eachindex(θs)) ./ wsum
    return θ̄, V
end

"""
    build_learned_hyperparams(method::LiuWest, layout) -> LearnedHyperparams

Build the particle-state tail for all priors owned by a block-level `LiuWest`
method. Each prior contributes one unconstrained coordinate appended after the
model state, so the augmented state has `layout.total_dim + H` entries.

# Example
```julia
learned = build_learned_hyperparams(
    LiuWest((
        positive_gaussian(:R0_baseline, 1.5, 0.5),
        positive_gaussian(:phi, 20.0, 5.0),
    ); discount = 0.97),
    layout,
)
# augmented particle = [core; obs; latent; R0_baseline_unc; phi_unc]
```
"""
function build_learned_hyperparams(
        method::LiuWest{H}, layout::StateLayout
    ) where {H}
    priors = method.priors
    names = ntuple(i -> prior_name(priors[i]), Val(H))
    offset = layout.total_dim
    value_slots = ntuple(i -> offset + i, Val(H))
    total_slots = H
    names_val = Val(names)
    constraints = map(ScalarConstraint, priors)   # resolved once; see `ScalarConstraint`

    @inline function extract(x::AbstractVector)
        vals = ntuple(i -> _to_constrained(constraints[i], x[value_slots[i]]), Val(H))
        return _named_tuple(names_val, vals)
    end

    @inline function to_unconstrained(nt::NamedTuple)
        return SVector{H}(ntuple(i -> _to_unconstrained(constraints[i], nt[names[i]]), Val(H)))
    end

    return LearnedHyperparams{H, typeof(method), typeof(names), typeof(priors), typeof(extract), typeof(to_unconstrained)}(
        method, names, priors, offset, value_slots, total_slots, extract, to_unconstrained
    )
end

"""
    build_hyperparam_updater(learned; rng = Random.default_rng()) -> update!

Build the ensemble update kernel for the learned-hyperparameter block.
`update!(particles, weights)` refreshes each particle's learned-parameter slots
IN PLACE, dispatching on the block-level `HyperUpdater`.

All `LiuWest` parameters are refreshed jointly by Liu-West shrink-jitter:

    θᵢ ← a·θᵢ + (1 − a)·θ̄_w + ε,   ε ~ N(0, (1 − a²)·V_w)

with `a = (3δ − 1) / 2δ` and `θ̄_w, V_w` the weight-weighted mean/cov of the
Liu-West value-block across the ensemble (unconstrained). Shrinking toward `θ̄_w`
preserves the weighted mean, and `a² + (1 − a²) = 1` preserves the variance — so
diversity lost to resampling is restored without inflating the posterior.
`discount` δ ∈ (0,1] (≈0.95–0.99; smaller → more jitter) is read
from the `LiuWest` method stored in `learned`.

Parameters named in the method's `forgetting_memory_days` are additionally pulled toward their
prior by Kulhavý forgetting, inside the same move; `dt` is the filter step in days, which makes
the memory a rate per day rather than per step.

Call between `correct!` and `predict!` in the filter loop, on the buffer
`predict!` will propagate (`state(pf).xprev`), using post-correct `expweights(pf)`.

"""
function build_hyperparam_updater(
        learned::LearnedHyperparams{H};
        rng = Random.default_rng(), dt::Real = 1.0,
    ) where {H}
    return _build_hyperparam_updater(learned.method, learned, rng, dt)
end

function _build_hyperparam_updater(
        method::LiuWest{H}, learned::LearnedHyperparams{H}, rng, dt::Real = 1.0
    ) where {H}
    discount = method.discount
    a = (3 * discount - 1) / (2 * discount)
    h2 = 1 - a^2
    slots = learned.value_slots          # all Liu-West ⇒ one value slot each
    hv = Val(H)
    # Jitter floor — the thing that stops collapse being an ABSORBING state.
    #
    # Liu-West's jitter covariance is `h2 * V`, proportional to the cloud's own current
    # variance, because the kernel is designed to be variance-PRESERVING. It therefore cannot
    # restore variance that the resampling step destroys: weight degeneracy shrinks `V`, the
    # jitter shrinks with it, and the cloud freezes wherever it happened to be. Measured on
    # `two_strain_escape`, the θ-cloud collapsed by ~4 orders of magnitude and the estimates
    # then depended on the INITIAL cloud width rather than on the data (`escape_mean` landing
    # anywhere from 0.165 to 0.847 across initial variances), which is the signature of a frozen
    # filter rather than a tight posterior. Neither a wider start nor 5x the particles fixed it.
    #
    # The floor is a fraction of each parameter's OWN prior variance in unconstrained space, so
    # it is scale-correct per parameter (a logit-scale persistence and a log-scale rate do not
    # share a natural step size) and cannot reach zero. It replaces the previous `1e-10 * I`,
    # which was a Cholesky positive-definiteness guard — about five orders of magnitude too
    # small to act as a floor.
    floor_diag = SVector{H}(
        ntuple(
            k -> max(
                method.jitter_floor_fraction * prior_unconstrained_variance(learned.priors[k]),
                1.0e-10,
            ),
            hv,
        )
    )
    ridge = SMatrix{H, H, Float64}(I) .* 1.0e-10  # keep the jitter cholesky PD
    # Kulhavý forgetting toward the prior, per parameter. `forget[k] = 1 − λ_k` with
    # `λ_k = exp(−dt / memory_k)`, so a daily and a weekly filter forget at the same rate per day;
    # it is 0 for a parameter with no memory configured. The update is MARGINAL: parameter `k`'s
    # Gaussian approximation `N(m, v)` becomes the geometric mean of itself and its prior
    # `N(m0, v0)`, written without inverting `v` so a collapsed cloud is safe:
    #
    #     v_new = v · v0 / (λ v0 + (1 − λ) v)
    #     m_new = m + (1 − λ) · v / (λ v0 + (1 − λ) v) · (m0 − m)
    #
    # The mean shift moves every particle. A widening (`v_new > v`, the usual case) is added as
    # independent jitter, which also restores distinct support; a narrowing (a cloud wider than
    # its prior) rescales that coordinate's deviations. Either way the post-move marginal is
    # exactly `N(m_new, v_new)`. Independent jitter decorrelates slightly, as the exact joint
    # update with an independent prior would.
    #
    # What it cannot do: the pull on the mean is precision-weighted — `(1 − λ) · v / v0` per step
    # while `v << v0` — so a cloud that weight degeneracy has collapsed barely feels it. The jitter
    # floor above is what re-widens a collapsed cloud; forgetting governs a healthy one.
    forget = SVector{H}(
        ntuple(
            k -> begin
                days = get(method.forgetting_memory_days, learned.names[k], Inf)
                isfinite(days) ? -expm1(-dt / days) : 0.0
            end,
            hv,
        )
    )
    has_forgetting = any(>(0), forget)
    prior_mean = SVector{H}(ntuple(k -> prior_unconstrained_mean(learned.priors[k]), hv))
    prior_var = SVector{H}(ntuple(k -> prior_unconstrained_variance(learned.priors[k]), hv))

    function update!(particles::AbstractVector, weights::AbstractVector)
        n = length(particles)
        θs = _theta_block(particles, learned)
        θ̄, V = _weighted_theta_moments(θs, weights)
        shift = zero(θ̄)
        widen = zero(θ̄)
        scale = one.(θ̄)
        if has_forgetting
            v = diag(V)
            denom = (1 .- forget) .* prior_var .+ forget .* v
            v_new = v .* prior_var ./ denom
            shift = forget .* v ./ denom .* (prior_mean .- θ̄)
            widen = max.(v_new .- v, 0.0)
            scale = SVector{H}(
                ntuple(k -> v[k] > 0 ? sqrt(min(v_new[k] / v[k], 1.0)) : 1.0, hv)
            )
            V = V .* (scale * scale')   # ≠ V only for a coordinate wider than its prior
        end
        # `max` on the diagonal, not `+`: the floor should take over only once the cloud has
        # degenerated below it, not inflate a healthy cloud every step.
        #
        # `ridge` is retained ON TOP of the floor and is NOT redundant. A degenerate cloud goes
        # rank-deficient as well as small — parameters collapse onto a lower-dimensional manifold
        # — and in that case the diagonal can already sit above the floor while the matrix is
        # singular, so the floor lifts nothing and the Cholesky fails with `PosDefException`.
        # The ridge is the positive-definiteness guarantee; the floor is the exploration one.
        jittered = h2 .* V
        lift = Diagonal(max.(zero(floor_diag), floor_diag .- diag(jittered)) .+ widen)
        Ljit = cholesky(Symmetric(jittered + lift + ridge)).L

        for i in 1:n
            θ_new = a .* θs[i] .+ (1 - a) .* θ̄ .+ Ljit * SVector{H}(randn(rng, H))
            # Kept as a separate step so a block with no forgetting runs the expression above
            # and nothing else, bit for bit.
            has_forgetting &&
                (θ_new = θ_new .+ shift .+ a .* (scale .- 1) .* (θs[i] .- θ̄))
            p = particles[i]
            for k in 1:H
                p = _set_slot(p, θ_new[k], slots[k])
            end
            particles[i] = p
        end
        return particles
    end

    return update!
end
