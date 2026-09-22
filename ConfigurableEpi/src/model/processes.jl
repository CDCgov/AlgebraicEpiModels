# Stochastic drivers: Gaussian coefficient latents stored in unconstrained coordinates (AR1,
# random walk, integrated), the particle-only ArrivalProcess jump, and the per-step StochasticUpdate.
#
# **Dynamics ordering**:
#
# extract old constrained latents
#        ↓
# advance all coefficient processes
#        ↓
# apply optional jumps using old latents
#        ↓
# extract the newly advanced latents
#        ↓
# integrate the Petri-net ODE using those new latents
#        ↓
# produce observations

abstract type ParamSpec end
abstract type ProcessParamSpec <: ParamSpec end

"""
    FixedParam(name, value)
    HyperParam(prior)
    DerivedParam(name, formula)

How a process parameter is resolved from the merged `(hyperparams, constrained latents)`
NamedTuple: a constant, a lookup under the prior's name, or `formula(params)`.
"""
struct FixedParam{F} <: ParamSpec
    name::Symbol
    value::F
end

struct HyperParam{P <: ParameterDistribution} <: ParamSpec
    name::Symbol
    prior::P
    HyperParam(prior::P) where {P <: ParameterDistribution} = new{P}(prior_name(prior), prior)
end

struct DerivedParam{F <: Function} <: ParamSpec
    name::Symbol
    formula::F
end

get_param_value(spec::FixedParam, ::NamedTuple) = spec.value
get_param_value(spec::ParamSpec, params::NamedTuple) = params[spec.name]
get_param_value(spec::DerivedParam, params::NamedTuple) = spec.formula(params)

_as_param_spec(::Symbol, spec::ParamSpec) = spec
_as_param_spec(name::Symbol, value::Real) = FixedParam(name, float(value))
_as_param_spec(::Symbol, prior::ParameterDistribution) = HyperParam(prior)

function _check_prior_name(name::Symbol, prior::ParameterDistribution)
    prior_name(prior) == name ||
        throw(ArgumentError("prior $(prior_name(prior)) does not match process name $name"))
    return prior
end

"""
    AR1ParamSpec(name; init, mu, tau, sigma)

Mean-reverting Ornstein–Uhlenbeck latent in the unconstrained chart of `init` (its initial
prior, which also fixes the constraint): `u' | u ~ Normal(m + rho (u - m), sigma^2 (1 - rho^2))`
with `rho = exp(-dt / tau)`. `mu` is the constrained stationary mean, `tau` the correlation time
in days and `sigma` the stationary sd; each may be a `Real`, a `ParameterDistribution` (learned
under its own name) or a `ParamSpec`.
"""
struct AR1ParamSpec{M <: ParamSpec, T <: ParamSpec, S <: ParamSpec, P <: ParameterDistribution} <:
    ProcessParamSpec
    name::Symbol
    init::P
    mu::M
    tau::T
    sigma::S
end

function AR1ParamSpec(name::Symbol; init::ParameterDistribution, mu, tau, sigma)
    _check_prior_name(name, init)
    return AR1ParamSpec(
        name, init, _as_param_spec(Symbol(name, :_mu), mu),
        _as_param_spec(Symbol(name, :_tau), tau), _as_param_spec(Symbol(name, :_sigma_stat), sigma),
    )
end

"""
    RWParamSpec(name; init, sigma_rate)

Driftless random walk in the unconstrained chart; a step of `dt` days has sd
`sigma_rate * sqrt(dt)`.
"""
struct RWParamSpec{S <: ParamSpec, P <: ParameterDistribution} <: ProcessParamSpec
    name::Symbol
    init::P
    sigma::S
end

function RWParamSpec(name::Symbol; init::ParameterDistribution, sigma_rate)
    _check_prior_name(name, init)
    return RWParamSpec(name, init, _as_param_spec(Symbol(name, :_sigma_rate), sigma_rate))
end

"""
    IntegratedParamSpec(name; init, rate, per_days = 1.0)

Noise-free latent whose unconstrained coordinate advances by `rate_value * dt / per_days`, where
`rate_value` is the constrained value of the latent or hyperparameter named `rate` at the start
of the step (explicit Euler). With a random-walk `rate` the pair is an integrated Brownian motion.
A negative `per_days` integrates with the opposite sign.
"""
struct IntegratedParamSpec{P <: ParameterDistribution} <: ProcessParamSpec
    name::Symbol
    init::P
    rate::Symbol
    per_days::Float64
end

function IntegratedParamSpec(
        name::Symbol; init::ParameterDistribution, rate::Symbol, per_days::Real = 1.0,
    )
    _check_prior_name(name, init)
    rate == name && throw(ArgumentError("IntegratedParamSpec $(repr(name)) cannot integrate itself"))
    isfinite(per_days) && per_days != 0 ||
        throw(ArgumentError("per_days must be finite and nonzero, got $per_days"))
    return IntegratedParamSpec(name, init, rate, Float64(per_days))
end

# --- jump driver ------------------------------------------------------------------------------

"""
    ArrivalProcess(name; rate, mark, transition!)

A marked point process on the compartments (particle filter only). On each stochastic step it
fires with probability `1 - exp(-rate * dt)`, where `rate` is a constant or
`(x_model, latent, hyper, t) -> λ`; on firing it draws `mark(hyper, rng) -> NamedTuple` and
applies `transition!(x_model, mark, hyper)` in place. It carries no state: the compartments record
that it fired, so a single-shot arrival is a rate that reads the compartment its own jump seeds.
"""
struct ArrivalProcess{R, Mk, Tr} <: ProcessParamSpec
    name::Symbol
    rate::R
    sample_mark::Mk
    transition!::Tr
end

ArrivalProcess(name::Symbol; rate, mark, transition!) =
    ArrivalProcess(name, rate isa Function ? rate : (_x, _l, _p, _t) -> float(rate), mark, transition!)

"""
    step_arrival_probability(rate, dt) -> Float64

`1 - exp(-rate * dt)` for a constant hazard `rate >= 0` over a step `dt > 0`.
"""
@inline function step_arrival_probability(rate::Real, dt::Real)
    isfinite(rate) || throw(ArgumentError("arrival rate must be finite, got $rate"))
    rate >= 0 || throw(ArgumentError("arrival rate must be non-negative, got $rate"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive, got $dt"))
    return -expm1(-rate * dt)
end

"""
    beta_mark(; mean_key = :mark_mean, concentration = 1.0, out_key = :mark)

Mark sampler `(hyper, rng) -> (; out_key => Beta(μν, (1 - μ)ν))` with mean `μ = hyper[mean_key]`.
"""
function beta_mark(; mean_key::Symbol = :mark_mean, concentration::Real = 1.0, out_key::Symbol = :mark)
    conc = float(concentration)
    conc > 0 || throw(ArgumentError("concentration must be positive, got $concentration"))
    return function (hyper, rng)
        m = getproperty(hyper, mean_key)
        0 < m < 1 || throw(ArgumentError("$mean_key must be in (0, 1), got $m"))
        return NamedTuple{(out_key,)}((rand(rng, Beta(m * conc, (1 - m) * conc)),))
    end
end

"""
    seed_transition(model_names, from, into; size_key) -> transition!

Move `mark[size_key]` individuals from compartment `from` into `into`, capped at what `from` holds.
"""
function seed_transition(model_names, from::Symbol, into::Symbol; size_key::Symbol)
    i_from = findfirst(==(from), model_names)
    i_into = findfirst(==(into), model_names)
    i_from === nothing && throw(ArgumentError("no compartment `$from` in model_names"))
    i_into === nothing && throw(ArgumentError("no compartment `$into` in model_names"))
    return function (x_model, mark, _hyper)
        requested = getproperty(mark, size_key)
        moved = min(max(requested, zero(requested)), x_model[i_from])
        x_model[i_from] -= moved
        x_model[i_into] += moved
        return x_model
    end
end

"""
    pool_redistribute!(x_model, sources, targets, weights) -> x_model

Pool the mass in `sources`, empty them, then add `weights[k] * pool` to `targets[k]` (weights
should sum to 1). Sources and targets may overlap; the pool is read before anything is written.
"""
@inline function pool_redistribute!(x_model, sources, targets, weights)
    length(targets) == length(weights) || throw(
        DimensionMismatch("pool_redistribute!: targets ($(length(targets))) and weights ($(length(weights))) differ")
    )
    T = eltype(x_model)
    pooled = zero(T)
    @inbounds for s in sources
        pooled += x_model[s]
    end
    @inbounds for s in sources
        x_model[s] = zero(T)
    end
    @inbounds for k in eachindex(targets)
        x_model[targets[k]] += weights[k] * pooled
    end
    return x_model
end

"""
    pro_rata_move!(x_model, sources, targets, amount) -> x_model

Move up to `amount` individuals out of `sources[k]` into `targets[k]`, split pro rata by each
source's occupancy and capped at what is available. `sources` and `targets` must be disjoint.
"""
@inline function pro_rata_move!(x_model, sources, targets, amount)
    length(sources) == length(targets) || throw(
        DimensionMismatch("pro_rata_move!: sources ($(length(sources))) and targets ($(length(targets))) differ")
    )
    T = eltype(x_model)
    requested = max(T(amount), zero(T))
    pool = zero(T)
    @inbounds for s in sources
        pool += x_model[s]
    end
    (requested > zero(T) && pool > zero(T)) || return x_model
    moved = min(requested, pool)
    @inbounds for k in eachindex(sources)
        available = x_model[sources[k]]
        take = min(moved * available / pool, available)
        x_model[sources[k]] -= take
        x_model[targets[k]] += take
    end
    return x_model
end

"""
    carries_state(spec) -> Bool
    supports_gaussian_filter(spec) -> Bool

Whether a driver claims latent slots in the [`StateLayout`](@ref), and whether a Gaussian (UKF)
filter can propagate it. Both hold for the coefficient processes and fail for
[`ArrivalProcess`](@ref), whose fired/not-fired mixture no single Gaussian represents.
"""
carries_state(::ParamSpec) = true
carries_state(::ArrivalProcess) = false
supports_gaussian_filter(::ParamSpec) = true
supports_gaussian_filter(::ArrivalProcess) = false

"""
    assert_gaussian_filter_compatible(specs)

Throw if any driver is particle-only.
"""
assert_gaussian_filter_compatible(spec::ParamSpec) = assert_gaussian_filter_compatible((spec,))
function assert_gaussian_filter_compatible(specs)
    bad = [typeof(s) for s in specs if !supports_gaussian_filter(s)]
    isempty(bad) || throw(
        ArgumentError("particle-only process(es) $bad cannot be propagated by a Gaussian filter; use PF")
    )
    return nothing
end

"""
    advance_arrival!(process, x_model, latent, hyper, dt, t, rng) -> x_model

Fire one jump driver in place with probability `step_arrival_probability(rate, dt)`.
"""
@inline function advance_arrival!(process::ArrivalProcess, x_model, latent, hyper, dt, t, rng)
    p_fire = step_arrival_probability(process.rate(x_model, latent, hyper, t), dt)
    if p_fire > 0 && rand(rng) < p_fire
        process.transition!(x_model, process.sample_mark(hyper, rng), hyper)
    end
    return x_model
end

@inline _advance_jumps!(x_model, ::Tuple{}, latent, hyper, dt, t, rng) = x_model
@inline function _advance_jumps!(x_model, jumps::Tuple, latent, hyper, dt, t, rng)
    advance_arrival!(first(jumps), x_model, latent, hyper, dt, t, rng)
    return _advance_jumps!(x_model, Base.tail(jumps), latent, hyper, dt, t, rng)
end

# --- constraint maps and one-step updates -----------------------------------------------------

# A prior's two constraint maps resolved once, so the per-particle transform is a static call.
struct ScalarConstraint{F, G}
    to_constrained::F
    to_unconstrained::G
end

function ScalarConstraint(prior::ParameterDistribution)
    c = only(get_all_constraints(prior))
    return ScalarConstraint(c.unconstrained_to_constrained, c.constrained_to_unconstrained)
end

@inline _to_constrained(c::ScalarConstraint, u) = c.to_constrained(u)
@inline function _to_unconstrained(c::ScalarConstraint, x)
    u = c.to_unconstrained(x)
    isfinite(u) || throw(
        ArgumentError("non-finite unconstrained value for $x: a bounded latent cannot start on its boundary")
    )
    return u
end
_to_constrained(prior::ParameterDistribution, u) = _to_constrained(ScalarConstraint(prior), u)
_to_unconstrained(prior::ParameterDistribution, x) = _to_unconstrained(ScalarConstraint(prior), x)

"""
    ou_step(tau, dt) -> (rho, innovation_factor)

Exact one-step OU transition, `rho = exp(-dt / tau)` and `sqrt(1 - rho^2)`, via `expm1` so a very
long `tau` does not cancel to zero.
"""
@inline function ou_step(tau::Real, dt::Real)
    isfinite(tau) && tau > 0 ||
        throw(ArgumentError("OU correlation time must be finite and positive, got tau = $tau"))
    x = dt / tau
    return exp(-x), sqrt(-expm1(-2x))
end

"""
    update_single(spec, old_unc, w, params, dt[, constraint])

Advance one latent's unconstrained coordinate over `dt` days given unit noise `w`. `params`
merges the hyperparameters with every latent's constrained value at the start of the step;
`constraint` is `spec.init`'s pre-resolved `ScalarConstraint`.
"""
update_single(spec::RWParamSpec, old_unc, w, params::NamedTuple, dt::Real, constraint = nothing) =
    old_unc + get_param_value(spec.sigma, params) * sqrt(dt) * w

function update_single(
        spec::AR1ParamSpec, old_unc, w, params::NamedTuple, dt::Real,
        constraint = ScalarConstraint(spec.init),
    )
    rho, innovation = ou_step(get_param_value(spec.tau, params), dt)
    mu = _to_unconstrained(constraint, get_param_value(spec.mu, params))
    return mu + rho * (old_unc - mu) + get_param_value(spec.sigma, params) * innovation * w
end

update_single(spec::IntegratedParamSpec, old_unc, w, params::NamedTuple, dt::Real, constraint = nothing) =
    old_unc + params[spec.rate] * dt / spec.per_days

"""
    StochasticUpdate{L}

The model's per-step stochastic driver over `L` coefficient latents, from
[`build_stochastic_update`](@ref):

- `advance(x, hyper, w, rng, t, dt)`: the pre-flow state after advancing each coefficient from
  unit noise `w[1:L]` and firing each jump driver on `rng` (`rng === nothing` skips jumps).
- `extract(x)`: the coefficients' constrained values as a NamedTuple.
- `extract_params(x, hyper)`: `merge(hyper, extract(x))`.
- `to_unconstrained(constrained::NamedTuple)`: the coefficients as an unconstrained `SVector{L}`.
"""
struct StochasticUpdate{L, A, E, EP, U}
    advance::A
    extract::E
    extract_params::EP
    to_unconstrained::U
    n_jumps::Int
end

StochasticUpdate{L}(advance, extract, extract_params, to_unconstrained, n_jumps::Integer) where {L} =
    StochasticUpdate{L, typeof(advance), typeof(extract), typeof(extract_params), typeof(to_unconstrained)}(
    advance, extract, extract_params, to_unconstrained, n_jumps
)

"""
    build_stochastic_update(layout, driver_specs) -> StochasticUpdate

Split `driver_specs` into state-carrying coefficient drivers (which must match
`layout.latent_names` in order) and stateless jump drivers, and build the step driver. Jumps read
the pre-step coefficients; the flow then runs on the advanced ones.
"""
function build_stochastic_update(layout::StateLayout{N, M, L}, driver_specs::Tuple) where {N, M, L}
    coeffs = Tuple(s for s in driver_specs if carries_state(s))
    jumps = Tuple(s for s in driver_specs if !carries_state(s))
    names = Tuple(s.name for s in coeffs)
    names == layout.latent_names || throw(
        ArgumentError("state-carrying drivers $names do not match the layout's latents $(layout.latent_names)")
    )
    constraints = map(s -> ScalarConstraint(s.init), coeffs)
    names_val = Val(names)
    start = first(layout.latent_range)
    n_ode = N + M
    total = layout.total_dim

    @inline extract(x::AbstractVector) = _named_tuple(
        names_val, ntuple(i -> _to_constrained(constraints[i], x[start + i - 1]), Val(L))
    )

    @inline function advance(x, hyper, w, rng, t, dt)
        out = Vector{eltype(x)}(undef, total)
        @inbounds for i in 1:n_ode
            out[i] = x[i]
        end
        latent = extract(x)
        params = merge(hyper, latent)
        advanced = ntuple(
            i -> update_single(coeffs[i], x[start + i - 1], w[i], params, dt, constraints[i]), Val(L)
        )
        @inbounds for i in 1:L
            out[start + i - 1] = advanced[i]
        end
        rng === nothing || _advance_jumps!(view(out, 1:n_ode), jumps, latent, hyper, dt, t, rng)
        return out
    end

    @inline extract_params(x::AbstractVector, hyper::NamedTuple) = merge(hyper, extract(x))

    @inline to_unconstrained(constrained::NamedTuple) = SVector{L}(
        ntuple(i -> _to_unconstrained(constraints[i], constrained[names[i]]), Val(L))
    )

    return StochasticUpdate{L}(advance, extract, extract_params, to_unconstrained, length(jumps))
end
