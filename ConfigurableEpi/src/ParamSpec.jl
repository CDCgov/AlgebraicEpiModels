# ============================================================================
# Parameter Specification Types
# ============================================================================

"""
    ParamSpec

Abstract base type for parameter specifications.
Determines how a parameter is handled during inference.
"""
abstract type ParamSpec end

abstract type ProcessParamSpec <: ParamSpec end

"""
    LatentParam <: ParamSpec

Reference to a latent state variable by name. Used to read time-varying
values from the current state when computing observations or derived quantities.

Unlike ProcessParamSpec types (AR1ParamSpec, RWParamSpec), this does not define
dynamics - it just provides a way to reference latent values by name.

# Fields
- `name::Symbol`: Name of the latent variable to look up
"""
struct LatentParam <: ParamSpec
    name::Symbol
end

"""
    FixedParam <: ParamSpec

A parameter fixed at a specific value (not inferred).

# Fields
- `value`: The fixed value of the parameter.
"""
struct FixedParam{F} <: ParamSpec
    name::Symbol
    value::F
end

struct HyperParam{P <: ParameterDistribution} <: ParamSpec
    name::Symbol
    prior::P
    function HyperParam(prior::P) where {P <: ParameterDistribution}
        return new{P}(prior_name(prior), prior)
    end
end

"""
    HyperParamRW <: ParamSpec

A hyperparameter that evolves as a random walk during filtering.

The process noise Q for this parameter is computed adaptively from the current
posterior covariance: Q_θ = adaptation_rate × P_θ, following Wan & van der Merwe (2000).
The adaptation_rate is passed at runtime to the filter, not stored here.

# Fields
- `name::Symbol`: Parameter name
- `prior::ParameterDistribution`: EKP prior distribution for initialization

The constraint lives inside `prior`, so there is no separate transform field here.
"""
struct HyperParamRW{P <: ParameterDistribution} <: ParamSpec
    name::Symbol
    prior::P
    function HyperParamRW(prior::P) where {P <: ParameterDistribution}
        return new{P}(prior_name(prior), prior)
    end
end

"""
    AR1ParamSpec(name; init, mu, tau, sigma)

A mean-reverting latent process using the exact Ornstein–Uhlenbeck discretisation in the
unconstrained coordinate:

    du = -(u - m)/tau * dt  +  sigma * sqrt(2/tau) * dW          (u = the UNCONSTRAINED coordinate)

sampled at step `dt`, which gives exactly

    u_next | u  ~  Normal( m + rho*(u - m),  sigma^2 * (1 - rho^2) ),    rho = exp(-dt/tau)

`init` fixes the constrained-space prior and bijector; `mu` is the constrained stationary mean,
`tau` the correlation time in days, and `sigma` the stationary standard deviation in the
unconstrained chart. The three process arguments accept a `Real`, `ParameterDistribution`, or
`ParamSpec`.

Construction is keyword-only so the retired positional `(mu, rho, innovation_sigma)` form cannot
silently reinterpret `rho` as a time in days. The parameterisation rationale is recorded in
`docs/parameter-provenance.md` and `docs/submodel-engineering-record.md`.
"""
struct AR1ParamSpec{
        M <: ParamSpec, T <: ParamSpec, S <: ParamSpec, P <: ParameterDistribution,
    } <:
    ProcessParamSpec
    name::Symbol
    init::P    # Initial state prior in constrained space; also defines the bijector/domain
    mu::M      # Stationary mean in the constrained domain
    tau::T     # Correlation time in DAYS (same clock as the filter's dt)
    sigma::S   # STATIONARY sd in the unconstrained domain
end

"""
    RWParamSpec(name; init, sigma_rate)

A driftless random walk in the unconstrained chart. `sigma_rate` is the per-sqrt-day diffusion
coefficient, so a step of length `dt` has sd `sigma_rate * sqrt(dt)`.
"""
struct RWParamSpec{S <: ParamSpec, P <: ParameterDistribution} <: ProcessParamSpec
    name::Symbol
    init::P     # Initial state prior in constrained space; also defines the bijector/domain
    sigma::S    # Diffusion coefficient per sqrt(day) in the unconstrained domain
end

# Fixed and hyper parameter constructors

function _validate_process_prior_name(name::Symbol, prior::ParameterDistribution)
    prior_name(prior) == name ||
        throw(
        ArgumentError(
            "ParameterDistribution name $(prior_name(prior)) does not match ParamSpec name $name"
        )
    )
    return prior
end

# AR and RW parameter constructors

"""
    _as_param_spec(fallback_name, value) -> ParamSpec

Lift a process-parameter argument to a `ParamSpec`. A `Real` becomes a `FixedParam` under
`fallback_name`; a `ParameterDistribution` becomes a `HyperParam` under **its own** name (which is
what lets `geographic_seir` mint one prior per location); a `ParamSpec` passes through.
"""
_as_param_spec(::Symbol, spec::ParamSpec) = spec
_as_param_spec(fallback::Symbol, value::Real) = FixedParam(fallback, float(value))
_as_param_spec(::Symbol, prior::ParameterDistribution) = HyperParam(prior)

function AR1ParamSpec(name::Symbol; init::ParameterDistribution, mu, tau, sigma)
    _validate_process_prior_name(name, init)
    mu_spec = _as_param_spec(Symbol(name, :_mu), mu)
    tau_spec = _as_param_spec(Symbol(name, :_tau), tau)
    sigma_spec = _as_param_spec(Symbol(name, :_sigma_stat), sigma)
    return AR1ParamSpec{
        typeof(mu_spec), typeof(tau_spec), typeof(sigma_spec), typeof(init),
    }(
        name, init, mu_spec, tau_spec, sigma_spec
    )
end

function RWParamSpec(name::Symbol; init::ParameterDistribution, sigma_rate)
    _validate_process_prior_name(name, init)
    spec = _as_param_spec(Symbol(name, :_sigma_rate), sigma_rate)
    return RWParamSpec{typeof(spec), typeof(init)}(name, init, spec)
end

# The former positional API described `sigma` per filter step. The continuous-time
# parameterisation instead requires an explicit per-sqrt-day rate, so accepting the old spelling
# would silently change its meaning whenever `dt != 1`.
function RWParamSpec(
        name::Symbol,
        ::Union{Real, ParameterDistribution};
        init::ParameterDistribution
    )
    throw(
        ArgumentError(
            "positional RWParamSpec(name, sigma; init) is retired because `sigma` meant " *
                "per-step noise; use RWParamSpec($(repr(name)); init, sigma_rate = ...) " *
                "to specify the per-sqrt-day diffusion rate explicitly"
        )
    )
end

"""
    IntegratedParamSpec(name; init, rate, per_days = 1.0)

A latent whose UNCONSTRAINED coordinate is the time integral of another quantity. Over a step of
`dt` days it advances by `rate_value * dt / per_days`, where `rate_value` is the constrained value,
at the START of the step, of the latent or hyperparameter named `rate`. It has no noise of its own:
its unit-noise draw is ignored.

`per_days` converts the rate's time unit to the filter clock, which is days: a per-year rate uses
`per_days = 365.25`. A negative `per_days` integrates the rate with the opposite sign, so a DECLINE
rate lowers the level.

Paired with an `RWParamSpec` rate it is an **integrated Brownian motion** (a local linear
trend): the level extrapolates along its current slope, and the slope's diffusion sets how fast that
line may bend. Every driver reads the state at the start of the step, so the pair is the
explicit-Euler form. The exact transition also carries level noise of variance `sigma^2 dt^3 / 3`
correlated with the slope's; the per-driver noise interface cannot express that, and at daily steps
with a slowly bending trend it is negligible.

Because the level is a STATE, a change to the rate only alters FUTURE increments. That is what
makes it suitable for a one-pass filter, where a static path parameter is not: a path parameter's
leverage on the observation grows with time, and moving it silently rewrites a history the
particle's other states were built under.
"""
struct IntegratedParamSpec{P <: ParameterDistribution} <: ProcessParamSpec
    name::Symbol
    init::P          # Initial state prior in constrained space; also defines the bijector/domain
    rate::Symbol     # Name of the latent/hyperparameter integrated, read in constrained space
    per_days::Float64
end

function IntegratedParamSpec(
        name::Symbol; init::ParameterDistribution, rate::Symbol, per_days::Real = 1.0,
    )
    _validate_process_prior_name(name, init)
    rate == name && throw(
        ArgumentError("IntegratedParamSpec $(repr(name)) cannot integrate itself")
    )
    isfinite(per_days) && per_days != 0 || throw(
        ArgumentError("per_days must be finite and nonzero, got $per_days")
    )
    return IntegratedParamSpec{typeof(init)}(name, init, rate, Float64(per_days))
end

# ============================================================================
# ArrivalProcess — a marked point process (the particle-only latent sibling)
# ============================================================================

"""
    ArrivalProcess(name; rate, mark, transition!)

Declarative spec for a **stochastic arrival** — a marked point process in time, the
particle-filter counterpart of the Gaussian latent processes `AR1ParamSpec` /
`RWParamSpec`. On every *stochastic* propagation step each particle may independently
fire an arrival; on firing it draws a **mark** (the sampled "characteristic") and applies a jump
to the model state.

An arrival is characterised by

1. a **rate of arrival** `rate` — a constant intensity, or a function `(x, latent, hyper, t) -> λ`
   of the current model state `x` (the compartments, indexable by their `ode_names` position), the
   constrained `latent` NamedTuple, time `t`, and (possibly learnable) hyperparameters such as
   `hyper.arrival_rate`; the per-step firing probability is
   [`step_arrival_probability`](@ref)`(λ, dt) = 1 - exp(-λ·dt)`, and
2. a **transition on the model state** — a mark sampler `mark(hyper, rng) -> NamedTuple` plus the
   jump `transition!(x_model, mark, hyper)` it induces, applied in place to the compartments.

An `ArrivalProcess` is **stateless**: it claims no slots in the [`StateLayout`](@ref)
(`carries_state(::ArrivalProcess) == false`). The mark is **absorbed-only** — applied once to the
compartments and never read back — so the model state *already records* that the arrival happened;
a separate `fired` flag would be redundant bookkeeping. Consequently **self-excitation is just a
state-reading rate**: a single-shot ("one new variant") arrival returns `0` once the compartment its
jump seeds is populated, e.g. `(x, l, p, t) -> x[i_invader] > 0 ? 0.0 : p.arrival_rate`. Left
ungated, the rate defines a genuine **recurring** (memoryless / Poisson-thinned) point process. The
process stays a pure function of state — nothing is mutated on the side.

Because a fired/not-fired mixture with a random mark cannot be represented by a single Gaussian, an
`ArrivalProcess` is **particle-filter only**: `supports_gaussian_filter(::ArrivalProcess) == false`,
and a UKF build over a model that declares one is rejected by
[`assert_gaussian_filter_compatible`](@ref). Pass it in the `driver_specs` list to
[`build_stochastic_update`](@ref) — it is a jump driver of the one Lévy noise driver, not a special
case; its learnable hyperparameters (the rate, the mark's mean, …) are ordinary [`LiuWest`](@ref)
parameters, learned online exactly like any other.

# Arguments
- `name::Symbol`: identifier for this arrival process (e.g. `:invader`).
- `rate`: the arrival intensity λ — a constant, or `(x, latent, hyper, t) -> λ`. Firing multiplicity
  lives here: read the state for single-shot, or leave it ungated for a recurring process.
- `mark`: the mark sampler `(hyper, rng) -> NamedTuple` drawing the sampled characteristic(s).
- `transition!`: the in-place jump `(x_model, mark, hyper)` applied to the compartments on firing
  (the mark is absorbed here).

# Example
```julia
process = ArrivalProcess(
    :invader;
    rate = (x, latent, p, t) -> p.arrival_rate,            # recurring; arrival_rate learnable via LiuWest
    mark = (p, rng) -> (import_size = p.import_mean * (0.6 + 0.8rand(rng)),),
    transition! = seed_transition(ode_names(layout), :S, :I; size_key = :import_size),
)
```
"""
struct ArrivalProcess{R, Mk, Tr} <: ProcessParamSpec
    name::Symbol
    rate::R
    sample_mark::Mk
    transition!::Tr
end

function ArrivalProcess(
        name::Symbol;
        rate,
        mark,
        transition!,
    )
    # A constant intensity is accepted as sugar for the (x, latent, hyper, t) -> λ form.
    rate_fn = rate isa Function ? rate : (_x, _l, _p, _t) -> float(rate)
    return ArrivalProcess(name, rate_fn, mark, transition!)
end

"""
    DerivedParam <: ParamSpec

A parameter derived from other parameters via a formula.

The formula takes the full parameter structure θ and computes the derived value.
This allows natural access to dependencies via `θ[:name]` or `θ.name`.

# Fields
- `formula::Function`: Function `θ -> value` to compute the derived parameter.

# Examples
```julia
# β = R0 * γ (transmission rate from R0 and recovery rate)
DerivedParam(:beta, θ -> θ[:R0] * θ[:gamma])

# More complex derived parameter
DerivedParam(:beta_complex, θ -> θ[:R0] * θ[:Rt_modifier] * θ[:gamma] / 1000)
```
"""
struct DerivedParam{Func <: Function} <: ParamSpec
    name::Symbol
    formula::Func
end
