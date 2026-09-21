# ============================================================================
# MEASUREMENT MODEL - UKF observation function factory
# ============================================================================
#
# For use with AUGMENTED UKF where noise v ~ N(0, I) is explicit argument.
# Sigma scaling is done inside the measurement function, so R2 = I.
#
# The measurement function maps from state x to predicted observation y.
# For epidemiological data, this is typically the increment in cumulative
# observations over the time step, computed against the previous cumulative
# value passed in through `u`.
#
# ============================================================================

"""
    abstract type ObservationNoiseSpec end

Base type for observation noise specifications.
Subtypes define how to compute the predicted observation and apply noise.
"""
abstract type ObservationNoiseSpec end

"""
    NegBinomialNoise <: ObservationNoiseSpec

Negative-binomial observation noise — the **default count observation model**.

Parameterization (mean `μ`, overdispersion `φ`, reporting noise `σ`):
    Var(y) = μ + μ²/φ + (σ·μ)²

The `μ²/φ` term is the negative-binomial overdispersion (larger `φ` → closer to
Poisson; `Var → μ` as `φ → ∞`) and `(σ·μ)²` is multiplicative reporting noise.

This single spec is consumed by both filters, with different fidelity:
- **UKF** (`apply_noise`): a Gaussian approximation matching the variance above,
  using ONE unit-normal term `v[1] ~ N(0,1)`:
      `y = μ + sqrt(μ + μ²/φ + (σ·μ)²) · v[1]`
  Sigma scaling is internal, so use `R2 = I` for the measurement dimensions.
- **PF** (`observation_logpdf` / `sample_observation`): the EXACT discrete
  `NegativeBinomial(r = φ, p = φ/(φ+μ))` (mean `μ`, variance `μ + μ²/φ`). The
  `(σ·μ)²` reporting term is not carried by the discrete likelihood — a
  Poisson/NB–LogNormal mixture has no closed form; express extra dispersion by
  lowering `φ`, or push reporting noise into a latent ascertainment state.

# Fields
Each parameter is either a plain `Real` (fixed, inline literal) or a function
`(latent, hyper, t) -> Real` (dynamic — reads a constrained latent value, a
hyperparameter, or time `t`), mirroring the `rates(latent, hyper, t)` convention.
- `sigma_mult`: multiplicative reporting-noise scale `σ` (use `0.0` for none)
- `phi`: negative-binomial overdispersion `φ` (larger = closer to Poisson)

# Example
```julia
# Fixed overdispersion, no multiplicative noise
noise_spec = NegBinomialNoise(phi = 10.0)

# Learnable overdispersion — the optimizer tunes hyper.phi
noise_spec = NegBinomialNoise(phi = (latent, hyper, t) -> hyper.phi)
```

!!! note "Out of scope"
    Learning `φ` online via a Storvik update is intentionally not implemented
    here; `φ` is a fixed value or a `(latent, hyper, t)` function tuned by the
    existing hyperparameter optimizer.
"""
struct NegBinomialNoise{S, P} <: ObservationNoiseSpec
    sigma_mult::S
    phi::P
end

NegBinomialNoise(; sigma_mult = 0.0, phi) = NegBinomialNoise(sigma_mult, phi)

"""
    PoissonNoise <: ObservationNoiseSpec

Poisson observation noise via normal approximation.

Uses the approximation:
    y ~ Poisson(μ)
    y ≈ μ + sqrt(μ) * v[1]  where v[1] ~ N(0,1)

Optionally includes multiplicative noise:
    μ = true_mean * exp(σ * v[1])
    y = μ + sqrt(μ) * v[2]

With sigma scaling inside, use R2 = I.

# Fields
- `sigma_mult`: optional multiplicative noise — `nothing` for pure Poisson, else a
  plain `Real` or a function `(latent, hyper, t) -> Real`

# Example
```julia
# Pure Poisson (variance = mean)
noise_spec = PoissonNoise()

# Over-dispersed Poisson with multiplicative noise
noise_spec = PoissonNoise(0.1)
```
"""
struct PoissonNoise{S} <: ObservationNoiseSpec
    sigma_mult::S
end

# Convenience constructor for pure Poisson
PoissonNoise() = PoissonNoise(nothing)

"""
    LogNormalNoise <: ObservationNoiseSpec

Log-normal multiplicative observation noise.

For positive observations where noise is proportional to the signal:
    y = μ * exp(σ * v[1])  where v[1] ~ N(0,1)

This is equivalent to:
    log(y) ~ N(log(μ), σ²)

# Fields
- `sigma`: log-normal scale — a plain `Real` or a function `(latent, hyper, t) -> Real`

# Example
```julia
noise_spec = LogNormalNoise(0.1)
```
"""
struct LogNormalNoise{S} <: ObservationNoiseSpec
    sigma::S
end

# ============================================================================
# Signal Observation Specification
# ============================================================================

"""
    SignalObservationSpec

Specification for observing a single signal from the state vector.

# Fields
- `signal_idx::Int`: Which signal (1-based) in the StateLayout
- `noise_spec::ObservationNoiseSpec`: How to apply noise
- `mean_modifier`: Optional ascertainment/reporting factor multiplying the signal mean —
  `nothing` (no modification), a plain `Real` (constant ascertainment), or a function
  `(latent, hyper, t) -> Real` (learnable via `hyper`, time-varying via `t`, or driven
  by a latent state via `latent`; `AscertainmentPath` is one such callable)
- `name::Symbol`: Optional name for the observation (defaults to signal name)

# Examples
```julia
# Simple observation (no ascertainment)
spec = SignalObservationSpec(1, NegBinomialNoise(phi = 10.0))

# With fixed 50% ascertainment
spec = SignalObservationSpec(1, NegBinomialNoise(phi = 10.0);
    mean_modifier = 0.5)

# With time-varying ascertainment driven by a latent state
spec = SignalObservationSpec(1, NegBinomialNoise(phi = 10.0);
    mean_modifier = (latent, hyper, t) -> latent.rho)
```
"""
struct SignalObservationSpec{N <: ObservationNoiseSpec, M, B}
    signal_idx::Int
    noise_spec::N
    mean_modifier::M
    baseline::B
    name::Symbol

    function SignalObservationSpec(
            signal_idx::Int,
            noise_spec::N;
            mean_modifier::M = nothing,
            baseline::B = nothing,
            name::Symbol = Symbol("obs_", signal_idx)
        ) where {N <: ObservationNoiseSpec, M, B}
        signal_idx >= 1 || throw(ArgumentError("signal_idx must be >= 1"))
        return new{N, M, B}(signal_idx, noise_spec, mean_modifier, baseline, name)
    end
end

"""
    AggregatedSignalSpec

Specification for observing the SUM of multiple signals as one observation.

Useful for age-structured models where we track per-age-group dynamics
but only observe total hospitalizations.

# Fields
- `signal_indices::Vector{Int}`: Which signals (1-based) to sum
- `noise_spec::ObservationNoiseSpec`: How to apply noise to the sum
- `mean_modifier`: Optional ascertainment/reporting factor — `nothing`, a plain `Real`,
  or a function `(latent, hyper, t) -> Real` (see `SignalObservationSpec`)
- `name::Symbol`: Name for the aggregated observation

# Example
```julia
# Sum signals 1, 2, 3 (e.g., child, adult, elderly) into total
spec = AggregatedSignalSpec([1, 2, 3], NegBinomialNoise(phi = 10.0); name = :total_hosp)

# With time-varying ascertainment
spec = AggregatedSignalSpec([1, 2, 3], NegBinomialNoise(phi = 10.0);
    mean_modifier = (latent, hyper, t) -> latent.rho, name = :total_hosp)
```
"""
struct AggregatedSignalSpec{N <: ObservationNoiseSpec, M, B}
    signal_indices::Vector{Int}
    noise_spec::N
    mean_modifier::M
    baseline::B
    name::Symbol
    is_all_signals::Bool  # true if empty indices means "all signals"

    # Inner constructor
    function AggregatedSignalSpec(
            signal_indices::Vector{Int},
            noise_spec::N,
            mean_modifier::M,
            baseline::B,
            name::Symbol,
            is_all_signals::Bool
        ) where {N <: ObservationNoiseSpec, M, B}
        if !is_all_signals
            length(signal_indices) >= 1 ||
                throw(ArgumentError("signal_indices must have at least 1 element"))
        end
        all(>=(1), signal_indices) ||
            throw(ArgumentError("all signal_indices must be >= 1"))
        return new{N, M, B}(
            signal_indices, noise_spec, mean_modifier, baseline, name, is_all_signals
        )
    end
end

# Keyword constructor for explicit indices
function AggregatedSignalSpec(
        signal_indices::Vector{Int},
        noise_spec::N;
        mean_modifier::M = nothing,
        baseline::B = nothing,
        name::Symbol = :obs_aggregated
    ) where {N <: ObservationNoiseSpec, M, B}
    return AggregatedSignalSpec(
        signal_indices, noise_spec, mean_modifier, baseline, name, false
    )
end

# Convenience for "all signals" - indices filled in at build time
function AggregatedSignalSpec(
        noise_spec::N;
        mean_modifier::M = nothing,
        baseline::B = nothing,
        name::Symbol = :obs_total
    ) where {N <: ObservationNoiseSpec, M, B}
    return AggregatedSignalSpec(Int[], noise_spec, mean_modifier, baseline, name, true)
end

"""
    ObservationSpec

Union type for observation specifications.
Allows mixing per-signal and aggregated observations.
"""
const ObservationSpec = Union{SignalObservationSpec, AggregatedSignalSpec}

# ============================================================================
# Apply noise (dispatched on noise spec type)
# ============================================================================
#
# Measurement noise draws v are unit-normal; sigma scaling is done here.
# This means R2 = I for measurement noise dimensions.
# ============================================================================

"""
    n_noise_terms(spec::ObservationNoiseSpec)

Return the number of noise terms needed for this noise specification.
"""
n_noise_terms(::NegBinomialNoise) = 1  # single Gaussian term carrying the full variance
n_noise_terms(spec::PoissonNoise) = isnothing(spec.sigma_mult) ? 1 : 2
n_noise_terms(::LogNormalNoise) = 1

# For observation specs
n_noise_terms(spec::SignalObservationSpec) = n_noise_terms(spec.noise_spec)
n_noise_terms(spec::AggregatedSignalSpec) = n_noise_terms(spec.noise_spec)

# Resolve an observation parameter to a value: a plain `Real` is used as-is, a
# function is evaluated at (latent, hyper, t), and `nothing` stays `nothing`.
# This mirrors the `rates(latent, hyper, t)` convention used by the dynamics.
@inline _obs_value(x::Real, latent, hyper, t) = x
@inline _obs_value(f, latent, hyper, t) = f(latent, hyper, t)
@inline _obs_value(::Nothing, latent, hyper, t) = nothing

"""
    apply_noise(spec::ObservationNoiseSpec, true_mean, v, latent, hyper, t)

Apply observation noise to the true mean, given unit noise v ~ N(0, I).
Sigma scaling is done internally. Returns the noisy observation.

# Arguments
- `spec`: Noise specification
- `true_mean`: True expected observation (from state)
- `v`: Unit normal noise vector (length = n_noise_terms(spec))
- `latent`: NamedTuple of constrained latent state values
- `hyper`: NamedTuple of hyperparameter values
- `t`: Time

Noise parameters are resolved with `_obs_value`, so each is either a fixed `Real`
or a function `(latent, hyper, t) -> Real`.

# Returns
Noisy observation value
"""
function apply_noise(spec::NegBinomialNoise, true_mean, v, latent, hyper, t)
    sigma_val = _obs_value(spec.sigma_mult, latent, hyper, t)
    phi_val = _obs_value(spec.phi, latent, hyper, t)

    mu = max(true_mean, eltype(true_mean)(1.0e-6))

    # Gaussian approximation of NegBin with multiplicative reporting noise:
    # Var(y) = μ + μ²/φ + (σ·μ)², carried by a single unit-normal term v[1].
    var = mu + mu^2 / phi_val + (sigma_val * mu)^2
    y = mu + sqrt(var) * v[1]

    return max(y, zero(y))
end

function apply_noise(spec::PoissonNoise, true_mean, v, latent, hyper, t)
    if isnothing(spec.sigma_mult)
        # Pure Poisson: variance = mean
        mu = max(true_mean, eltype(true_mean)(1.0e-6))
        y = mu + sqrt(mu) * v[1]
    else
        # Over-dispersed Poisson with multiplicative noise
        sigma_val = _obs_value(spec.sigma_mult, latent, hyper, t)
        mu = max(true_mean * exp(sigma_val * v[1]), eltype(true_mean)(1.0e-6))
        y = mu + sqrt(mu) * v[2]
    end

    return max(y, zero(y))
end

function apply_noise(spec::LogNormalNoise, true_mean, v, latent, hyper, t)
    sigma_val = _obs_value(spec.sigma, latent, hyper, t)

    # Log-normal: y = μ * exp(σ * v)
    y = true_mean * exp(sigma_val * v[1])

    return max(y, eltype(y)(1.0e-6))
end

# ============================================================================
# Observation log-likelihood and sampling (particle filter)
# ============================================================================
#
# Unlike the UKF's Gaussian `apply_noise`, the particle filter can use the TRUE
# observation distribution: `observation_logpdf` scores an observed count under
# the exact Poisson / NegativeBinomial / LogNormal density (for particle
# weighting), and `sample_observation` draws from it (for simulation). Both reuse
# the same noise specs and the resolved `true_mean` produced by the measurement
# model, so the UKF and PF share one observation model.
# ============================================================================

# Distributions.jl `NegativeBinomial(r, p)` has mean r(1-p)/p and variance
# r(1-p)/p². With r = φ and p = φ/(φ+μ): mean = μ and variance = μ + μ²/φ, i.e.
# the negative-binomial core of the `NegBinomialNoise` variance (the (σ·μ)²
# reporting term has no closed-form discrete density and is dropped here).
@inline function _nb_dist(mu, phi)
    p = phi / (phi + mu)
    return NegativeBinomial(phi, p)
end

# The PF uses the EXACT distribution. PoissonNoise's optional multiplicative
# `sigma_mult` would make it a Poisson–LogNormal mixture (no closed-form density),
# and dropping it would silently remove the over-dispersion the user asked for —
# so the PF rejects it. Use NegBinomialNoise for over-dispersed counts, or
# PoissonNoise() for pure Poisson. (Resolved at compile time per concrete type, so
# pure Poisson pays nothing.)
@inline function _check_pure_poisson(spec::PoissonNoise)
    isnothing(spec.sigma_mult) || throw(
        ArgumentError(
            "PoissonNoise with sigma_mult is not supported by the particle filter's exact " *
                "likelihood/sampler (no closed-form Poisson–LogNormal mixture); use " *
                "NegBinomialNoise for over-dispersed counts, or PoissonNoise() for pure Poisson."
        )
    )
    return nothing
end

"""
    observation_logpdf(spec::ObservationNoiseSpec, y, true_mean, latent, hyper, t) -> Float64

Log-density of observation `y` under the TRUE observation distribution (the exact
discrete count distribution for `PoissonNoise`/`NegBinomialNoise`, the exact
log-normal for `LogNormalNoise`) — not the Gaussian approximation used by
`apply_noise`. `true_mean` is the ascertainment-modified mean from the
measurement model. Used to weight particles in the bootstrap filter.

`PoissonNoise` must be pure here (no `sigma_mult`); the over-dispersed variant is
rejected (no closed-form Poisson–LogNormal mixture) — use `NegBinomialNoise` for
over-dispersed counts.
"""
function observation_logpdf(spec::NegBinomialNoise, y, true_mean, latent, hyper, t)
    phi_val = _obs_value(spec.phi, latent, hyper, t)
    mu = max(true_mean, 1.0e-6)
    return logpdf(_nb_dist(mu, phi_val), round(Int, y))
end

function observation_logpdf(spec::PoissonNoise, y, true_mean, latent, hyper, t)
    _check_pure_poisson(spec)
    mu = max(true_mean, 1.0e-6)
    return logpdf(Poisson(mu), round(Int, y))
end

function observation_logpdf(spec::LogNormalNoise, y, true_mean, latent, hyper, t)
    sigma_val = _obs_value(spec.sigma, latent, hyper, t)
    mu = max(true_mean, 1.0e-6)
    # median-parameterized: log(y) ~ N(log μ, σ²), matching apply_noise (y = μ·exp(σ·v))
    return logpdf(LogNormal(log(mu), sigma_val), max(y, 1.0e-6))
end

"""
    sample_observation(spec::ObservationNoiseSpec, true_mean, latent, hyper, t, rng) -> Float64

Draw an observation from the TRUE observation distribution (discrete counts for
Poisson/NegBinomial, log-normal otherwise). Backs the particle filter's
`measurement` function and model-consistent simulation. Counts are returned as
`Float64` so observation vectors stay homogeneously typed.
"""
function sample_observation(spec::NegBinomialNoise, true_mean, latent, hyper, t, rng)
    phi_val = _obs_value(spec.phi, latent, hyper, t)
    mu = max(true_mean, 1.0e-6)
    return float(rand(rng, _nb_dist(mu, phi_val)))
end

function sample_observation(spec::PoissonNoise, true_mean, latent, hyper, t, rng)
    _check_pure_poisson(spec)
    mu = max(true_mean, 1.0e-6)
    return float(rand(rng, Poisson(mu)))
end

function sample_observation(spec::LogNormalNoise, true_mean, latent, hyper, t, rng)
    sigma_val = _obs_value(spec.sigma, latent, hyper, t)
    mu = max(true_mean, 1.0e-6)
    return rand(rng, LogNormal(log(mu), sigma_val))
end

# ============================================================================
# Helper: compute true mean for a spec
# ============================================================================

"""
    compute_true_mean(spec::SignalObservationSpec, x, accumulator_indices)

Read this signal's per-step incidence directly from its reset-accumulator
compartment — no cross-step difference, no `u`.
"""
@inline function compute_true_mean(
        spec::SignalObservationSpec, x, accumulator_indices
    )
    return max(x[accumulator_indices[spec.signal_idx]], zero(eltype(x)))
end

"""
    compute_true_mean(spec::AggregatedSignalSpec, x, accumulator_indices)

Sum the per-step incidence across the aggregated signals' reset-accumulators.
"""
@inline function compute_true_mean(
        spec::AggregatedSignalSpec, x, accumulator_indices
    )
    total = zero(eltype(x))
    for s in spec.signal_indices
        total += max(x[accumulator_indices[s]], zero(eltype(x)))
    end
    return total
end

# ============================================================================
# Measurement model factory
# ============================================================================

# Apply the optional mean modifier (ascertainment / reporting factor) to the raw
# signal mean. The modifier is `nothing`, a plain `Real`, or a function
# `(latent, hyper, t) -> Real`, mirroring the rate-function convention.
@inline _apply_mean_modifier(::Nothing, raw_mean, latent, hyper, t) = raw_mean
@inline _apply_mean_modifier(m::Real, raw_mean, latent, hyper, t) = raw_mean * m
@inline _apply_mean_modifier(f, raw_mean, latent, hyper, t) = raw_mean * f(latent, hyper, t)

# Additive observation baseline, applied AFTER the multiplicative modifier. Same three shapes.
@inline _apply_baseline(::Nothing, mean, latent, hyper, t) = mean
@inline _apply_baseline(b::Real, mean, latent, hyper, t) = mean + b
@inline _apply_baseline(f, mean, latent, hyper, t) = mean + f(latent, hyper, t)

"""
    resolve_signal_indices(spec, n_signals) -> spec

Validate a spec's signal indices against the layout, resolving an `AggregatedSignalSpec`'s
"all signals" (empty indices) into concrete ones.
"""
function resolve_signal_indices(spec::SignalObservationSpec, n_signals::Integer)
    1 <= spec.signal_idx <= n_signals || throw(
        ArgumentError("signal_idx $(spec.signal_idx) out of range [1, $n_signals]")
    )
    return spec
end

function resolve_signal_indices(spec::AggregatedSignalSpec, n_signals::Integer)
    indices = isempty(spec.signal_indices) ? collect(1:n_signals) : spec.signal_indices
    for idx in indices
        1 <= idx <= n_signals ||
            throw(ArgumentError("signal_idx $idx out of range [1, $n_signals]"))
    end
    return AggregatedSignalSpec(
        indices, spec.noise_spec, spec.mean_modifier, spec.baseline, spec.name, false
    )
end

"""
    observation_mean(spec, raw_mean, latent, hyper, t) -> Real

The observation mean is `raw_mean * mean_modifier + baseline`. The order is deliberate:
ascertainment scales infections, while the additive floor belongs to the surveillance signal. It
does not reseed transmission; see `docs/parameter-provenance.md`.
"""
@inline function observation_mean(spec, raw_mean, latent, hyper, t)
    modified = _apply_mean_modifier(spec.mean_modifier, raw_mean, latent, hyper, t)
    return _apply_baseline(spec.baseline, modified, latent, hyper, t)
end

"""
    observation_scale(spec, latent, hyper, t) -> Real

The multiplicative factor the spec applies to the raw accumulator mean at model time `t` — the
ascertainment in force at that time — so that

    observation_mean(spec, raw, latent, hyper, t) ==
        observation_scale(spec, latent, hyper, t) * raw + observation_baseline(spec, latent, hyper, t)

Reporting code that maps a filtered or predicted accumulator to a count must go through this (or
through `observation_mean`) rather than reading a hyperparameter such as `hyper.ascertainment`
directly: the modifier may be time-varying (`AscertainmentPath`) or latent-driven, and a constant
read silently disagrees with the likelihood the filter actually scored.
"""
@inline observation_scale(spec, latent, hyper, t) =
    _apply_mean_modifier(spec.mean_modifier, 1.0, latent, hyper, t)

"""
    observation_baseline(spec, latent, hyper, t) -> Real

The additive baseline the spec adds after the multiplicative modifier (`0.0` when none).
"""
@inline observation_baseline(spec, latent, hyper, t) =
    _apply_baseline(spec.baseline, 0.0, latent, hyper, t)

"""
    observation_gaussian_moments(spec, raw_mean, raw_var, latent, hyper, t) -> (; mean, var)

Gaussian moments of the observation given Gaussian moments of its accumulator — the UKF forecast
mapping from a predicted state to a predictive count. For a `NegBinomialNoise` spec,

    mean = observation_mean(spec, max(raw_mean, 0), latent, hyper, t)
    var  = scale^2 * raw_var + mean + mean^2 / phi + (sigma * mean)^2

with `scale` the spec's modifier at `t` (`observation_scale`, so a declining ascertainment is
evaluated at the horizon's own time), and `phi`, `sigma` the noise parameters at `t`. Exact when
the modifier does not depend on the state, which holds for every shipped submodel; a state-driven
modifier would need the delta-method cross term. Other noise families have no closed form here
and throw.
"""
function observation_gaussian_moments(spec, raw_mean, raw_var, latent, hyper, t)
    return _observation_gaussian_moments(
        spec.noise_spec, spec, raw_mean, raw_var, latent, hyper, t
    )
end

function _observation_gaussian_moments(
        noise::NegBinomialNoise, spec, raw_mean, raw_var, latent, hyper, t
    )
    raw = max(raw_mean, zero(raw_mean))
    scale = observation_scale(spec, latent, hyper, t)
    mu = observation_mean(spec, raw, latent, hyper, t)
    phi = _obs_value(noise.phi, latent, hyper, t)
    sigma = _obs_value(noise.sigma_mult, latent, hyper, t)
    return (; mean = mu, var = scale^2 * raw_var + mu + mu^2 / phi + (sigma * mu)^2)
end

_observation_gaussian_moments(noise, spec, raw_mean, raw_var, latent, hyper, t) = throw(
    ArgumentError(
        "observation_gaussian_moments: only NegBinomialNoise specs have a closed-form Gaussian " *
            "forecast mapping; got $(typeof(noise))"
    )
)

@inline function _extract_latent_named(
        state::AbstractVector,
        latent_names::Val,
        ::Val{L},
        ::Val{start_idx}
    ) where {L, start_idx}
    vals = ntuple(i -> state[start_idx + i], Val(L))
    return _named_tuple(latent_names, vals)
end

@inline _svector_from_tuple(::Val{N}, values::Tuple) where {N} = SVector{N}(values)

"""
    build_measurement_model(layout::StateLayout, obs_specs::Tuple{Vararg{ObservationSpec}})

Build a UKF measurement function with explicit measurement noise.

Measurement noise v ~ N(0, I) is an explicit argument, and sigma scaling is
done inside the function, so use R2 = I.

Supports three modes:
1. **Per-signal**: Each `SignalObservationSpec` observes one signal
2. **Aggregated**: `AggregatedSignalSpec` sums multiple signals into one observation
3. **Mixed**: Combine both types in the tuple

# Arguments
- `layout`: StateLayout defining state vector structure
- `obs_specs`: Tuple of ObservationSpec (SignalObservationSpec or AggregatedSignalSpec)

# Returns
A tuple `(measure, n_obs, n_noise)` where:
- `measure(x, u, p, t, v) -> SVector{n_obs}`: Measurement function
- `n_obs`: Number of observations (= length(obs_specs))
- `n_noise`: Total noise terms needed for R2 = I(n_noise)

# Examples

## Per-signal (observe each signal separately)
```julia
layout = StateLayout(core, obs, latent; signal_names=(:child, :adult))
obs_specs = (
    SignalObservationSpec(1, NegBinomialNoise(...), :child_hosp),
    SignalObservationSpec(2, NegBinomialNoise(...), :adult_hosp),
)
measure, n_obs, n_noise = build_measurement_model(layout, obs_specs)
# n_obs = 2
```

## Aggregated (sum all signals into one observation)
```julia
obs_specs = (AggregatedSignalSpec([1, 2], NegBinomialNoise(...), :total_hosp),)
measure, n_obs, n_noise = build_measurement_model(layout, obs_specs)
# n_obs = 1
```

## Mixed (some aggregated, some individual)
```julia
obs_specs = (
    AggregatedSignalSpec([1, 2], NegBinomialNoise(...), :under65_hosp),
    SignalObservationSpec(3, NegBinomialNoise(...), :over65_hosp),
)
measure, n_obs, n_noise = build_measurement_model(layout, obs_specs)
# n_obs = 2
```
"""
function build_measurement_model(
        layout::StateLayout{N, M, K, L, S},
        obs_specs::Tuple{Vararg{ObservationSpec}}
    ) where {N, M, K, L, S}
    Sobs = length(obs_specs)
    Sobs >= 1 || throw(ArgumentError("obs_specs must have at least 1 element"))

    # Validate specs and resolve "all signals" for empty AggregatedSignalSpec
    resolved_specs = map(obs_specs) do spec
        resolve_signal_indices(spec, S)
    end

    # Compute noise term indices for each spec as tuples so the returned closure
    # captures immutable configuration rather than mutable vectors.
    noise_counts = ntuple(i -> n_noise_terms(resolved_specs[i]), Val(Sobs))
    total_noise = sum(noise_counts)
    noise_starts = ntuple(
        i -> i == 1 ? 1 : 1 + sum(noise_counts[j] for j in 1:(i - 1)),
        Val(Sobs)
    )

    accumulator_indices = layout.accumulator_indices
    latent_names_val = Val(layout.latent_names)
    latent_dim_val = Val(L)
    latent_start_val = Val(N + M + K)
    obs_dim_val = Val(Sobs)

    # Build measurement function
    @inline function measure(x, u, p, t, v)

        # Extract latent as NamedTuple for param resolution
        latent_nt = _extract_latent_named(
            x,
            latent_names_val,
            latent_dim_val,
            latent_start_val
        )

        # Compute each observation
        obs = ntuple(obs_dim_val) do i
            spec = resolved_specs[i]

            # Compute raw signal mean (dispatched on spec type)
            raw_mean = compute_true_mean(spec, x, accumulator_indices)

            # Apply mean modifier (ascertainment) if present — fixed Real or
            # a function (latent, hyper, t) -> Real
            true_mean = observation_mean(spec, raw_mean, latent_nt, p, t)

            # Get noise slice for this observation
            noise_start = noise_starts[i]
            noise_end = noise_start + noise_counts[i] - 1
            v_slice = @view v[noise_start:noise_end]

            # Apply noise
            apply_noise(spec.noise_spec, true_mean, v_slice, latent_nt, p, t)
        end

        return _svector_from_tuple(obs_dim_val, obs)
    end

    return (measure = measure, n_obs = Sobs, n_noise = total_noise)
end

"""
    build_measurement_model(layout, obs_specs, latent_dynamics::StochasticUpdate)

Build measurement model with proper constrained latent extraction.

This overload takes `latent_dynamics` to enable:
1. Proper extraction of **constrained** latent values (e.g., Rt not log(Rt))
2. Using `LatentParam(:name)` in `mean_modifier` for time-varying observation scaling

# Arguments
- `layout`: StateLayout defining state structure
- `obs_specs`: Tuple of observation specifications
- `latent_dynamics`: StochasticUpdate object (same as used in build_full_dynamics)

# Example
```julia
latent_specs = (
    RWParamSpec(:obs_scale;
        init = positive_gaussian(:obs_scale, 1.0, 0.25), sigma_rate = 0.1),
)  # time-varying in log space
latent_dynamics = build_stochastic_update(layout, latent_specs)

obs_specs = (
    SignalObservationSpec(1, NegBinomialNoise(phi = 100.0);
        mean_modifier = (latent, hyper, t) -> latent.obs_scale),  # reads latent state
)

measurement, ny, nv = build_measurement_model(layout, obs_specs, latent_dynamics)
```
"""
function build_measurement_model(
        layout::StateLayout{N, M, K, L, S},
        obs_specs::Tuple{Vararg{ObservationSpec}},
        latent_dynamics::StochasticUpdate{L}
    ) where {N, M, K, L, S}
    Sobs = length(obs_specs)
    Sobs >= 1 || throw(ArgumentError("obs_specs must have at least 1 element"))

    # Validate specs and resolve "all signals" for empty AggregatedSignalSpec
    resolved_specs = map(obs_specs) do spec
        resolve_signal_indices(spec, S)
    end

    # Compute noise term indices for each spec as tuples so the returned closure
    # captures immutable configuration rather than mutable vectors.
    noise_counts = ntuple(i -> n_noise_terms(resolved_specs[i]), Val(Sobs))
    total_noise = sum(noise_counts)
    noise_starts = ntuple(
        i -> i == 1 ? 1 : 1 + sum(noise_counts[j] for j in 1:(i - 1)),
        Val(Sobs)
    )

    accumulator_indices = layout.accumulator_indices
    extract_fn = latent_dynamics.extract  # Use StochasticUpdate.extract for constrained values
    obs_dim_val = Val(Sobs)

    # Build measurement function with constrained latent extraction
    @inline function measure(x, u, p, t, v)

        # Extract latent as NamedTuple with CONSTRAINED values
        # Uses StochasticUpdate.extract which applies inverse links
        latent_nt = extract_fn(x)

        # Compute each observation
        obs = ntuple(obs_dim_val) do i
            spec = resolved_specs[i]

            # Compute raw signal mean (dispatched on spec type)
            raw_mean = compute_true_mean(spec, x, accumulator_indices)

            # Apply mean modifier (ascertainment) if present — a constrained
            # latent can drive it via the (latent, hyper, t) function form
            true_mean = observation_mean(spec, raw_mean, latent_nt, p, t)

            # Get noise slice for this observation
            noise_start = noise_starts[i]
            noise_end = noise_start + noise_counts[i] - 1
            v_slice = @view v[noise_start:noise_end]

            # Apply noise
            apply_noise(spec.noise_spec, true_mean, v_slice, latent_nt, p, t)
        end

        return _svector_from_tuple(obs_dim_val, obs)
    end

    return (measure = measure, n_obs = Sobs, n_noise = total_noise)
end

# ============================================================================
# Measurement log-likelihood factory (particle filter)
# ============================================================================

# Validate obs specs against the layout's signal count and resolve an "all
# signals" AggregatedSignalSpec to explicit indices — the same logic the
# build_measurement_model overloads apply inline.
@inline function _resolve_obs_specs(
        obs_specs::Tuple{Vararg{ObservationSpec}}, ::Val{S}
    ) where {S}
    return map(obs_specs) do spec
        resolve_signal_indices(spec, S)
    end
end

"""
    build_measurement_logpdf(layout, obs_specs, latent_dynamics) -> g(x, u, y, p, t)

Build the particle-filter measurement-likelihood function: the summed
`observation_logpdf` of the observed vector `y` across all signal specs, under
the TRUE observation distributions.

The signature `g(x, u, y, p, t) -> Float64` matches the `measurement_likelihood`
argument of `LowLevelParticleFilters.AdvancedParticleFilter`. It reuses the same
observation specs, `compute_true_mean`, and ascertainment `mean_modifier`
machinery as `build_measurement_model`, and extracts CONSTRAINED latent values
via `latent_dynamics.extract` (so a `mean_modifier` may read a latent state).

# Arguments
- `layout`: StateLayout defining the state vector structure
- `obs_specs`: Tuple of observation specifications (same as `build_measurement_model`)
- `latent_dynamics`: StochasticUpdate (same object used in `build_full_dynamics`)

# Example
```julia
obs_model = (SignalObservationSpec(1, NegBinomialNoise(phi = 100.0); name = :reports),)
g = build_measurement_logpdf(layout, obs_model, latent_dynamics)
ll = g(x, u, [42.0], hyperparams, 0.0)   # log p(y = 42 | x)
```
"""
function build_measurement_logpdf(
        layout::StateLayout{N, M, K, L, S},
        obs_specs::Tuple{Vararg{ObservationSpec}},
        latent_dynamics::StochasticUpdate{L};
        learned = nothing
    ) where {N, M, K, L, S}
    Sobs = length(obs_specs)
    Sobs >= 1 || throw(ArgumentError("obs_specs must have at least 1 element"))

    resolved_specs = _resolve_obs_specs(obs_specs, Val(S))
    accumulator_indices = layout.accumulator_indices
    extract_fn = latent_dynamics.extract
    obs_dim_val = Val(Sobs)

    @inline function logpdf_fn(x, u, y, p, t)
        # Per-particle learned hyperparameters (e.g. NB φ) override the shared `p`.
        p_eff = learned === nothing ? p : merge(p, learned.extract(x))
        latent_nt = extract_fn(x)
        lps = ntuple(obs_dim_val) do i
            spec = resolved_specs[i]
            raw_mean = compute_true_mean(spec, x, accumulator_indices)
            true_mean = observation_mean(spec, raw_mean, latent_nt, p_eff, t)
            observation_logpdf(spec.noise_spec, y[i], true_mean, latent_nt, p_eff, t)
        end
        return sum(lps)
    end

    return logpdf_fn
end

"""
    build_measurement_logpdf(layout, noise_spec::ObservationNoiseSpec, latent_dynamics)

Convenience overload for a single-signal layout.
"""
function build_measurement_logpdf(
        layout::StateLayout{N, M, K, L, S},
        noise_spec::ObservationNoiseSpec,
        latent_dynamics::StochasticUpdate{L};
        learned = nothing
    ) where {N, M, K, L, S}
    S == 1 || throw(
        ArgumentError(
            "Single noise_spec only valid for single-signal layouts (got $S signals)"
        )
    )
    return build_measurement_logpdf(
        layout, (SignalObservationSpec(1, noise_spec),), latent_dynamics; learned
    )
end

# ============================================================================
# Convenience: Single signal measurement model
# ============================================================================

"""
    build_measurement_model(layout::StateLayout, noise_spec::ObservationNoiseSpec)

Convenience method for single-signal observation.

# Example
```julia
layout = StateLayout(core, obs, latent; signal_names=(:y,))
measure, n_obs, n_noise = build_measurement_model(layout, NegBinomialNoise(...))
```
"""
function build_measurement_model(
        layout::StateLayout{N, M, K, L, S},
        noise_spec::ObservationNoiseSpec
    ) where {N, M, K, L, S}
    S == 1 || throw(
        ArgumentError(
            "Single noise_spec only valid for single-signal layouts (got $S signals)"
        )
    )

    obs_specs = (SignalObservationSpec(1, noise_spec),)
    return build_measurement_model(layout, obs_specs)
end

# ============================================================================
# Convenience: Aggregated measurement model (sum all signals)
# ============================================================================

"""
    build_measurement_model(layout::StateLayout, noise_spec::ObservationNoiseSpec, ::Val{:aggregated})

Convenience method to observe SUM of all signals as one observation.

Useful for age-structured models where dynamics are per-age-group
but we only observe total hospitalizations.

# Example
```julia
# 3 age groups, but observe total
layout = StateLayout(core, obs, latent; signal_names=(:child, :adult, :elderly))
measure, n_obs, n_noise = build_measurement_model(layout, NegBinomialNoise(...), Val(:aggregated))
# n_obs = 1 (the sum)
```
"""
function build_measurement_model(
        layout::StateLayout{N, M, K, L, S},
        noise_spec::ObservationNoiseSpec,
        ::Val{:aggregated}
    ) where {N, M, K, L, S}
    # Empty signal_indices means "all signals" - resolved at build time
    obs_specs = (AggregatedSignalSpec(noise_spec; name = :obs_total),)
    return build_measurement_model(layout, obs_specs)
end
