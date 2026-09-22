# Observation model: noise families, per-signal specs, and the measurement, likelihood and
# sampling functions built from them. Latents reach the specs constrained via `stochastic.extract`.

abstract type ObservationNoiseSpec end

"""
    NegBinomialNoise(; phi, sigma_mult = 0.0)

Count noise with `Var(y) = μ + μ²/φ + (σμ)²`. The UKF uses the Gaussian approximation with one
unit-normal term; the particle filter uses the exact `NegativeBinomial(φ, φ/(φ+μ))`, which drops
the `(σμ)²` reporting term. Each parameter is a `Real` or a function `(latent, hyper, t) -> Real`.
"""
struct NegBinomialNoise{S, P} <: ObservationNoiseSpec
    sigma_mult::S
    phi::P
end
NegBinomialNoise(; sigma_mult = 0.0, phi) = NegBinomialNoise(sigma_mult, phi)

"""
    PoissonNoise(sigma_mult = nothing)

Poisson count noise `y ≈ μ + sqrt(μ) v`; with `sigma_mult` the mean is first perturbed by
`exp(σ v₁)` (UKF only: the particle filter rejects the mixture).
"""
struct PoissonNoise{S} <: ObservationNoiseSpec
    sigma_mult::S
end
PoissonNoise() = PoissonNoise(nothing)

"""
    LogNormalNoise(sigma)

Multiplicative noise `y = μ exp(σ v)`.
"""
struct LogNormalNoise{S} <: ObservationNoiseSpec
    sigma::S
end

"""
    SignalObservationSpec(signal_idx, noise; mean_modifier = nothing, baseline = nothing, name)
    AggregatedSignalSpec(signal_indices, noise; mean_modifier = nothing, baseline = nothing, name)
    AggregatedSignalSpec(noise; ...)                       # the sum of every signal

One observation: a signal's reset accumulator, or the sum over several. Its mean is
`raw * mean_modifier + baseline`, where each of the two is `nothing`, a `Real` or a function
`(latent, hyper, t) -> Real` (an [`AscertainmentPath`](@ref), say).
"""
struct SignalObservationSpec{N <: ObservationNoiseSpec, M, B}
    signal_idx::Int
    noise_spec::N
    mean_modifier::M
    baseline::B
    name::Symbol
    function SignalObservationSpec(
            signal_idx::Int, noise_spec::N;
            mean_modifier::M = nothing, baseline::B = nothing, name::Symbol = Symbol("obs_", signal_idx),
        ) where {N <: ObservationNoiseSpec, M, B}
        signal_idx >= 1 || throw(ArgumentError("signal_idx must be >= 1"))
        return new{N, M, B}(signal_idx, noise_spec, mean_modifier, baseline, name)
    end
end

struct AggregatedSignalSpec{N <: ObservationNoiseSpec, M, B}
    signal_indices::Vector{Int}
    noise_spec::N
    mean_modifier::M
    baseline::B
    name::Symbol
    is_all_signals::Bool
    function AggregatedSignalSpec(
            signal_indices::Vector{Int}, noise_spec::N, mean_modifier::M, baseline::B,
            name::Symbol, is_all_signals::Bool,
        ) where {N <: ObservationNoiseSpec, M, B}
        is_all_signals || !isempty(signal_indices) ||
            throw(ArgumentError("signal_indices must have at least one element"))
        all(>=(1), signal_indices) || throw(ArgumentError("all signal_indices must be >= 1"))
        return new{N, M, B}(signal_indices, noise_spec, mean_modifier, baseline, name, is_all_signals)
    end
end

AggregatedSignalSpec(
    signal_indices::Vector{Int}, noise_spec::ObservationNoiseSpec;
    mean_modifier = nothing, baseline = nothing, name::Symbol = :obs_aggregated,
) = AggregatedSignalSpec(signal_indices, noise_spec, mean_modifier, baseline, name, false)

AggregatedSignalSpec(
    noise_spec::ObservationNoiseSpec; mean_modifier = nothing, baseline = nothing, name::Symbol = :obs_total,
) = AggregatedSignalSpec(Int[], noise_spec, mean_modifier, baseline, name, true)

const ObservationSpec = Union{SignalObservationSpec, AggregatedSignalSpec}

n_noise_terms(::NegBinomialNoise) = 1
n_noise_terms(spec::PoissonNoise) = spec.sigma_mult === nothing ? 1 : 2
n_noise_terms(::LogNormalNoise) = 1
n_noise_terms(spec::ObservationSpec) = n_noise_terms(spec.noise_spec)

# A noise parameter: a constant or a `(latent, hyper, t)` function.
@inline _obs_value(x::Real, latent, hyper, t) = x
@inline _obs_value(f, latent, hyper, t) = f(latent, hyper, t)

"""
    apply_noise(noise, mean, v, latent, hyper, t)

Gaussian-approximation observation given unit noise `v` (the UKF measurement).
"""
function apply_noise(spec::NegBinomialNoise, true_mean, v, latent, hyper, t)
    sigma = _obs_value(spec.sigma_mult, latent, hyper, t)
    phi = _obs_value(spec.phi, latent, hyper, t)
    mu = max(true_mean, eltype(true_mean)(1.0e-6))
    y = mu + sqrt(mu + mu^2 / phi + (sigma * mu)^2) * v[1]
    return max(y, zero(y))
end

function apply_noise(spec::PoissonNoise, true_mean, v, latent, hyper, t)
    floor = eltype(true_mean)(1.0e-6)
    if spec.sigma_mult === nothing
        mu = max(true_mean, floor)
        y = mu + sqrt(mu) * v[1]
    else
        sigma = _obs_value(spec.sigma_mult, latent, hyper, t)
        mu = max(true_mean * exp(sigma * v[1]), floor)
        y = mu + sqrt(mu) * v[2]
    end
    return max(y, zero(y))
end

function apply_noise(spec::LogNormalNoise, true_mean, v, latent, hyper, t)
    y = true_mean * exp(_obs_value(spec.sigma, latent, hyper, t) * v[1])
    return max(y, eltype(y)(1.0e-6))
end

@inline _nb_dist(mu, phi) = NegativeBinomial(phi, phi / (phi + mu))

@inline function _check_pure_poisson(spec::PoissonNoise)
    spec.sigma_mult === nothing || throw(
        ArgumentError(
            "PoissonNoise with sigma_mult has no closed-form likelihood for the particle filter; " *
                "use NegBinomialNoise for over-dispersed counts",
        )
    )
    return nothing
end

"""
    observation_logpdf(noise, y, mean, latent, hyper, t)
    sample_observation(noise, mean, latent, hyper, t, rng)

Exact log-density and draw of an observation (particle weighting and simulation).
"""
function observation_logpdf(spec::NegBinomialNoise, y, true_mean, latent, hyper, t)
    phi = _obs_value(spec.phi, latent, hyper, t)
    return logpdf(_nb_dist(max(true_mean, 1.0e-6), phi), round(Int, y))
end
function observation_logpdf(spec::PoissonNoise, y, true_mean, latent, hyper, t)
    _check_pure_poisson(spec)
    return logpdf(Poisson(max(true_mean, 1.0e-6)), round(Int, y))
end
function observation_logpdf(spec::LogNormalNoise, y, true_mean, latent, hyper, t)
    sigma = _obs_value(spec.sigma, latent, hyper, t)
    return logpdf(LogNormal(log(max(true_mean, 1.0e-6)), sigma), max(y, 1.0e-6))
end

function sample_observation(spec::NegBinomialNoise, true_mean, latent, hyper, t, rng)
    phi = _obs_value(spec.phi, latent, hyper, t)
    return float(rand(rng, _nb_dist(max(true_mean, 1.0e-6), phi)))
end
function sample_observation(spec::PoissonNoise, true_mean, latent, hyper, t, rng)
    _check_pure_poisson(spec)
    return float(rand(rng, Poisson(max(true_mean, 1.0e-6))))
end
function sample_observation(spec::LogNormalNoise, true_mean, latent, hyper, t, rng)
    sigma = _obs_value(spec.sigma, latent, hyper, t)
    return rand(rng, LogNormal(log(max(true_mean, 1.0e-6)), sigma))
end

# This step's incidence, read from the reset accumulator(s) and floored at zero.
@inline compute_true_mean(spec::SignalObservationSpec, x, accumulators) =
    max(x[accumulators[spec.signal_idx]], zero(eltype(x)))
@inline function compute_true_mean(spec::AggregatedSignalSpec, x, accumulators)
    total = zero(eltype(x))
    for s in spec.signal_indices
        total += max(x[accumulators[s]], zero(eltype(x)))
    end
    return total
end

@inline _modify(::Nothing, raw, latent, hyper, t) = raw
@inline _modify(m::Real, raw, latent, hyper, t) = raw * m
@inline _modify(f, raw, latent, hyper, t) = raw * f(latent, hyper, t)
@inline _offset(::Nothing, mean, latent, hyper, t) = mean
@inline _offset(b::Real, mean, latent, hyper, t) = mean + b
@inline _offset(f, mean, latent, hyper, t) = mean + f(latent, hyper, t)

"""
    observation_mean(spec, raw, latent, hyper, t)      # raw * modifier + baseline
    observation_scale(spec, latent, hyper, t)          # the modifier at `t`
    observation_baseline(spec, latent, hyper, t)       # the baseline at `t`

Reporting code must map accumulators to counts through these, never through a hyperparameter
read directly, so a time-varying or latent-driven ascertainment is honoured.
"""
@inline observation_mean(spec, raw, latent, hyper, t) =
    _offset(spec.baseline, _modify(spec.mean_modifier, raw, latent, hyper, t), latent, hyper, t)
@inline observation_scale(spec, latent, hyper, t) = _modify(spec.mean_modifier, 1.0, latent, hyper, t)
@inline observation_baseline(spec, latent, hyper, t) = _offset(spec.baseline, 0.0, latent, hyper, t)

"""
    resolve_signal_indices(spec, n_signals) -> spec

Check a spec's signal indices against the layout, expanding an all-signals aggregate.
"""
function resolve_signal_indices(spec::SignalObservationSpec, n_signals::Integer)
    1 <= spec.signal_idx <= n_signals ||
        throw(ArgumentError("signal_idx $(spec.signal_idx) out of range [1, $n_signals]"))
    return spec
end
function resolve_signal_indices(spec::AggregatedSignalSpec, n_signals::Integer)
    indices = spec.is_all_signals ? collect(1:n_signals) : spec.signal_indices
    all(i -> 1 <= i <= n_signals, indices) ||
        throw(ArgumentError("signal indices $indices out of range [1, $n_signals]"))
    return AggregatedSignalSpec(indices, spec.noise_spec, spec.mean_modifier, spec.baseline, spec.name, false)
end

function _resolve_specs(specs::Tuple{Vararg{ObservationSpec}}, layout::StateLayout)
    isempty(specs) && throw(ArgumentError("obs_specs must have at least one element"))
    return map(s -> resolve_signal_indices(s, n_signals(layout)), specs)
end

"""
    observation_gaussian_moments(spec, raw_mean, raw_var, latent, hyper, t) -> (; mean, var)

Gaussian moments of a NegBinomial observation given Gaussian moments of its accumulator (the UKF
forecast mapping): `mean = observation_mean(max(raw_mean, 0))` and
`var = scale² raw_var + mean + mean²/φ + (σ mean)²`, with the spec's scale and noise at `t`.
"""
observation_gaussian_moments(spec, raw_mean, raw_var, latent, hyper, t) =
    _gaussian_moments(spec.noise_spec, spec, raw_mean, raw_var, latent, hyper, t)

function _gaussian_moments(noise::NegBinomialNoise, spec, raw_mean, raw_var, latent, hyper, t)
    raw = max(raw_mean, zero(raw_mean))
    scale = observation_scale(spec, latent, hyper, t)
    mu = observation_mean(spec, raw, latent, hyper, t)
    phi = _obs_value(noise.phi, latent, hyper, t)
    sigma = _obs_value(noise.sigma_mult, latent, hyper, t)
    return (; mean = mu, var = scale^2 * raw_var + mu + mu^2 / phi + (sigma * mu)^2)
end
_gaussian_moments(noise, args...) = throw(
    ArgumentError("only NegBinomialNoise has a closed-form Gaussian forecast mapping; got $(typeof(noise))")
)

# Every observation's mean at `(x, t)`; `specs` is a tuple so the loop unrolls over spec types.
@inline _observation_means(specs::Tuple, x, latent, p, t, accumulators, ::Val{K}) where {K} =
    ntuple(i -> observation_mean(specs[i], compute_true_mean(specs[i], x, accumulators), latent, p, t), Val(K))

@inline _effective_params(::Nothing, p, x) = p
@inline _effective_params(learned, p, x) = merge(p, learned.extract(x))

"""
    build_measurement_model(layout, obs_specs, stochastic) -> (; measure, n_obs, n_noise)
    build_measurement_model(layout, noise::ObservationNoiseSpec, stochastic)

The UKF measurement `measure(x, u, p, t, v) -> SVector{n_obs}` with unit noise `v` of length
`n_noise` (so `R2 = I`). The second form observes the single signal of a one-signal layout.
"""
function build_measurement_model(
        layout::StateLayout, obs_specs::Tuple{Vararg{ObservationSpec}}, stochastic::StochasticUpdate,
    )
    specs = _resolve_specs(obs_specs, layout)
    counts = map(n_noise_terms, specs)
    starts = ntuple(i -> 1 + sum(counts[1:(i - 1)]; init = 0), length(specs))
    accumulators = layout.accumulator_indices
    extract = stochastic.extract
    n_val = Val(length(specs))
    @inline function measure(x, u, p, t, v)
        latent = extract(x)
        means = _observation_means(specs, x, latent, p, t, accumulators, n_val)
        y = ntuple(n_val) do i
            slice = view(v, starts[i]:(starts[i] + counts[i] - 1))
            apply_noise(specs[i].noise_spec, means[i], slice, latent, p, t)
        end
        return SVector(y)
    end
    return (; measure, n_obs = length(specs), n_noise = sum(counts))
end

function build_measurement_model(layout::StateLayout, noise::ObservationNoiseSpec, stochastic::StochasticUpdate)
    n_signals(layout) == 1 ||
        throw(ArgumentError("a bare noise spec observes one signal; this layout has $(n_signals(layout))"))
    return build_measurement_model(layout, (SignalObservationSpec(1, noise),), stochastic)
end

"""
    build_measurement_logpdf(layout, obs_specs, stochastic; learned = nothing) -> g(x, u, y, p, t)

Particle weighting: the summed exact `observation_logpdf` of `y` over the observation specs.
`learned` (a `LearnedHyperparams`) lets each particle's own hyperparameters override `p`.
"""
function build_measurement_logpdf(
        layout::StateLayout, obs_specs::Tuple{Vararg{ObservationSpec}}, stochastic::StochasticUpdate;
        learned = nothing,
    )
    specs = _resolve_specs(obs_specs, layout)
    accumulators = layout.accumulator_indices
    extract = stochastic.extract
    n_val = Val(length(specs))
    @inline function logpdf_fn(x, u, y, p, t)
        p_eff = _effective_params(learned, p, x)
        latent = extract(x)
        means = _observation_means(specs, x, latent, p_eff, t, accumulators, n_val)
        return sum(ntuple(i -> observation_logpdf(specs[i].noise_spec, y[i], means[i], latent, p_eff, t), n_val))
    end
    return logpdf_fn
end

build_measurement_logpdf(layout::StateLayout, noise::ObservationNoiseSpec, stochastic::StochasticUpdate; learned = nothing) =
    build_measurement_logpdf(layout, (SignalObservationSpec(1, noise),), stochastic; learned)
