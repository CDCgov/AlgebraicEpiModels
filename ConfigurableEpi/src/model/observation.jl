# Observation model: noise families, per-signal specs, and the measurement, likelihood and
# sampling functions built from them. Latents reach the specs constrained via `stochastic.extract`.

abstract type ObservationNoiseSpec end

# A noise or modifier parameter: a constant or a `(latent, hyper, t)` function.
@inline _obs_value(x::Real, latent, hyper, t) = x
@inline _obs_value(f, latent, hyper, t) = f(latent, hyper, t)

"""
    NegBinomialNoise(; phi, sigma_mult = 0.0)

Count noise with `Var(y) = μ + μ²/φ + (σμ)²`. The UKF uses the Gaussian approximation with one
unit-normal term. A simulated draw is `NegativeBinomial(φ, φ/(φ+μ'))` about the mean perturbed by
a mean-one log-normal, `μ' = μ exp(σz − σ²/2)`. That mixture has no closed-form density, so the
particle filter refuses `σ ≠ 0`. Each parameter is a `Real` or a function `(latent, hyper, t) -> Real`.
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
    RawMean

What an observation's mean is read from, before its modifier and baseline. `raw_mean(r, x,
accumulators)` gives it for a resolved kind; [`resolve_observation`](@ref) resolves a kind against
the layout.
"""
abstract type RawMean end

# One signal's reset accumulator.
struct Signal <: RawMean
    index::Int
    Signal(index::Integer) = index >= 1 ? new(index) : throw(ArgumentError("signal_idx must be >= 1"))
end

# The sum of several signals' reset accumulators.
struct SignalSum <: RawMean
    indices::Vector{Int}
    function SignalSum(indices::AbstractVector)
        isempty(indices) && throw(ArgumentError("signal_indices must have at least one element"))
        all(>=(1), indices) || throw(ArgumentError("all signal_indices must be >= 1"))
        return new(collect(Int, indices))
    end
end

# Every signal, expanded to a `SignalSum` once the layout is known.
struct AllSignals <: RawMean end

# A weighted sum of named ODE states, `Σ weights[i] * max(x[names[i]], 0)`; `WeightedSlots` once resolved.
struct WeightedStates <: RawMean
    names::Vector{Symbol}
    weights::Vector{Float64}
    function WeightedStates(names::AbstractVector, weights::AbstractVector)
        isempty(names) && throw(ArgumentError("a weighted state observation needs at least one state"))
        length(weights) == length(names) || throw(DimensionMismatch("$(length(weights)) weights for $(length(names)) states"))
        allunique(names) || throw(ArgumentError("state_names must be distinct; got $names"))
        all(w -> isfinite(w) && w >= 0, weights) || throw(ArgumentError("weights must be finite and non-negative; got $weights"))
        return new(Vector{Symbol}(names), Vector{Float64}(weights))
    end
end

struct WeightedSlots <: RawMean
    indices::Vector{Int}
    weights::Vector{Float64}
end

# A share of incidence, `Σ_numerator acc / Σ_denominator acc`.
struct SignalShare <: RawMean
    numerator::Vector{Int}
    denominator::Vector{Int}
    function SignalShare(numerator::AbstractVector, denominator::AbstractVector)
        isempty(numerator) && throw(ArgumentError("a share needs at least one numerator signal"))
        allunique(numerator) && allunique(denominator) || throw(ArgumentError("share signal indices must be distinct"))
        issubset(numerator, denominator) ||
            throw(ArgumentError("share numerator $numerator is not a subset of denominator $denominator"))
        all(>=(1), denominator) || throw(ArgumentError("all signal indices must be >= 1"))
        return new(collect(Int, numerator), collect(Int, denominator))
    end
end

# No compartment at all: the mean is the modifier alone.
struct LatentOnly <: RawMean end

"""
    ObservationSpec{R <: RawMean}

One observation: its mean is `raw * mean_modifier + baseline`, where `raw` is read from the
state as the [`RawMean`](@ref) kind says, and `mean_modifier` and `baseline` are each `nothing`, a
`Real` or a function `(latent, hyper, t) -> Real` (an [`AscertainmentPath`](@ref), say). Built by
[`SignalObservationSpec`](@ref), [`AggregatedSignalSpec`](@ref), [`WeightedStateSpec`](@ref),
[`ShareSignalSpec`](@ref) or [`LatentObservationSpec`](@ref).
"""
struct ObservationSpec{R <: RawMean, N <: ObservationNoiseSpec, M, B}
    raw::R
    noise_spec::N
    mean_modifier::M
    baseline::B
    name::Symbol
end

"""
    SignalObservationSpec(signal_idx, noise; mean_modifier = nothing, baseline = nothing, name)

An observation of one signal's reset accumulator.
"""
SignalObservationSpec(
    signal_idx::Integer, noise::ObservationNoiseSpec;
    mean_modifier = nothing, baseline = nothing, name::Symbol = Symbol("obs_", signal_idx),
) = ObservationSpec(Signal(signal_idx), noise, mean_modifier, baseline, name)

"""
    AggregatedSignalSpec(signal_indices, noise; mean_modifier = nothing, baseline = nothing, name)
    AggregatedSignalSpec(noise; ...)   # the sum of every signal

An observation of the sum of several signals' reset accumulators.
"""
AggregatedSignalSpec(
    signal_indices::AbstractVector, noise::ObservationNoiseSpec;
    mean_modifier = nothing, baseline = nothing, name::Symbol = :obs_aggregated,
) = ObservationSpec(SignalSum(signal_indices), noise, mean_modifier, baseline, name)

AggregatedSignalSpec(noise::ObservationNoiseSpec; mean_modifier = nothing, baseline = nothing, name::Symbol = :obs_total) =
    ObservationSpec(AllSignals(), noise, mean_modifier, baseline, name)

"""
    WeightedStateSpec(state_names, weights, noise; mean_modifier = nothing, baseline = nothing, name = :obs_weighted)

An observation of `Σ_i weights[i] * max(x[state_names[i]], 0)` over named ODE states: the stages
of a shedding chain, say, or several signals' accumulators each with its own ascertainment.
"""
WeightedStateSpec(
    state_names, weights, noise::ObservationNoiseSpec;
    mean_modifier = nothing, baseline = nothing, name::Symbol = :obs_weighted,
) = ObservationSpec(WeightedStates(state_names, weights), noise, mean_modifier, baseline, name)

"""
    ShareSignalSpec(numerator, denominator, noise; mean_modifier = nothing, baseline = nothing, name = :obs_share)

An observation of a share of incidence, `Σ_numerator acc / Σ_denominator acc` (the H3 fraction of
infections, say), usually scaled to counts by a known per-slot total ([`SlotSeries`](@ref)). The
numerator signals must be a subset of the denominator's.
"""
ShareSignalSpec(
    numerator::AbstractVector, denominator::AbstractVector, noise::ObservationNoiseSpec;
    mean_modifier = nothing, baseline = nothing, name::Symbol = :obs_share,
) = ObservationSpec(SignalShare(numerator, denominator), noise, mean_modifier, baseline, name)

"""
    LatentObservationSpec(mean, noise; name)

An observation that reads no compartment: its mean is `mean(latent, hyper, t)`.
"""
LatentObservationSpec(mean, noise::ObservationNoiseSpec; name::Symbol) =
    ObservationSpec(LatentOnly(), noise, mean, nothing, name)

"""
    SlotSeries(values; step = 1.0)

Mean modifier returning `values[k]` at the grid slot `t = (k - 1) * step`, and the last value
beyond the series (forecast horizons). A time between slots is an error.
"""
struct SlotSeries
    values::Vector{Float64}
    step::Float64
    function SlotSeries(values::AbstractVector; step::Real = 1.0)
        isempty(values) && throw(ArgumentError("a slot series needs at least one value"))
        step > 0 || throw(ArgumentError("slot step must be positive, got $step"))
        vals = Float64.(values)
        all(v -> isfinite(v) && v >= 0, vals) || throw(ArgumentError("slot series values must be finite and non-negative"))
        return new(vals, Float64(step))
    end
end

const _SLOT_TOLERANCE = 1.0e-6

@inline function (series::SlotSeries)(latent, hyper, t)
    slot = t / series.step
    k = round(Int, slot)
    abs(slot - k) <= _SLOT_TOLERANCE || _off_slot_grid(t, series.step)
    return series.values[clamp(k + 1, 1, length(series.values))]
end

@noinline _off_slot_grid(t, step) =
    throw(ArgumentError("a slot series with step $step has no value at t = $t, between its slots"))

function _validated_reporting_fractions(fractions::AbstractVector)
    values = Float64[ismissing(f) ? 1.0 : Float64(f) for f in fractions]
    all(f -> isfinite(f) && 0.0 <= f <= 1.0, values) || throw(ArgumentError("reporting fractions must be finite and lie in [0, 1]"))
    return values
end

"""
    ReportingFractionModifier(inner, fractions)
    FullReporting(inner)

Mean modifiers for a partially reported daily count. `ReportingFractionModifier` multiplies
`inner` by the reporting fraction of each fitted day from `t = 0` (`missing` counts as one); later
days, the forecast horizons, are fully reported. `FullReporting` is `inner` unchanged.
[`set_reporting_fractions!`](@ref) gives either a new report vintage in place.
"""
struct ReportingFractionModifier{M}
    inner::M
    fractions::Vector{Float64}
    ReportingFractionModifier(inner::M, fractions::AbstractVector) where {M} = new{M}(inner, _validated_reporting_fractions(fractions))
end

struct FullReporting{M}
    inner::M
end

@inline function _reporting_fraction(modifier::ReportingFractionModifier, t)
    slot = round(Int, t) + 1
    return 1 <= slot <= length(modifier.fractions) ? modifier.fractions[slot] : 1.0
end

@inline (modifier::ReportingFractionModifier)(latent, hyper, t) =
    modifier.inner(latent, hyper, t) * _reporting_fraction(modifier, t)
@inline (modifier::FullReporting)(latent, hyper, t) = modifier.inner(latent, hyper, t)

"""
    set_reporting_fractions!(modifier, fractions) -> modifier

Replace a [`ReportingFractionModifier`](@ref)'s fractions in place, so a persistent engine sees a
new report vintage; the identity for `FullReporting`.
"""
function set_reporting_fractions!(modifier::ReportingFractionModifier, fractions::AbstractVector)
    values = _validated_reporting_fractions(fractions)
    empty!(modifier.fractions)
    append!(modifier.fractions, values)
    return modifier
end
set_reporting_fractions!(modifier::FullReporting, ::Any) = modifier

"""
    build_reporting_observation(mode::ReportingMode, inner, fractions)

The mean modifier for a reporting mode: `FullReporting` for [`CorrectedObservations`](@ref), or a
[`ReportingFractionModifier`](@ref) for `ThinnedObservations`, which needs the fractions.
"""
build_reporting_observation(::CorrectedObservations, inner, fractions) = FullReporting(inner)
build_reporting_observation(::ThinnedObservations, inner, fractions::AbstractVector) = ReportingFractionModifier(inner, fractions)
build_reporting_observation(::ThinnedObservations, inner, ::Nothing) =
    throw(ArgumentError("thinned observations need the reporting fraction of every fitted slot"))

"""
    LinearScale(latent::Symbol, inner; link = identity)

Mean modifier `link(latent[name]) * inner(latent, hyper, t)`: linear in the linked, constrained
value of the named latent. `inner` is a `Real` or a `(latent, hyper, t)` function that must not
read `latent[name]`.
"""
struct LinearScale{M, F}
    latent::Symbol
    inner::M
    link::F
end

LinearScale(latent::Symbol, inner; link = identity) = LinearScale(latent, inner, link)

@inline (m::LinearScale)(latent, hyper, t) = m.link(latent[m.latent]) * _obs_value(m.inner, latent, hyper, t)

n_noise_terms(::NegBinomialNoise) = 1
n_noise_terms(spec::PoissonNoise) = spec.sigma_mult === nothing ? 1 : 2
n_noise_terms(::LogNormalNoise) = 1
n_noise_terms(spec::ObservationSpec) = n_noise_terms(spec.noise_spec)

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
@inline function _check_pure_negbin(sigma)
    iszero(sigma) || throw(
        ArgumentError(
            "NegBinomialNoise with sigma_mult has no closed-form likelihood for the particle filter; " *
                "the (σμ)² term is simulated but never scored",
        )
    )
    return nothing
end

function observation_logpdf(spec::NegBinomialNoise, y, true_mean, latent, hyper, t)
    _check_pure_negbin(_obs_value(spec.sigma_mult, latent, hyper, t))
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

# With `sigma_mult = 0` no extra randomness is drawn, so a seeded path is unchanged.
function sample_observation(spec::NegBinomialNoise, true_mean, latent, hyper, t, rng)
    phi = _obs_value(spec.phi, latent, hyper, t)
    sigma = _obs_value(spec.sigma_mult, latent, hyper, t)
    mu = max(true_mean, 1.0e-6)
    iszero(sigma) || (mu *= exp(sigma * randn(rng) - sigma^2 / 2))
    return float(rand(rng, _nb_dist(mu, phi)))
end
function sample_observation(spec::PoissonNoise, true_mean, latent, hyper, t, rng)
    _check_pure_poisson(spec)
    return float(rand(rng, Poisson(max(true_mean, 1.0e-6))))
end
function sample_observation(spec::LogNormalNoise, true_mean, latent, hyper, t, rng)
    sigma = _obs_value(spec.sigma, latent, hyper, t)
    return rand(rng, LogNormal(log(max(true_mean, 1.0e-6)), sigma))
end

# This step's raw mean, read from the state and floored at zero.
@inline _floored(v) = max(v, zero(v))
@inline function _signal_total(indices, x, accumulators)
    total = zero(eltype(x))
    for s in indices
        total += _floored(x[accumulators[s]])
    end
    return total
end

@inline raw_mean(r::Signal, x, accumulators) = _floored(x[accumulators[r.index]])
@inline raw_mean(r::SignalSum, x, accumulators) = _signal_total(r.indices, x, accumulators)
@inline function raw_mean(r::WeightedSlots, x, accumulators)
    total = zero(eltype(x))
    @inbounds for (k, i) in enumerate(r.indices)
        total += r.weights[k] * _floored(x[i])
    end
    return total
end
@inline raw_mean(r::SignalShare, x, accumulators) =
    _signal_total(r.numerator, x, accumulators) / max(_signal_total(r.denominator, x, accumulators), eltype(x)(1.0e-12))
@inline raw_mean(::LatentOnly, x, accumulators) = one(eltype(x))
raw_mean(r::Union{AllSignals, WeightedStates}, x, accumulators) =
    throw(ArgumentError("a $(nameof(typeof(r))) observation must be resolved against a layout first (`resolve_observation`)"))

@inline compute_true_mean(spec::ObservationSpec, x, accumulators) = raw_mean(spec.raw, x, accumulators)

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

_check_signals(indices, n) = all(i -> 1 <= i <= n, indices) || throw(ArgumentError("signal indices $indices out of range [1, $n]"))

_resolve(r::Signal, layout) = (_check_signals((r.index,), n_signals(layout)); r)
_resolve(r::SignalSum, layout) = (_check_signals(r.indices, n_signals(layout)); r)
_resolve(::AllSignals, layout) = SignalSum(1:n_signals(layout))
_resolve(r::SignalShare, layout) = (_check_signals(r.denominator, n_signals(layout)); r)
_resolve(r::Union{WeightedSlots, LatentOnly}, layout) = r
function _resolve(r::WeightedStates, layout)
    names = ode_names(layout)
    indices = map(r.names) do name
        i = findfirst(==(name), names)
        i === nothing && throw(ArgumentError("a weighted state observation names `$name`, which is not an ODE state of the layout $names"))
        i
    end
    return WeightedSlots(indices, r.weights)
end

"""
    resolve_observation(spec, layout) -> spec

`spec` checked against `layout`: signal indices in range, an all-signals sum expanded and state
names turned into slots. The measurement builders resolve every spec.
"""
resolve_observation(spec::ObservationSpec, layout::StateLayout) =
    ObservationSpec(_resolve(spec.raw, layout), spec.noise_spec, spec.mean_modifier, spec.baseline, spec.name)

function _resolve_specs(specs::Tuple{Vararg{ObservationSpec}}, layout::StateLayout)
    isempty(specs) && throw(ArgumentError("obs_specs must have at least one element"))
    return map(s -> resolve_observation(s, layout), specs)
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
