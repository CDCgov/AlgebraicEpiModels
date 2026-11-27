# Shared machinery of the inference engines: model assembly, the initial covariance, the replay
# log-likelihood, the rolling window checkpoint, fitted means and the summary table.

"""
    InferenceEngine

A `(filter, hyper)` pairing with all of its mutable state: the filter (cloud or ensemble and its
random streams) and the hyperparameter state (learned values, warm-start ensembles, the window
checkpoint). Passing it to [`fit_forecast!`](@ref) again continues inference. A `deepcopy` of a UKF
or EnKF engine is an isolated fork. A particle-filter fork copies the cloud and its own streams,
but LowLevelParticleFilters resamples from the task-local RNG, as does an engine built with the
default `rng`, and the copies share it.
"""
abstract type InferenceEngine end

"""
    EngineSettings(; dt, supersample = 2, integrator = Rk4, n_ahead, n_draws = 2000)

The run-level settings every engine shares: the observation interval in days, the integrator and
its substeps per interval (see [`build_full_dynamics`](@ref)), forecast horizon and forecast draw
count.
"""
struct EngineSettings
    dt::Float64
    supersample::Int
    integrator::Type{<:AbstractIntegrator}
    n_ahead::Int
    n_draws::Int
    function EngineSettings(;
            dt::Real, supersample::Integer = 2, integrator::Type{<:AbstractIntegrator} = Rk4, n_ahead::Integer, n_draws::Integer = 2000,
        )
        isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive, got $dt"))
        supersample >= 1 || throw(ArgumentError("supersample must be >= 1, got $supersample"))
        n_ahead >= 1 || throw(ArgumentError("n_ahead must be positive, got $n_ahead"))
        n_draws >= 1 || throw(ArgumentError("n_draws must be positive, got $n_draws"))
        return new(dt, supersample, integrator, n_ahead, n_draws)
    end
end

# A model's dynamics and measurement assembled at one `(dt, supersample, integrator, obs_jitter)`.
# `forecast_dynamics` is the step at `obs_jitter = 0`: the jitter keeps a Gaussian filter's
# covariance full-rank and is not process noise, so a forecast leaves it out.
struct Assembly{D, F, H}
    dynamics::D
    forecast_dynamics::F
    measure::H
    n_obs::Int
    n_noise::Int
end

function Assembly(model::EpiModel, s::EngineSettings, obs_jitter::Real)
    build(jitter) = build_full_dynamics(
        model.vectorfield!, model.stochastic, model.layout;
        dt = s.dt, s.supersample, s.integrator, obs_jitter = jitter,
    )
    dynamics = build(obs_jitter)
    measure, n_obs, n_noise = build_measurement_model(model.layout, model.observation, model.stochastic)
    return Assembly(dynamics, iszero(obs_jitter) ? dynamics : build(0.0), measure, n_obs, n_noise)
end

const _NO_INPUT = Float64[]

# What an engine fixes at build time: the model, its assembly, the run settings and the priors of
# the learned parameters.
struct EngineCore{M <: EpiModel, A <: Assembly, B <: ParameterPriorBundle}
    model::M
    assembly::A
    settings::EngineSettings
    bundle::B
end

EngineCore(model::EpiModel, settings::EngineSettings, obs_jitter::Real) =
    EngineCore(model, Assembly(model, settings, obs_jitter), settings, ParameterPriorBundle(model))

"""
    ForecastResult

What [`fit_forecast!`](@ref) returns: `quantiles`, `fitted_means`, `summary`, `samples` and
`latents`.
"""
struct ForecastResult{Q, F, S, L}
    quantiles::Q
    fitted_means::F
    summary::DataFrame
    samples::S
    latents::L
end

# The engines carry an observation axis on every per-observation array (`fitted_means[k, o]`,
# `quantiles[h, o, q]`, `samples[h, j, o]`); a one-observation model drops it here.
ForecastResult(; quantiles = nothing, fitted_means, summary, samples = nothing, latents) = ForecastResult(
    _drop_observation_axis(quantiles, 2), _drop_observation_axis(fitted_means, 2), summary,
    _drop_observation_axis(samples, 3), latents,
)

_drop_observation_axis(::Nothing, _) = nothing
_drop_observation_axis(A::AbstractArray, dim) = size(A, dim) == 1 ? dropdims(A; dims = dim) : A

"Default initial variance of a latent coefficient slot (unconstrained space): in a log chart, sd 0.45."
const DEFAULT_LATENT_VARIANCE = 0.2
"Initial sd of each compartment as a fraction of its own initial value."
const DEFAULT_MODEL_RELATIVE_SD = 0.05
"Variance floor keeping `P0` positive definite for a compartment seeded at zero."
const MIN_MODEL_VARIANCE = 1.0e-6

# Initial augmented covariance: compartments at `(relative_sd * x0)^2` floored at the minimum
# (reset accumulators overridable by signal name, which the ensemble filter needs because its
# sample cross-covariance is noise rather than an exact zero), latent slots at the default unless
# overridden by name, then the learned tail.
function _initial_state_covariance(
        layout::StateLayout, x0, learned_variance::AbstractVector{<:Real} = Float64[];
        latent_variance::NamedTuple = (;), accumulator_variance::NamedTuple = (;),
        model_relative_sd::Real = DEFAULT_MODEL_RELATIVE_SD,
    )
    _check_known_names(latent_variance, layout.latent_names, "initial_latent_variance")
    _check_known_names(accumulator_variance, layout.signal_names, "initial_accumulator_variance")
    model_relative_sd >= 0 || throw(ArgumentError("model_relative_sd must be non-negative, got $model_relative_sd"))
    model_var = [max((model_relative_sd * float(x0[i]))^2, MIN_MODEL_VARIANCE) for i in 1:n_ode_states(layout)]
    for (signal, slot) in zip(layout.signal_names, layout.accumulator_indices)
        haskey(accumulator_variance, signal) || continue
        v = Float64(accumulator_variance[signal])
        v >= 0 || throw(ArgumentError("initial_accumulator_variance[$signal] must be non-negative, got $v"))
        model_var[slot] = max(v, MIN_MODEL_VARIANCE)
    end
    latent_var = [Float64(get(latent_variance, name, DEFAULT_LATENT_VARIANCE)) for name in layout.latent_names]
    return Matrix(Diagonal(vcat(model_var, latent_var, collect(float.(learned_variance)))))
end

function _check_known_names(overrides::NamedTuple, known, what::AbstractString)
    for name in keys(overrides)
        name in known || throw(ArgumentError("$what names `$name`, which is not one of $(collect(known))"))
    end
    return nothing
end

"""
    positive_cholesky!(R)

Positive-definite Cholesky via `PositiveFactorizations.ldlt!`, which unlike its `cholesky!`
wrapper accepts `ForwardDiff.Dual` matrices. It factors in correlation coordinates, so the repair
tolerance does not overwrite small latent variances beside large compartment ones, and scales the
lower factor back to state units. Only an exactly zero diagonal takes `MIN_MODEL_VARIANCE` as its
scale.
"""
function positive_cholesky!(R)
    A = Matrix(R)
    T = eltype(A)
    n = size(A, 1)
    scale = [sqrt(iszero(A[i, i]) ? T(MIN_MODEL_VARIANCE) : abs(A[i, i])) for i in 1:n]
    @inbounds for j in 1:n, i in 1:n
        A[i, j] /= scale[i] * scale[j]
    end
    F = ldlt!(Positive{T}, A; tol = default_tol(A), blocksize = default_blocksize(floattype(T)))[1]
    L = F.factors
    @inbounds for j in 1:n, i in j:n
        L[i, j] *= scale[i]
    end
    return F
end

"""
    marginal_loglik(filter, ys, p; warmup = 0) -> Real

Filter marginal log-likelihood of the observation vectors `ys` under hyperparameters `p` after
`reset!`, accumulated at the promoted element type so it differentiates
(`forward_trajectory(...).ll` sizes its buffers as Float64). A `missing` entry is a grid slot
without an observation: the filter predicts through it without correcting. The filter corrects on
the first `warmup` slots without counting their terms.
"""
function marginal_loglik(filter, ys, p; warmup::Integer = 0)
    reset!(filter)
    return _filter_loglik!(filter, ys, p, eachindex(ys); warmup)
end

_loglik_zero(kf::AbstractKalmanFilter) = zero(eltype(state(kf)))
_loglik_zero(pf::LLPF.AbstractParticleFilter) = zero(eltype(LLPF.weights(pf)))

# Advance an initialised filter through the grid slots in `range` at absolute model time,
# correcting on present observations and predicting through every slot.
function _filter_loglik!(filter, ys, p, range; warmup::Integer = 0)
    ll = _loglik_zero(filter)
    for k in range
        t = (k - 1) * filter.Ts
        if ys[k] !== missing
            term = first(_correct!(filter, ys[k], p, t))
            k > warmup && (ll += term)
        end
        predict!(filter, _NO_INPUT, p, t)
    end
    return ll
end

# One filtering pass over the whole grid from `reset!`, as `_filter_loglik!` but recording the
# corrected state mean and covariance at every slot. `at_origin(filter)` is evaluated at the last
# slot before its `predict!`, which is where the corrected ensemble a forecast starts from lives.
function _filter_pass!(kf, ys, p; at_origin = Returns(nothing))
    reset!(kf)
    T = length(ys)
    xt = Vector{Vector{Float64}}(undef, T)
    Rt = Vector{Matrix{Float64}}(undef, T)
    ll = _loglik_zero(kf)
    origin = nothing
    for k in 1:T
        t = (k - 1) * kf.Ts
        ys[k] === missing || (ll += first(_correct!(kf, ys[k], p, t)))
        xt[k] = Vector{Float64}(state(kf))
        Rt[k] = Matrix{Float64}(covariance(kf))
        k == T && (origin = at_origin(kf))
        predict!(kf, _NO_INPUT, p, t)
    end
    return (; xt, Rt, ll, origin)
end

# Filtered latents `(; mean, sd)` over `slots` in the unconstrained chart, each `[slot, latent]`.
_latent_moments(xt, Rt, latent_range, slots) = (;
    mean = Float64[xt[k][s] for k in slots, s in latent_range],
    sd = Float64[sqrt(max(Rt[k][s, s], 0.0)) for k in slots, s in latent_range],
)

function _cloud_latent_moments(cloud, weights, latent_range)
    wsum = sum(weights)
    mean = [sum(weights[i] * cloud[i][slot] for i in eachindex(cloud)) / wsum for slot in latent_range]
    sd = [
        sqrt(max(sum(weights[i] * (cloud[i][slot] - mean[l])^2 for i in eachindex(cloud)) / wsum, 0.0))
            for (l, slot) in enumerate(latent_range)
    ]
    return mean, sd
end

_stack_latents(means, sds, n_latent) = isempty(means) ? (; mean = zeros(0, n_latent), sd = zeros(0, n_latent)) :
    (; mean = permutedims(reduce(hcat, means)), sd = permutedims(reduce(hcat, sds)))

# The slots a marginal log-likelihood counts: the trailing `window` (all for `nothing`), never the
# model's first `warmup`.
_loss_range(T::Integer, ::Nothing, warmup::Integer = 0) = (Int(warmup) + 1):Int(T)
_loss_range(T::Integer, window::Integer, warmup::Integer = 0) = max(Int(warmup) + 1, Int(T) - Int(window) + 1):Int(T)

_warmup_steps(model::EpiModel, dt) = ceil(Int, model.likelihood_warmup_days / dt - 1.0e-9)

# A calibration needs at least one observation to score in `loss_range`.
_require_scored(ys, loss_range, warmup::Integer, what) = any(k -> ys[k] !== missing, loss_range) || throw(
    ArgumentError(
        "$what has no observation to score in slots $(first(loss_range)):$(last(loss_range)) of " *
            "$(length(ys)) (the model's likelihood warm-up covers the first $warmup). Start the first calibration later."
    )
)

# A complete filter stopped just before the first scored observation of a rolling window. It is
# built once and only ever advanced forward; rebuilding the prefix at new parameters would
# silently substitute the constant-θ prefix that `window_length = nothing` computes honestly, so a
# window that must move backwards is refused. Value revisions behind the boundary are ignored.
mutable struct WindowCheckpoint
    filter::Any
    start::Int
end
WindowCheckpoint() = WindowCheckpoint(nothing, 1)

# Returns `(filter_or_nothing, mode, steps_advanced)` with mode `:full_history`, `:anchored` or
# `:advanced`. A window starting inside the warm-up is the full history, so every candidate replays
# the warm-up from its own starting state; a later window's checkpoint includes the warm-up.
function _prepare_window_checkpoint!(cp::WindowCheckpoint, build_anchor, p, ys, loss_range; warmup::Integer = 0)
    new_start = first(loss_range)
    if new_start <= warmup + 1
        cp.filter, cp.start = nothing, 1
        return nothing, :full_history, 0
    end
    if cp.filter === nothing
        anchor = build_anchor()
        reset!(anchor)
        _filter_loglik!(anchor, ys, p, 1:(new_start - 1))
        cp.filter, cp.start = anchor, new_start
        return anchor, :anchored, new_start - 1
    end
    old_start = cp.start
    new_start >= old_start || error(
        "window checkpoint cannot move backwards: it stands before observation $old_start but the " *
            "window now starts at $new_start. Set `window_length = nothing` to score the complete history.",
    )
    new_start == old_start && return cp.filter, :advanced, 0
    advancing = deepcopy(cp.filter)   # advance a copy and swap on success
    _filter_loglik!(advancing, ys, p, old_start:(new_start - 1))
    cp.filter, cp.start = advancing, new_start
    return advancing, :advanced, new_start - old_start
end

# A candidate filter resumed from the checkpoint at the candidate's parameters. A Gaussian
# candidate is rebuilt at the AD element type and receives the checkpoint's (constant) state; an
# ensemble candidate is a deep copy carrying its RNG position, which keeps common random numbers.
function _candidate_from_checkpoint(cp::LLPF.AbstractUnscentedKalmanFilter, build_candidate, ::Type{T}, p) where {T}
    candidate = build_candidate(T, p)
    candidate.x = typeof(candidate.x)(state(cp))
    candidate.R .= covariance(cp)
    candidate.t = LLPF.index(cp)
    return candidate
end
function _candidate_from_checkpoint(cp::EnsembleKalmanFilter, _build, _T, p)
    candidate = deepcopy(cp)
    candidate.p = p
    return candidate
end

# One slot's observation as the filters read it: a Float64 vector, or `missing` for a slot with
# nothing observed. An absent component (a signal unsampled that slot) becomes `NaN`, which every
# filter skips; `NaN` in the data itself is refused so a corrupt value cannot pass as an absent one.
_observation_vector(y::Real) = [_observation_component(y)]
_observation_vector(y::AbstractVector) = all(ismissing, y) ? missing : map(_observation_component, collect(y))
_observation_vector(::Missing) = missing
_observation_component(::Missing) = NaN
_observation_component(v::Real) = isnan(v) ?
    throw(ArgumentError("observation $v is NaN; use `missing` for an absent observation")) : Float64(v)
_observation_vectors(ys::AbstractVector) = [_observation_vector(y) for y in ys]

# One correction at a present slot. A Gaussian filter with absent components corrects on the
# observed sub-block, the exact marginal update; the particle log-density skips them itself.
_correct!(filter, y, p, t) = correct!(filter, _NO_INPUT, y, p, t)
function _correct!(ukf::UnscentedKalmanFilter, y, p, t)
    observed = findall(!isnan, y)
    length(observed) == length(y) && return correct!(ukf, _NO_INPUT, y, p, t)
    mm = ukf.measurement_model
    sub = LLPF.UKFMeasurementModel{eltype(ukf.x), false, true}(
        (x, u, p, t, v) -> mm.measurement(x, u, p, t, v)[observed], mm.R2;
        nx = length(ukf.x), ny = length(observed), weight_params = mm.weight_params, static = false,
    )
    return correct!(ukf, sub, _NO_INPUT, y[observed], p, t)
end

# The EnKF engine's measurement, narrowed in place to a slot's observed components while that slot
# is corrected: the ensemble filter sizes its update by its `ny` and corrects every component.
mutable struct _ObservedComponents{H}
    const measurement::H
    components::Union{Colon, Vector{Int}}
end
_ObservedComponents(measurement) = _ObservedComponents(measurement, Colon())
(m::_ObservedComponents)(x, u, p, t, v) = m.measurement(x, u, p, t, v)[m.components]

function _correct!(enkf::EnsembleKalmanFilter{true, true, <:Any, <:_ObservedComponents}, y, p, t)
    observed = findall(!isnan, y)
    length(observed) == length(y) && return correct!(enkf, _NO_INPUT, y, p, t)
    ny = enkf.ny
    enkf.ny, enkf.measurement.components = length(observed), observed
    try
        return correct!(enkf, _NO_INPUT, y[observed], p, t)
    finally
        enkf.ny, enkf.measurement.components = ny, Colon()
    end
end

# Fitted observation means `[slot, observation]`: each filtered state mean through its spec's
# `observation_mean` at its own model time.
_fitted_observation_means(specs, accumulators, extract, xt, hyper, dt, range) = [
    observation_mean(spec, compute_true_mean(spec, xt[k], accumulators), extract(xt[k]), hyper, (k - 1) * dt)
        for k in range, spec in specs
]

_cloud_sd(values) = length(values) < 2 ? 0.0 : Float64(std(values))

# Per-horizon predictive sd of each latent slot over the forecast draws, `[horizon, latent]`.
_latent_sd(latent_samples::AbstractArray{<:Real, 3}) =
    [_cloud_sd(view(latent_samples, h, :, l)) for h in axes(latent_samples, 1), l in axes(latent_samples, 3)]

_summary() = DataFrame(parameter = String[], statistic = String[], value = Float64[])

function _validate_fit(observations, update_range, forecast_number)
    T = length(observations)
    isempty(update_range) && throw(ArgumentError("update_range must not be empty"))
    first(update_range) >= 1 && last(update_range) == T || throw(
        ArgumentError("update_range must end at the forecast origin $T; got $(first(update_range)):$(last(update_range))")
    )
    forecast_number > 0 || throw(ArgumentError("forecast_number must be positive, got $forecast_number"))
    return T
end

_require_full_range(update_range, T, what) = update_range == 1:T || throw(
    ArgumentError("$what replays the complete series and requires update_range 1:$T; got $(first(update_range)):$(last(update_range))")
)

"""
    build_inference(filter, hyper, model::EpiModel; dt, supersample = 2, integrator = Rk4, n_ahead,
                    n_draws = 2000, rng = Random.default_rng(), kwargs...) -> InferenceEngine
    build_inference(cfg::InferenceConfig, model::EpiModel; rng = Random.default_rng())

Build the inference engine for a `(filter, hyper)` pairing: [`UKF`](@ref) + [`Optimise`](@ref),
[`PF`](@ref) + [`LiuWest`](@ref) or [`EnKF`](@ref) + [`EKP`](@ref). Drive it with
[`fit_forecast!`](@ref). The [`InferenceConfig`](@ref) form takes the pairing, `dt = step_days`,
`supersample`, `integrator`, `n_ahead` and `n_draws` from the config.
"""
build_inference(filter::StateFilter, hyper::HyperMethod, ::EpiModel; kwargs...) = throw(
    ArgumentError("build_inference is not implemented for $(nameof(typeof(filter))) + $(nameof(typeof(hyper)))")
)

build_inference(cfg::InferenceConfig, model::EpiModel; rng = Random.default_rng()) = build_inference(
    cfg.filter, cfg.hyper, model;
    dt = cfg.step_days, cfg.supersample, integrator = integrator_type(cfg.integrator), cfg.n_ahead, cfg.n_draws, rng,
)

"""
    fit_forecast!(engine, observations, forecast_number; update_range = eachindex(observations),
                  emit_forecast = true) -> ForecastResult

Assimilate `observations`, one entry per slot of the regular `dt` grid, and forecast `n_ahead`
steps ahead. An entry is a number, a vector with one component per observation, or `missing`
where nothing is observed: the filter predicts through the slot without correcting. A `missing`
component is a signal absent that slot, and every filter corrects on the observed ones only.
`forecast_number` counts fitted origins and drives the re-optimisation cadence. The replay engines
(UKF, EnKF) require the complete `update_range`; the online PF engine keeps its cloud between calls
and `update_range` must start at the slot after the last one it assimilated (the default), so a
replay is `reset!(engine.filter)` followed by the full range.

`fitted_means` covers `update_range` (a nowcast at a `missing` slot). For one observation it is a
vector, `quantiles` is `[horizon, quantile]` and `samples` (the predictive draws behind them, each
simulated through the model's own process and observation noise) is `[horizon, draw]`; several
observations add an observation axis:
`[slot, observation]`, `[horizon, observation, quantile]` and `[horizon, draw, observation]`.
`quantiles` and `samples` are `nothing` when `emit_forecast = false`. `summary` is a
`(parameter, statistic, value)` table of estimates and diagnostics. `latents` holds the filtered
latents over `update_range` in their unconstrained chart, `(; mean, sd)` as `[slot, latent]`
matrices (see [`latent_path_rows`](@ref)).
"""
function fit_forecast! end
