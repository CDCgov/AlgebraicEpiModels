# Shared machinery of the inference engines: model assembly, the initial covariance, the replay
# log-likelihood, the rolling window checkpoint, fitted means and the summary table.

abstract type InferenceEngine end

"""
    EngineSettings(; dt, supersample, n_ahead, n_draws)

The run-level numbers every engine shares: the observation interval in days, RK4 substeps per
interval, forecast horizon and forecast draw count.
"""
struct EngineSettings
    dt::Float64
    supersample::Int
    n_ahead::Int
    n_draws::Int
    function EngineSettings(; dt::Real, supersample::Integer = 2, n_ahead::Integer, n_draws::Integer = 2000)
        isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive, got $dt"))
        supersample >= 1 || throw(ArgumentError("supersample must be >= 1, got $supersample"))
        n_ahead >= 1 || throw(ArgumentError("n_ahead must be positive, got $n_ahead"))
        n_draws >= 1 || throw(ArgumentError("n_draws must be positive, got $n_draws"))
        return new(dt, supersample, n_ahead, n_draws)
    end
end

# A model's dynamics and measurement assembled at one `(dt, supersample, obs_jitter)`.
struct Assembly{D, H}
    dynamics::D
    measure::H
    n_obs::Int
    n_noise::Int
end

function Assembly(model::EpiModel, s::EngineSettings, obs_jitter::Real)
    dynamics = build_full_dynamics(
        model.vectorfield!, model.stochastic, model.layout; dt = s.dt, supersample = s.supersample, obs_jitter,
    )
    measure, n_obs, n_noise = build_measurement_model(model.layout, model.observation, model.stochastic)
    return Assembly(dynamics, measure, n_obs, n_noise)
end

const _NO_INPUT = Float64[]

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
wrapper accepts `ForwardDiff.Dual` matrices.
"""
function positive_cholesky!(R)
    A = Matrix(R)
    T = eltype(A)
    return ldlt!(Positive{T}, A; tol = default_tol(A), blocksize = default_blocksize(floattype(T)))[1]
end

"""
    marginal_loglik(filter, ys, p) -> Real

Filter marginal log-likelihood of the observation vectors `ys` under hyperparameters `p` after
`reset!`, accumulated at the promoted element type so it differentiates
(`forward_trajectory(...).ll` sizes its buffers as Float64). A `missing` entry is a grid slot
without an observation: the filter predicts through it without correcting.
"""
function marginal_loglik(filter, ys, p)
    reset!(filter)
    return _filter_loglik!(filter, ys, p, eachindex(ys))
end

_loglik_zero(kf::AbstractKalmanFilter) = zero(eltype(state(kf)))
_loglik_zero(pf::LLPF.AbstractParticleFilter) = zero(eltype(LLPF.weights(pf)))

# Advance an initialised filter through the grid slots in `range` at absolute model time,
# correcting on present observations and predicting through every slot.
function _filter_loglik!(filter, ys, p, range)
    ll = _loglik_zero(filter)
    for k in range
        t = (k - 1) * filter.Ts
        ys[k] === missing || (ll += first(correct!(filter, _NO_INPUT, ys[k], p, t)))
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
        ys[k] === missing || (ll += first(correct!(kf, _NO_INPUT, ys[k], p, t)))
        xt[k] = Vector{Float64}(state(kf))
        Rt[k] = Matrix{Float64}(covariance(kf))
        k == T && (origin = at_origin(kf))
        predict!(kf, _NO_INPUT, p, t)
    end
    return (; xt, Rt, ll, origin)
end

_loss_range(T::Integer, ::Nothing) = 1:Int(T)
_loss_range(T::Integer, window::Integer) = max(1, Int(T) - Int(window) + 1):Int(T)

# A complete filter stopped just before the first scored observation of a rolling window. It is
# built once and only ever advanced forward; rebuilding the prefix at new parameters would
# silently substitute the constant-θ prefix that `window_length = nothing` computes honestly, so a
# window that must move backwards is refused. Value revisions behind the boundary are ignored.
mutable struct WindowCheckpoint
    filter::Any
    start::Int
end
WindowCheckpoint() = WindowCheckpoint(nothing, 1)

# Returns `(filter_or_nothing, mode, steps_advanced)` with mode `:full_history`, `:anchored` or `:advanced`.
function _prepare_window_checkpoint!(cp::WindowCheckpoint, build_anchor, p, ys, loss_range)
    new_start = first(loss_range)
    if new_start == 1
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
function _candidate_from_checkpoint(cp::AugmentedEnsembleKalmanFilter, _build, _T, p)
    candidate = deepcopy(cp)
    candidate.p = p
    return candidate
end

_observation_vector(y::Real) = [Float64(y)]
_observation_vector(y::AbstractVector) = collect(Float64, y)
_observation_vector(::Missing) = missing
_observation_vectors(ys::AbstractVector) = [_observation_vector(y) for y in ys]

# Fitted observation means: each filtered state mean through its spec's `observation_mean` at its
# own model time. A Vector for one spec, a `[time, observation]` matrix otherwise.
function _fitted_observation_means(specs, accumulators, extract, xt, hyper, dt, range)
    means = [
        observation_mean(spec, compute_true_mean(spec, xt[k], accumulators), extract(xt[k]), hyper, (k - 1) * dt)
            for k in range, spec in specs
    ]
    return length(specs) == 1 ? vec(means) : means
end

function _symmetrize_covariance(R)
    M = Matrix(R)
    return (M .+ M') ./ 2 + 1.0e-8 * Diagonal(ones(eltype(M), size(M, 1)))
end

_cloud_sd(values) = length(values) < 2 ? 0.0 : Float64(std(values))

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
    build_inference(filter, hyper, model::EpiModel; dt, supersample = 2, n_ahead, n_draws = 2000,
                    rng = Random.default_rng(), kwargs...) -> InferenceEngine
    build_inference(cfg::RunConfig, model::EpiModel; rng = Random.default_rng())

Build the inference engine for a `(filter, hyper)` pairing: [`UKF`](@ref) + [`Optimise`](@ref),
[`PF`](@ref) + [`LiuWest`](@ref) or [`EnKF`](@ref) + [`EKP`](@ref). Drive it with
[`fit_forecast!`](@ref). The `RunConfig` form reads `dt`, `supersample`, `n_ahead` and `n_draws`
from the run config.
"""
build_inference(filter::StateFilter, hyper::HyperMethod, ::EpiModel; kwargs...) = throw(
    ArgumentError("build_inference is not implemented for $(nameof(typeof(filter))) + $(nameof(typeof(hyper)))")
)

build_inference(cfg::RunConfig, model::EpiModel; rng = Random.default_rng()) = build_inference(
    cfg.filter, cfg.hyper, model;
    dt = cfg.step_days, supersample = cfg.supersample, n_ahead = cfg.n_ahead, n_draws = cfg.n_draws, rng,
)

"""
    fit_forecast!(engine, observations, forecast_number; update_range = eachindex(observations),
                  emit_forecast = true) -> (; quantiles, fitted_means, summary, samples)

Assimilate `observations`, one entry per slot of the regular `dt` grid (a number or a vector, or
`missing` where there is no observation: the filter then predicts through the slot without
correcting), and forecast `n_ahead` steps ahead. `forecast_number` counts fitted origins and
drives the re-optimisation cadence. The replay engines (UKF, EnKF) require the complete
`update_range`; the online PF engine keeps its cloud between calls and `update_range` must start
at the slot after the last one it assimilated (the default), so a replay is `reset!(engine.filter)`
followed by the full range. `fitted_means` covers `update_range` (a nowcast at a `missing` slot). `quantiles` is `[horizon, quantile]` (`[horizon, observation, quantile]` for a
multi-signal EnKF) and `samples` the predictive draws behind it (`nothing` for the analytic UKF);
both are `nothing` when `emit_forecast = false`. `summary` is a `(parameter, statistic, value)`
table of estimates and diagnostics.
"""
function fit_forecast! end
