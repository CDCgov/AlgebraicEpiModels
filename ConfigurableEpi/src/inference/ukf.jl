# UKF + Optimise: replay the complete series each origin; every `reopt_interval` origins maximise
# the differentiable filter marginal log-posterior over the learned hyperparameters. The forecast
# samples the filtered origin Gaussian and simulates the shared stochastic model forward.

mutable struct UKFOptimiseEngine{C <: EngineCore, S <: Tuple, AD, F} <: InferenceEngine
    const core::C
    const hyper::Optimise
    const stages::S
    const adtype::AD
    hyperparams::NamedTuple
    filter::F
    const checkpoint::WindowCheckpoint
    const rng_forecast::Random.Xoshiro
end

# The UKF at element type `T` (Dual when differentiating) and hyperparameters `hp`; the filter
# is preallocated and mutated in place, so a Float64 one cannot take a Dual update.
function _ukf(::Type{T}, a::Assembly, model::EpiModel, hp, dt) where {T}
    x0 = initial_state(model, hp)
    P0 = _initial_state_covariance(model.layout, x0; latent_variance = model.initial_latent_variance)
    P0 = model.initial_covariance(hp, x0, P0)
    return UnscentedKalmanFilter{false, false, true, true}(
        a.dynamics, a.measure, T.(Matrix(build_R1(model.layout))), Matrix{T}(I, a.n_noise, a.n_noise),
        MvNormal(T.(x0), T.(P0));
        p = hp, ny = a.n_obs, nu = 0, weight_params = TrivialParams(), Ts = dt, cholesky! = positive_cholesky!,
    )
end

"""
    SharedModelSimulator(dynamics, measurement)

The shared stochastic model as a [`forecast_ensemble`](@ref) simulator: the particle filter's
adapters over the jitter-free dynamics and the exact observation sampler.
"""
struct SharedModelSimulator{D, M}
    dynamics::D
    measurement::M
end

LLPF.sample_state(sim::SharedModelSimulator, x, u, p, t; noise = true) = sim.dynamics(x, u, p, t, noise)
LLPF.sample_measurement(sim::SharedModelSimulator, x, u, p, t; noise = true) = sim.measurement(x, u, p, t, noise)

_forecast_simulator(a::Assembly, model::EpiModel, rng::Random.AbstractRNG) = SharedModelSimulator(
    build_pf_dynamics(a.forecast_dynamics, model.layout; rng),
    build_pf_measurement(model.layout, model.observation, model.stochastic; rng),
)

"A negative origin-covariance eigenvalue below this fraction of the largest is warned about, not only clamped."
const ORIGIN_COVARIANCE_INDEFINITE_TOL = 1.0e-6

"""
Joint draws from the origin Gaussian `(x, R)`, with negative eigenvalues of `R` clamped to zero and
the `ode_range` slots floored at `STATE_FLOOR`; returns the draws and the repair diagnostics.
"""
function _draw_origin_states(rng::Random.AbstractRNG, x, R, n_draws::Integer; ode_range::AbstractUnitRange{<:Integer})
    E = eigen(Symmetric((R .+ R') ./ 2))
    lo, hi = extrema(E.values)
    lo < -ORIGIN_COVARIANCE_INDEFINITE_TOL * max(hi, 0.0) && @warn(
        "UKF: origin covariance is materially indefinite; its negative directions are clamped to zero variance",
        min_eigenvalue = lo, max_eigenvalue = hi,
    )
    factor = E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))
    mu = Vector{Float64}(x)
    floored = 0
    states = map(1:n_draws) do _
        draw = mu .+ factor * randn(rng, length(mu))
        ode = view(draw, ode_range)
        floored += any(<(STATE_FLOOR), ode)
        ode .= max.(ode, STATE_FLOOR)
        draw
    end
    return (; states, min_eigenvalue = lo, n_clamped = count(<(0), E.values), floored_fraction = floored / n_draws)
end

"""
    build_inference(filter::UKF, hyper::Optimise, model; dt, supersample = 2, integrator = Rk4, n_ahead,
                    n_draws = 2000, rng = Random.default_rng(), optimiser = DEFAULT_OPTIMISER_STAGES,
                    adtype = AutoForwardDiff())

`optimiser` is a single optimiser, a tuple of them, or `(optimiser, options)` pairs. `rng` seeds the
forecast draws; the filter and the optimiser are deterministic.
"""
function build_inference(
        filter::UKF, hyper::Optimise, model::EpiModel;
        dt, supersample = 2, integrator::Type{<:AbstractIntegrator} = Rk4, n_ahead, n_draws = 2000, rng = Random.default_rng(),
        optimiser = DEFAULT_OPTIMISER_STAGES, adtype = AutoForwardDiff(),
    )
    _validate(filter)
    _validate(hyper)
    core = EngineCore(model, EngineSettings(; dt, supersample, integrator, n_ahead, n_draws), filter.obs_jitter)
    model.stochastic.n_jumps == 0 ||
        throw(ArgumentError("a Gaussian filter cannot propagate the model's jump drivers; use PF"))
    ukf = _ukf(Float64, core.assembly, model, model.hyperparams, core.settings.dt)
    return UKFOptimiseEngine(
        core, hyper, optimiser_stages(optimiser), adtype, model.hyperparams, ukf, WindowCheckpoint(),
        Random.Xoshiro(rand(rng, UInt64)),
    )
end

function fit_forecast!(
        e::UKFOptimiseEngine, observations, forecast_number;
        update_range = eachindex(observations), emit_forecast::Bool = true,
    )
    T = _validate_fit(observations, update_range, forecast_number)
    _require_full_range(update_range, T, "UKF + Optimise")
    ys = _observation_vectors(observations)
    (; model, assembly, bundle) = e.core
    s, hyper = e.core.settings, e.hyper
    build(::Type{E}, hp) where {E} = _ukf(E, assembly, model, hp, s.dt)

    if (forecast_number - 1) % hyper.reopt_interval == 0
        warmup = _warmup_steps(model, s.dt)
        loss_range = _loss_range(T, hyper.window_length, warmup)
        _require_scored(ys, loss_range, warmup, "UKF + Optimise")
        current = e.hyperparams
        checkpoint, mode, steps = _prepare_window_checkpoint!(
            e.checkpoint, () -> build(Float64, current), current, ys, loss_range; warmup
        )
        cold = forecast_number == 1 || !hyper.warm_start
        start_from = hyper.warm_start ? e.hyperparams : model.hyperparams
        initial = NamedTuple{bundle.names}(Tuple(start_from[n] for n in bundle.names))
        # The prior term sits outside the `try`, so a diverged filter scores `Inf` with finite
        # partials pointing back toward the prior mode (a finite penalty could sit below the
        # feasible objective and reward divergence).
        objective = (u, _) -> begin
            log_prior = prior_logpdf(bundle, u)
            try
                hp = merge(model.hyperparams, constrained_values(bundle, u))
                kf = checkpoint === nothing ? build(eltype(u), hp) :
                    _candidate_from_checkpoint(checkpoint, build, eltype(u), hp)
                ll = checkpoint === nothing ? marginal_loglik(kf, ys, hp; warmup) : _filter_loglik!(kf, ys, hp, loss_range)
                value = -ll - log_prior
                isfinite(value) ? value : Inf - log_prior
            catch
                Inf - log_prior
            end
        end
        @info "UKF: hyperparameter optimisation starting" forecast_number observations = T loss_observations =
            length(loss_range) checkpoint = mode checkpoint_steps = steps parameters = collect(bundle.names) cold_start = cold
        result = optimize_hyperparams(
            objective, initial, bundle;
            stages = e.stages, adtype = e.adtype, options = (maxiters = cold ? hyper.maxiters_burnin : hyper.maxiters,),
        )
        e.hyperparams = merge(model.hyperparams, result.θ)
        @info "UKF: hyperparameter optimisation complete" forecast_number retcode = result.retcode log_posterior =
            result.ll estimates = result.θ
    end

    hp = e.hyperparams
    @info "UKF: filtering observation history" forecast_number observations = T
    kf = build(Float64, hp)
    solution = _filter_pass!(kf, ys, hp)
    e.filter = kf
    layout, extract = model.layout, model.stochastic.extract
    specs = _resolve_specs(model.observation, layout)
    fitted = _fitted_observation_means(specs, layout.accumulator_indices, extract, solution.xt, hp, s.dt, update_range)
    summary = _summary()
    for name in bundle.names
        push!(summary, (string(name), "estimate", Float64(hp[name])))
    end
    latents = _latent_moments(solution.xt, solution.Rt, layout.latent_range, update_range)
    emit_forecast || return ForecastResult(; fitted_means = fitted, summary, latents)

    @info "UKF: generating forecast" forecast_number draws = s.n_draws horizons = s.n_ahead
    rng = Random.Xoshiro(rand(e.rng_forecast, UInt64))
    draws = _draw_origin_states(rng, solution.xt[end], solution.Rt[end], s.n_draws; ode_range = 1:n_ode_states(layout))
    samples, latent_samples = forecast_ensemble(
        _forecast_simulator(assembly, model, rng), draws.states, hp;
        n_ahead = s.n_ahead, t0 = (T - 1) * s.dt, dt = s.dt, latent_range = layout.latent_range, n_obs = assembly.n_obs,
    )
    append_latent_spread!(summary, layout.latent_names, _latent_sd(latent_samples))
    push!(summary, ("ukf", "origin_cov_min_eigenvalue", draws.min_eigenvalue))
    push!(summary, ("ukf", "origin_cov_clamped_dims", draws.n_clamped))
    push!(summary, ("ukf", "origin_draws_floored_frac", draws.floored_fraction))
    append_latent_audit!(summary, layout.latent_names, solution.xt, layout.latent_range; dt = s.dt)
    return ForecastResult(; quantiles = forecast_quantiles(samples), fitted_means = fitted, summary, samples, latents)
end
