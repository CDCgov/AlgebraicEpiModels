# UKF + Optimise: replay the complete series each origin; every `reopt_interval` origins maximise
# the differentiable filter marginal log-posterior over the learned hyperparameters.

mutable struct UKFOptimiseEngine{M <: EpiModel, A <: Assembly, B <: ParameterPriorBundle, S <: Tuple, AD, F} <: InferenceEngine
    const model::M
    const assembly::A
    const settings::EngineSettings
    const hyper::Optimise
    const bundle::B
    const stages::S
    const adtype::AD
    hyperparams::NamedTuple
    filter::F
    const checkpoint::WindowCheckpoint
end

# The UKF at element type `T` (Dual when differentiating) and hyperparameters `hp`; the filter
# is preallocated and mutated in place, so a Float64 one cannot take a Dual update.
function _ukf(::Type{T}, a::Assembly, model::EpiModel, hp, dt) where {T}
    x0 = initial_state(model, hp)
    P0 = _initial_state_covariance(model.layout, x0; latent_variance = model.initial_latent_variance)
    return UnscentedKalmanFilter{false, false, true, true}(
        a.dynamics, a.measure, T.(Matrix(build_R1(model.layout))), Matrix{T}(I, a.n_noise, a.n_noise),
        MvNormal(T.(x0), T.(P0));
        p = hp, ny = a.n_obs, nu = 0, weight_params = TrivialParams(), Ts = dt, cholesky! = positive_cholesky!,
    )
end

"""
    build_inference(filter::UKF, hyper::Optimise, model; dt, supersample = 2, n_ahead, n_draws = 2000,
                    rng = nothing, optimiser = DEFAULT_OPTIMISER_STAGES, adtype = AutoForwardDiff())

`optimiser` is a single optimiser, a tuple of them, or `(optimiser, options)` pairs; `rng` is unused.
"""
function build_inference(
        filter::UKF, hyper::Optimise, model::EpiModel;
        dt, supersample = 2, n_ahead, n_draws = 2000, rng = nothing,
        optimiser = DEFAULT_OPTIMISER_STAGES, adtype = AutoForwardDiff(),
    )
    _validate(filter)
    _validate(hyper)
    settings = EngineSettings(; dt, supersample, n_ahead, n_draws)
    assembly = Assembly(model, settings, filter.obs_jitter)
    assembly.n_obs == 1 || throw(
        ArgumentError("UKF + Optimise produces one count-forecast matrix and needs exactly one observation signal; got $(assembly.n_obs)")
    )
    model.stochastic.n_jumps == 0 ||
        throw(ArgumentError("a Gaussian filter cannot propagate the model's jump drivers; use PF"))
    ukf = _ukf(Float64, assembly, model, model.hyperparams, settings.dt)
    return UKFOptimiseEngine(
        model, assembly, settings, hyper, ParameterPriorBundle(model.priors), optimiser_stages(optimiser), adtype,
        model.hyperparams, ukf, WindowCheckpoint(),
    )
end

function fit_forecast!(
        e::UKFOptimiseEngine, observations, forecast_number;
        update_range = eachindex(observations), emit_forecast::Bool = true,
    )
    T = _validate_fit(observations, update_range, forecast_number)
    _require_full_range(update_range, T, "UKF + Optimise")
    ys = _observation_vectors(observations)
    model, s, hyper, bundle = e.model, e.settings, e.hyper, e.bundle
    build(::Type{E}, hp) where {E} = _ukf(E, e.assembly, model, hp, s.dt)

    if (forecast_number - 1) % hyper.reopt_interval == 0
        loss_range = _loss_range(T, hyper.window_length)
        current = e.hyperparams
        checkpoint, mode, steps = _prepare_window_checkpoint!(
            e.checkpoint, () -> build(Float64, current), current, ys, loss_range
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
                ll = checkpoint === nothing ? marginal_loglik(kf, ys, hp) : _filter_loglik!(kf, ys, hp, loss_range)
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
    solution = forward_trajectory(kf, fill(_NO_INPUT, T), ys, hp)
    e.filter = kf
    layout, extract = model.layout, model.stochastic.extract
    spec = only(_resolve_specs(model.observation, layout))
    fitted = _fitted_observation_means((spec,), layout.accumulator_indices, extract, solution.xt, hp, s.dt, update_range)
    summary = _summary()
    for name in bundle.names
        push!(summary, (string(name), "estimate", Float64(hp[name])))
    end
    emit_forecast || return (; quantiles = nothing, fitted_means = fitted, summary, samples = nothing)

    @info "UKF: generating forecast" forecast_number horizons = s.n_ahead
    t0 = (T - 1) * s.dt
    means, covs = forecast_states(
        kf, solution.xt[end], _symmetrize_covariance(solution.Rt[end]); n_ahead = s.n_ahead, t0, dt = s.dt, p = hp,
    )
    acc = only(layout.accumulator_indices)
    quantiles = Matrix{Float64}(undef, s.n_ahead, length(DEFAULT_QS))
    for h in 1:s.n_ahead
        m = observation_gaussian_moments(spec, means[h][acc], covs[h][acc, acc], extract(means[h]), hp, t0 + h * s.dt)
        dist = Normal(m.mean, sqrt(max(m.var, 1.0e-12)))
        for (k, q) in enumerate(DEFAULT_QS)
            quantiles[h, k] = max(0.0, quantile(dist, q))
        end
    end
    latent_sd = [sqrt(max(covs[h][i, i], 0.0)) for h in 1:s.n_ahead, i in layout.latent_range]
    append_latent_spread!(summary, layout.latent_names, latent_sd)
    append_latent_audit!(summary, layout.latent_names, solution.xt, layout.latent_range; dt = s.dt)
    return (; quantiles, fitted_means = fitted, summary, samples = nothing)
end
