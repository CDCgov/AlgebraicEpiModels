# EnKF + EKP: an ensemble filter whose Monte-Carlo likelihood has no usable derivative, so the
# static parameters are calibrated by an OUTER ensemble Kalman inversion in which every candidate
# is scored by a complete inner replay. Every candidate filter is constructed from the same integer
# seed (common random numbers), so `G(θ)` depends on θ alone.

const _EKP_TARGET = [0.0]
const _EKP_LOSS_COV = Matrix(1.0I, 1, 1)

"""
    EnKFEKPEngine

EnKF + EKP: replays the complete series each origin and recalibrates every `reopt_interval`
origins, warm-starting from the previous outer ensemble.
"""
mutable struct EnKFEKPEngine{M <: EpiModel, A <: Assembly, B <: ParameterPriorBundle, F} <: InferenceEngine
    const model::M
    const assembly::A
    const settings::EngineSettings
    const filter_cfg::EnKF
    const hyper::EKP
    const bundle::B
    const inner_seed::UInt64
    const rng_outer::Random.Xoshiro
    const rng_forecast::Random.Xoshiro
    hyperparams::NamedTuple
    filter::F
    ensemble_unconstrained::Union{Nothing, Matrix{Float64}}
    ensemble_constrained::Union{Nothing, Matrix{Float64}}
    iterations::Int
    algorithmic_time::Float64
    const warned_positive_ll::Base.RefValue{Bool}
    const checkpoint::WindowCheckpoint
end

function _enkf(a::Assembly, model::EpiModel, hp, dt, cfg::EnKF, seed)
    x0 = initial_state(model, hp)
    P0 = _initial_state_covariance(
        model.layout, x0;
        latent_variance = model.initial_latent_variance, accumulator_variance = model.initial_accumulator_variance,
    )
    return AugmentedEnsembleKalmanFilter(
        a.dynamics, a.measure, build_R1(model.layout), Diagonal(ones(a.n_noise)), MvNormal(collect(float.(x0)), P0), cfg.n_ensemble;
        nu = 0, ny = a.n_obs, p = hp, Ts = dt, inflation = cfg.inflation, threads = cfg.threads, rng = Random.Xoshiro(seed),
    )
end

# Both ensembles span `members - 1` dimensions; a rank-deficient one produces spurious
# cross-covariances that surface as ODE divergence rather than a rank error.
function _check_ensemble_rank(filter::EnKF, hyper::EKP, model::EpiModel)
    nx = model.layout.total_dim
    filter.n_ensemble > nx || throw(
        ArgumentError(
            "EnKF sample covariance is rank-deficient: n_ensemble = $(filter.n_ensemble) spans " *
                "$(filter.n_ensemble - 1) dimensions but the state has $nx; raise filter.enkf.n_ensemble",
        )
    )
    n_learned = length(model.priors)
    hyper.n_ensemble > n_learned || throw(
        ArgumentError(
            "EKP outer ensemble is rank-deficient: n_ensemble = $(hyper.n_ensemble) for $n_learned learned " *
                "parameters $(collect(keys(model.priors))); raise hyper.ekp.n_ensemble",
        )
    )
    return nothing
end

function build_inference(
        filter::EnKF, hyper::EKP, model::EpiModel;
        dt, supersample = 2, n_ahead, n_draws = 2000, rng = Random.default_rng(),
    )
    _validate(filter)
    _validate(hyper)
    filter.threads && hyper.threads && throw(
        ArgumentError("nested ensemble threading is unsupported: disable `filter.enkf.threads` when `hyper.ekp.threads` is enabled")
    )
    model.stochastic.n_jumps == 0 ||
        throw(ArgumentError("a Gaussian filter cannot propagate the model's jump drivers; use PF"))
    settings = EngineSettings(; dt, supersample, n_ahead, n_draws)
    assembly = Assembly(model, settings, 0.0)
    length(model.layout.accumulator_indices) == assembly.n_obs || throw(
        ArgumentError("the ensemble path maps one accumulator per observation; the layout has $(length(model.layout.accumulator_indices)) for $(assembly.n_obs) observations")
    )
    _check_ensemble_rank(filter, hyper, model)
    # Three streams off the caller's rng: the outer ensemble, the inner replays (a seed re-used
    # per candidate) and forecast simulation.
    stream_seed = rand(rng, UInt64)
    inner_seed = stream_seed + 0x01
    enkf = _enkf(assembly, model, model.hyperparams, settings.dt, filter, inner_seed)
    return EnKFEKPEngine(
        model, assembly, settings, filter, hyper, ParameterPriorBundle(model.priors), inner_seed,
        Random.Xoshiro(stream_seed), Random.Xoshiro(stream_seed + 0x02), model.hyperparams, enkf,
        nothing, nothing, 0, 0.0, Ref(false), WindowCheckpoint(),
    )
end

# `TransformInversion`'s default scheduler stops once accumulated algorithmic time reaches 1,
# turning the configured iteration count into an upper bound; a fixed unit step diverges on the
# unscaled deviance loss. Keep the adaptive step and only change the termination action.
function _new_ekp(u0, prior, inflation, rng)
    process = TransformInversion(prior; default_multiplicative_inflation = inflation)
    return EnsembleKalmanProcess(
        u0, _EKP_TARGET, _EKP_LOSS_COV, process;
        rng, scheduler = DataMisfitController(terminate_at = Inf, on_terminate = "continue"),
        failure_handler_method = SampleSuccGauss(), verbose = false,
    )
end

function _outer_ekp_threads_enabled(requested::Bool; nthreads::Integer = Threads.nthreads())
    if requested && nthreads == 1
        @warn "EKP: outer threading requested but Julia has only one thread; candidates will be evaluated serially"
        return false
    end
    return requested
end

# Score every outer candidate. Failures become NaN (EKP's resampling signal) and are reported
# by the caller after all workers have joined.
function _evaluate_ekp_candidates(score, phi; threads::Bool)
    members = axes(phi, 2)
    G = Matrix{Float64}(undef, 1, length(members))
    positive_logliks = Vector{Union{Nothing, Float64}}(nothing, length(members))
    failures = Vector{Any}(nothing, length(members))
    function evaluate!(member)
        try
            ll = score(view(phi, :, member))
            if !isfinite(ll)
                G[1, member] = NaN
                failures[member] = (ErrorException("candidate returned non-finite log-likelihood: $ll"), nothing)
                return nothing
            end
            ll > 0 && (positive_logliks[member] = Float64(ll))
            G[1, member] = sqrt(max(-2 * ll, eps()))
        catch error
            G[1, member] = NaN
            failures[member] = (error, catch_backtrace())
        end
        return nothing
    end
    if threads
        previous_blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            Threads.@threads for member in members
                evaluate!(member)
            end
        finally
            BLAS.set_num_threads(previous_blas_threads)
        end
    else
        foreach(evaluate!, members)
    end
    return G, positive_logliks, failures
end

function _report_ekp_candidate_diagnostics!(warned_positive_ll, positive_logliks, failures; forecast_number, iteration)
    positive_member = findfirst(!isnothing, positive_logliks)
    if positive_member !== nothing && !warned_positive_ll[]
        warned_positive_ll[] = true
        @warn "EKP: positive filter log-likelihood; the deviance loss saturates and cannot distinguish candidates. Check the observation scale." forecast_number iteration member =
            positive_member log_likelihood = positive_logliks[positive_member]
    end
    for member in eachindex(failures)
        failure = failures[member]
        failure === nothing && continue
        error, backtrace = failure
        if backtrace === nothing
            @warn "EKP: outer candidate failed" forecast_number iteration member reason = sprint(showerror, error)
        else
            @warn "EKP: outer candidate failed" forecast_number iteration member exception = (error, backtrace)
        end
    end
    return nothing
end

function fit_forecast!(
        e::EnKFEKPEngine, observations, forecast_number;
        update_range = eachindex(observations), emit_forecast::Bool = true,
    )
    T = _validate_fit(observations, update_range, forecast_number)
    _require_full_range(update_range, T, "EnKF + EKP")
    ys = _observation_vectors(observations)
    model, s, hyper, bundle = e.model, e.settings, e.hyper, e.bundle
    names = bundle.names
    build(hp) = _enkf(e.assembly, model, hp, s.dt, e.filter_cfg, e.inner_seed)

    if (forecast_number - 1) % hyper.reopt_interval == 0
        loss_range = _loss_range(T, hyper.window_length)
        current = e.hyperparams
        checkpoint, mode, steps = _prepare_window_checkpoint!(e.checkpoint, () -> build(current), current, ys, loss_range)
        outer_threads = _outer_ekp_threads_enabled(hyper.threads)
        function score(theta)
            hp = merge(model.hyperparams, NamedTuple{names}(Tuple(theta)))
            kf = checkpoint === nothing ? build(hp) : _candidate_from_checkpoint(checkpoint, nothing, Float64, hp)
            return checkpoint === nothing ? marginal_loglik(kf, ys, hp) : _filter_loglik!(kf, ys, hp, loss_range)
        end
        # Resuming from the previous final ENSEMBLE carries the spread too, which is what lets
        # `iterations` be a fraction of `burnin_iterations`; a cold start gets the burn-in budget.
        resume = hyper.warm_start ? e.ensemble_unconstrained : nothing
        n_iterations = (forecast_number == 1 || resume === nothing) ? hyper.burnin_iterations : hyper.iterations
        u0 = resume === nothing ? construct_initial_ensemble(e.rng_outer, bundle.prior, hyper.n_ensemble) : resume
        ekp = _new_ekp(u0, bundle.prior, hyper.inflation, e.rng_outer)
        @info "EKP: calibration starting" forecast_number observations = T loss_observations = length(loss_range) checkpoint =
            mode checkpoint_steps = steps parameters = collect(names) outer_ensemble = hyper.n_ensemble iterations =
            n_iterations warm_started = resume !== nothing outer_threads
        realised = 0
        for iteration in 1:n_iterations
            phi = get_ϕ_final(bundle.prior, ekp)   # [n_params, n_ensemble], constrained
            evaluation = @timed _evaluate_ekp_candidates(score, phi; threads = outer_threads)
            G, positive_logliks, failures = evaluation.value
            _report_ekp_candidate_diagnostics!(e.warned_positive_ll, positive_logliks, failures; forecast_number, iteration)
            n_failed = count(isnan, view(G, 1, :))
            @info "EKP: candidate evaluation complete" forecast_number iteration candidates = size(G, 2) failed =
                n_failed wall_time_seconds = evaluation.time
            if n_failed == size(G, 2)
                first_failure = findfirst(!isnothing, failures)
                reason = first_failure === nothing ? "" :
                    " First failure (member $first_failure): " * sprint(showerror, first(failures[first_failure]))
                error(
                    "EKP: every outer candidate failed at forecast origin $forecast_number, iteration $iteration. " *
                        "Constrained ensemble: $(NamedTuple{names}(Tuple(view(phi, :, 1))))." * reason,
                )
            end
            terminate = update_ensemble!(ekp, G)
            realised += 1
            if terminate === true
                @warn "EKP: scheduler halted before the configured iteration count" forecast_number realised configured = n_iterations
                break
            end
        end
        phi_final = Matrix{Float64}(get_ϕ_final(bundle.prior, ekp))
        # The mean of the CONSTRAINED ensemble, not ϕ(mean(u)).
        e.hyperparams = merge(model.hyperparams, NamedTuple{names}(Tuple(vec(mean(phi_final, dims = 2)))))
        e.ensemble_unconstrained = Matrix{Float64}(get_u_final(ekp))
        e.ensemble_constrained = phi_final
        e.iterations = realised
        e.algorithmic_time = sum(get_Δt(ekp))
        @info "EKP: calibration complete" forecast_number realised algorithmic_time = e.algorithmic_time estimates = e.hyperparams[names]
    end

    hp = e.hyperparams
    @info "EnKF: filtering observation history" forecast_number observations = T members = e.filter_cfg.n_ensemble
    kf = build(hp)
    # The forecast starts from the CORRECTED ensemble at the origin; the pass's last act is
    # `predict!`, so `kf.ensemble` afterwards is one step past it.
    solution = _filter_pass!(kf, ys, hp; at_origin = emit_forecast ? (f -> deepcopy(f.ensemble)) : Returns(nothing))
    isfinite(solution.ll) ||
        error("EnKF: filtering pass produced a non-finite log-likelihood at forecast origin $forecast_number under $(hp[names])")
    e.filter = kf

    layout = model.layout
    specs = _resolve_specs(model.observation, layout)
    fitted = _fitted_observation_means(specs, layout.accumulator_indices, model.stochastic.extract, solution.xt, hp, s.dt, update_range)
    # The outer spread is a calibration diagnostic (how converged the inversion is), not a
    # posterior interval, hence the `ekp_` prefix.
    summary = _summary()
    push!(summary, ("ekp", "iterations", Float64(e.iterations)))
    push!(summary, ("ekp", "algorithmic_time", e.algorithmic_time))
    for (i, name) in enumerate(names)
        push!(summary, (string(name), "estimate", Float64(hp[name])))
        e.ensemble_constrained === nothing && continue
        values = collect(view(e.ensemble_constrained, i, :))
        for (stat, q) in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))
            push!(summary, (string(name), "ekp_" * stat, quantile(values, q)))
        end
    end
    emit_forecast || return (; quantiles = nothing, fitted_means = fitted, summary, samples = nothing)

    @info "EnKF: generating forecast ensemble" forecast_number draws = s.n_draws horizons = s.n_ahead
    corrected = solution.origin
    initial_states = corrected[rand(e.rng_forecast, eachindex(corrected), s.n_draws)]
    forecast_filter = deepcopy(kf)
    Random.seed!(forecast_filter.rng, rand(e.rng_forecast, UInt64))
    samples, latent_samples = forecast_ensemble(
        forecast_filter, initial_states, hp;
        n_ahead = s.n_ahead, t0 = (T - 1) * s.dt, dt = s.dt, latent_range = layout.latent_range, n_obs = e.assembly.n_obs,
    )
    latent_sd = [_cloud_sd(view(latent_samples, h, :, l)) for h in 1:s.n_ahead, l in axes(latent_samples, 3)]
    append_latent_spread!(summary, layout.latent_names, latent_sd)
    append_latent_audit!(summary, layout.latent_names, solution.xt, layout.latent_range; dt = s.dt)
    @info "EnKF: forecast complete" forecast_number log_likelihood = solution.ll
    return (; quantiles = forecast_quantiles(samples), fitted_means = fitted, summary, samples)
end
