# ============================================================================
# ENSEMBLE FILTER + OUTER EKP CALIBRATION
# ============================================================================
#
# The `(EnKF, EKPCalibration)` inference path. Where the UKF path differentiates the filter's
# marginal log-posterior and hands it to an optimizer, this one cannot: an ensemble filter
# resamples from `d0`, so its likelihood is a Monte-Carlo estimate with no usable derivative.
# Calibration therefore moves OUTWARD — an ensemble of candidate parameter vectors, each scored
# by an isolated inner filter replay, updated by ensemble Kalman inversion. That replay starts
# from the initial distribution for full-history scoring, or from a deep copy of the retained
# left-window checkpoint when `window_length` is active.
#
# Three things make that honest rather than merely derivative-free:
#
#  * COMMON RANDOM NUMBERS. Every candidate's filter is constructed from the SAME integer seed,
#    so the inner noise stream is identical across candidates and across iterations. Without
#    this, `G(θ)` differs between candidates by Monte-Carlo noise of the same order as the
#    parameter signal, and the outer ensemble chases the noise. Note the seed is re-used to
#    CONSTRUCT each filter, never shared as a live RNG object — a shared stream would make
#    candidate 7's likelihood depend on candidates 1..6, which breaks common random numbers,
#    repeatability and order-independence in one go.
#  * AN ISOLATED FILTER PER CANDIDATE. Full-history scoring constructs a fresh filter because
#    `build_x0` makes the initial state a function of the parameters (the equilibrium seed
#    `S(0) = N / R0` is only coherent at the candidate's `R0`). Windowed scoring instead deep
#    copies the historical checkpoint once per candidate; pre-window state is intentionally fixed
#    and the candidate parameters apply only to the scored suffix.
#  * AN EXPLICIT SCHEDULER. See `_new_ekp` below; the default one silently turns the configured
#    iteration count into an upper bound.

# EKP inverts toward a target; the loss below is a deviance residual, so the target is zero and
# the loss covariance is the identity. Both are built once — EKP wants a real `Matrix`, not `I`.
const _EKP_TARGET = [0.0]
const _EKP_LOSS_COV = Matrix(1.0I, 1, 1)

"""
    _enkf_backend(dynamics, measure, layout, x0_state, dt, n_observations,
                  n_measurement_noise, base_hyperparams, initial_latent_variance,
                  n_ensemble, inflation, threads, seed)

Assemble the concrete [`AugmentedEnsembleKalmanFilter`](@ref) used by the `EnKF` +
[`EKPCalibration`](@ref) combination.

Unlike [`_ukf_backend`](@ref) there is no element-type parameter: nothing on this path is
differentiated, so the filter is always `Float64`. `seed` is an integer, not an RNG — see the
common-random-numbers note at the top of this file.

Note the process-noise dimension: `build_R1(layout)` is `nw × nw` with `nw = n_latent + n_signals`,
which is smaller than the state dimension. That is the augmented filter's `nw != nx` case, and it
is why the additive ensemble filter cannot represent this model — the equivalent additive process
covariance would be singular.
"""
function _enkf_backend(
        dynamics,
        measure,
        layout,
        x0_state,
        dt,
        n_observations,
        n_measurement_noise,
        base_hyperparams,
        initial_latent_variance,
        initial_accumulator_variance,
        n_ensemble,
        inflation,
        threads,
        seed,
    )
    _check_initial_state(layout, x0_state)
    R1 = build_R1(layout)                                    # Diagonal, nw × nw, nw != nx
    R2 = Diagonal(ones(n_measurement_noise))
    P0 = _initial_state_covariance(
        layout, x0_state;
        latent_variance = initial_latent_variance,
        # Without this the first correction is a ~1e6-sigma innovation and the ensemble's spurious
        # cross-covariance blows the compartments up; see `_initial_state_covariance`.
        accumulator_variance = initial_accumulator_variance,
    )
    return AugmentedEnsembleKalmanFilter(
        dynamics, measure, R1, R2, MvNormal(collect(float.(x0_state)), P0), n_ensemble;
        nu = 0,
        ny = n_observations,
        p = base_hyperparams,
        Ts = float(dt),
        inflation = inflation,
        threads = threads,
        rng = Random.Xoshiro(seed),
    )
end

# A fresh EKP process, either from the prior or warm-started from a previous final ensemble.
#
# THE SCHEDULER IS LOAD-BEARING, in both directions, and the two requirements pull against each
# other:
#
#  * `TransformInversion`'s default is `DataMisfitController(terminate_at = 1, on_terminate =
#    "stop")`, which stops stepping once accumulated algorithmic time reaches 1. Left alone, the
#    configured `iterations` silently becomes an upper bound and the run performs an unpredictable
#    number of updates.
#  * But a FIXED step (`DefaultScheduler(1.0)`) diverges immediately here. The loss is a deviance
#    residual, `sqrt(-2·ll)`, which on real weekly counts is order 10²–10³ against a unit loss
#    covariance — so a unit step is hundreds of misfit standard deviations, and one update throws
#    the whole ensemble onto the constraint boundaries (`R0_baseline = Inf`, `Rt_rho = 0`). Every
#    candidate then fails and the guard in `fit_forecast!` trips. Measured, not hypothesised.
#
# `DataMisfitController` is what makes an unscaled deviance loss usable at all: it sizes each step
# from the realised data misfit (Iglesias & Yan 2021) instead of trusting the loss covariance to be
# calibrated. So keep it, and change only the termination ACTION — `on_terminate = "continue"` with
# `terminate_at = Inf` keeps the adaptive step and performs exactly the configured number of
# updates. The caller still honours a `terminate` return and records both the realised iteration
# count and the accumulated algorithmic time, so a future scheduler change cannot pass unnoticed.
function _new_ekp(u0, prior, inflation, rng)
    process = TransformInversion(
        prior; default_multiplicative_inflation = inflation,
    )
    return EnsembleKalmanProcess(
        u0, _EKP_TARGET, _EKP_LOSS_COV, process;
        rng = rng,
        scheduler = DataMisfitController(terminate_at = Inf, on_terminate = "continue"),
        failure_handler_method = SampleSuccGauss(),
        verbose = false,
    )
end

function _check_ensemble_threading(inner_threads::Bool, outer_threads::Bool)
    inner_threads && outer_threads && throw(
        ArgumentError(
            "nested ensemble threading is unsupported: disable `filter.enkf.threads` " *
                "when `hyper.ekp.threads` is enabled",
        )
    )
    return nothing
end

function _outer_ekp_threads_enabled(
        requested::Bool; nthreads::Integer = Threads.nthreads()
    )
    if requested && nthreads == 1
        @warn(
            "EKP: outer threading requested but Julia has only one thread; " *
                "candidate evaluation will run serially",
        )
        return false
    end
    return requested
end

# Score an outer ensemble without allowing worker tasks to mutate shared diagnostics. Each worker
# owns one matrix cell and one slot in each ordinary Vector (not a packed BitVector). Exceptions
# and positive likelihoods are reported by the caller only after every worker has joined.
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
                failures[member] = (
                    ErrorException("candidate returned non-finite log-likelihood: $ll"),
                    nothing,
                )
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
        for member in members
            evaluate!(member)
        end
    end

    return G, positive_logliks, failures
end

function _report_ekp_candidate_diagnostics!(
        warned_positive_ll,
        positive_logliks,
        failures;
        forecast_number,
        iteration,
    )
    positive_member = findfirst(value -> value !== nothing, positive_logliks)
    if positive_member !== nothing && !warned_positive_ll[]
        warned_positive_ll[] = true
        @warn(
            "EKP: positive filter log-likelihood — the deviance loss saturates " *
                "and cannot distinguish candidates. Check the observation scale.",
            forecast_number,
            iteration,
            member = positive_member,
            log_likelihood = positive_logliks[positive_member],
        )
    end

    for member in eachindex(failures)
        failure = failures[member]
        failure === nothing && continue
        error, backtrace = failure
        if backtrace === nothing
            @warn(
                "EKP: outer candidate failed",
                forecast_number,
                iteration,
                member,
                reason = sprint(showerror, error),
            )
        else
            @warn(
                "EKP: outer candidate failed",
                forecast_number,
                iteration,
                member,
                exception = (error, backtrace),
            )
        end
    end
    return nothing
end

"""
    build_inference(
        filter_method::EnKF, hyper_method::EKPCalibration,
        petri_vf!, layout, stochastic_dynamics, obs_model,
        base_hyperparams, x0_state;
        n_ahead, reopt_interval=1, window_length=nothing, warm_start=true,
        n_draws=2000, rng=Random.default_rng(),
        initial_latent_variance=(;), build_x0=_ -> x0_state,
    ) -> (enkf, fit_forecast!)

Build the ensemble-filter inference path with outer EKP calibration. At every origin the live
filter replays the complete series; every `reopt_interval` origins the static parameters are
recalibrated by ensemble Kalman inversion (`burnin_iterations` at the first origin, `iterations`
afterwards), warm-starting from the previous final ensemble. `window_length` optionally limits
candidate losses to the most recent observations, starting from a rolling in-memory filter
checkpoint. `warm_start = false` instead draws a fresh ensemble from the prior at every
recalibration and gives each one the `burnin_iterations` budget. Non-recalibration origins retain
the previous constrained ensemble mean.

The returned `fit_forecast!` honours the shared contract; see [`build_inference`](@ref).
"""
function build_inference(
        filter_method::EnKF,
        hyper_method::EKPCalibration,
        petri_vf!,
        layout,
        stochastic_dynamics,
        obs_model,
        base_hyperparams,
        x0_state;
        n_ahead::Integer,
        reopt_interval::Integer = 1,
        window_length::Union{Nothing, Integer} = nothing,
        warm_start::Bool = true,
        n_draws::Integer = 2000,
        rng = Random.default_rng(),
        initial_latent_variance::NamedTuple = (;),
        initial_accumulator_variance::NamedTuple = (;),
        build_x0 = _ -> x0_state,
    )
    _check_ensemble_threading(filter_method.threads, hyper_method.threads)
    n_ahead > 0 || throw(ArgumentError("n_ahead must be positive, got $n_ahead"))
    reopt_interval > 0 || throw(
        ArgumentError("reopt_interval must be positive, got $reopt_interval")
    )
    validated_window_length = _validate_window_length(window_length)
    n_draws > 0 || throw(ArgumentError("n_draws must be positive, got $n_draws"))
    outer_threads = _outer_ekp_threads_enabled(hyper_method.threads)
    dynamics, measure, n_observations, n_measurement_noise =
        build_dynamics_and_measurement(
        petri_vf!, layout, obs_model, stochastic_dynamics;
        dt = filter_method.dt,
        supersample = filter_method.supersample,
        obs_jitter = filter_method.obs_jitter,
    )
    # NOTE the absence of `_check_single_signal`. The UKF and PF paths still call it (at
    # build_inference.jl's `:499` and `:688`), which is the cheapest possible guarantee that
    # generalising the results contract cannot have changed their output; the ensemble path is the
    # one that emits `[horizon, observation, quantile]` and `[time, observation]` when the model has
    # more than one signal, and the historical two-dimensional shapes when it has exactly one.
    build_backend(hp, seed) = _enkf_backend(
        dynamics,
        measure,
        layout,
        build_x0(hp),
        filter_method.dt,
        n_observations,
        n_measurement_noise,
        hp,
        initial_latent_variance,
        initial_accumulator_variance,
        filter_method.n_ensemble,
        filter_method.inflation,
        filter_method.threads,
        seed,
    )

    priors = hyper_method.priors
    learned_names = priors.names
    # One accumulator per observation signal, in signal order — not `only(...)`, which is what
    # confined the older paths to a single signal.
    accumulators = collect(layout.accumulator_indices)
    length(accumulators) == n_observations || throw(
        ArgumentError(
            "layout has $(length(accumulators)) accumulators for $n_observations " *
                "observations; the ensemble path maps one accumulator per signal",
        )
    )
    # Resolved observation specs, in observation order: the fitted means below go through each
    # spec's `observation_mean` at its own model time, as the measurement function does.
    observation_specs = _resolve_obs_specs(obs_model, Val(n_signals(layout)))

    # Three deterministic streams off the caller's rng, so a rerun at the same `seed` reproduces
    # and the three roles cannot interfere: the outer parameter ensemble, the common-random-number
    # inner replays (a SEED, re-used per candidate — see the file header), and forecast simulation.
    stream_seed = rand(rng, UInt64)
    rng_outer = Random.Xoshiro(stream_seed)
    inner_seed = stream_seed + 0x01
    rng_forecast = Random.Xoshiro(stream_seed + 0x02)

    enkf = build_backend(base_hyperparams, inner_seed)

    # Calibration state carried across origins.
    theta_mean = Ref((; (name => base_hyperparams[name] for name in learned_names)...))
    # `last_unconstrained` warm-starts the next EKP process; `last_constrained` is kept purely so
    # the spread diagnostic can be reported on non-recalibration origins too, where it is the
    # retained spread from the most recent calibration rather than a fresh one.
    last_unconstrained = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
    last_constrained = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
    # Reported in the summary so "EKP ran the configured number of iterations" is an OBSERVABLE
    # rather than an assumption — the scheduler is capable of stopping early (see `_new_ekp`).
    last_iterations = Ref(0)
    last_algorithmic_time = Ref(0.0)
    warned_positive_ll = Ref(false)
    window_checkpoint = Ref{Union{Nothing, typeof(enkf)}}(nothing)
    window_checkpoint_start = Ref(1)
    window_checkpoint_date = Ref{Union{Nothing, Date}}(nothing)

    function fit_forecast!(
            enkf, asof, update_range, us, forecast_number;
            emit_forecast::Bool = true, sample_callback = nothing,
        )
        T = _validate_fit_inputs(asof, update_range, us)
        update_range == (1:T) || throw(
            ArgumentError(
                "EnKF + EKPCalibration replays the complete series and therefore requires " *
                    "the update range 1:$T; got $(first(update_range)):$(last(update_range))",
            )
        )
        forecast_number > 0 || throw(
            ArgumentError("forecast_number must be positive, got $forecast_number")
        )
        observations = _observation_vectors(asof.counts)

        if (forecast_number - 1) % reopt_interval == 0
            loss_range = _loss_range(T, validated_window_length)
            current_hyperparameters = merge(base_hyperparams, theta_mean[])
            checkpoint_filter, checkpoint_mode, checkpoint_steps =
                _prepare_window_checkpoint!(
                window_checkpoint,
                window_checkpoint_start,
                window_checkpoint_date,
                () -> build_backend(current_hyperparameters, inner_seed),
                current_hyperparameters,
                us,
                observations,
                asof.date,
                loss_range,
            )

            # One outer candidate: either a fresh filter for a full-history score or one deep copy
            # of the rolling checkpoint. The copy happens once per candidate, never once per
            # observation, and includes the EnKF RNG position so common random numbers survive
            # windowing. `_evaluate_ekp_candidates` converts failures to `NaN`, EKP's resampling
            # signal, and keeps diagnostics out of worker tasks.
            function score(theta_vector)
                candidate_hyperparameters = merge(
                    base_hyperparams,
                    NamedTuple{learned_names}(Tuple(theta_vector)),
                )
                kf = if checkpoint_filter === nothing
                    build_backend(candidate_hyperparameters, inner_seed)
                else
                    _candidate_from_checkpoint(
                        checkpoint_filter, nothing, Float64, candidate_hyperparameters
                    )
                end
                return checkpoint_filter === nothing ?
                    marginal_loglik(kf, us, observations, candidate_hyperparameters) :
                    _filter_loglik!(
                        kf, us, observations, candidate_hyperparameters, loss_range
                    )
            end

            # Resuming from the previous FINAL ENSEMBLE rather than from its mean is the point:
            # the spread travels too, so `iterations` can be a small fraction of
            # `burnin_iterations`. Without it every recalibration is as cold as the burn-in and
            # has to get the burn-in budget — otherwise `warm_start = false` would be comparing a
            # converged calibration against a 5-iteration one and reading the gap as drift in θ.
            resume_from = warm_start ? last_unconstrained[] : nothing
            n_iterations = (forecast_number == 1 || resume_from === nothing) ?
                hyper_method.burnin_iterations : hyper_method.iterations
            u0 = resume_from === nothing ?
                construct_initial_ensemble(
                    rng_outer, priors.prior, hyper_method.n_ensemble
                ) : resume_from
            ekp = _new_ekp(u0, priors.prior, hyper_method.inflation, rng_outer)
            @info "EKP: calibration starting" forecast_number observations = T loss_observations =
                length(loss_range) window_start = first(loss_range) checkpoint_mode checkpoint_steps parameters =
                collect(learned_names) outer_ensemble = hyper_method.n_ensemble iterations = n_iterations warm_started =
                resume_from !== nothing outer_threads julia_threads = Threads.nthreads()

            realised_iterations = 0
            for iteration in 1:n_iterations
                phi = get_ϕ_final(priors.prior, ekp)      # [n_params, n_ensemble], constrained
                evaluation = @timed _evaluate_ekp_candidates(
                    score, phi; threads = outer_threads
                )
                G, positive_logliks, failures = evaluation.value
                _report_ekp_candidate_diagnostics!(
                    warned_positive_ll,
                    positive_logliks,
                    failures;
                    forecast_number,
                    iteration,
                )
                n_failed = count(isnan, view(G, 1, :))
                @info "EKP: candidate evaluation complete" forecast_number iteration candidates =
                    size(G, 2) failed = n_failed outer_threads julia_threads = Threads.nthreads() wall_time_seconds =
                    evaluation.time
                # EKP's failure handler resamples failed members from the successful ones, so an
                # all-failed ensemble leaves it sampling from an empty set. Fail here instead,
                # where the parameters that produced it can still be reported.
                if n_failed == size(G, 2)
                    first_failure = findfirst(failure -> failure !== nothing, failures)
                    first_reason = first_failure === nothing ? "" :
                        " First failure (member $first_failure): " *
                        sprint(showerror, first(failures[first_failure]))
                    error(
                        "EKP: every outer candidate failed at forecast origin " *
                            "$forecast_number, iteration $iteration. Constrained ensemble: " *
                            "$(NamedTuple{learned_names}(Tuple(view(phi, :, 1))))." *
                            first_reason,
                    )
                end
                terminate = update_ensemble!(ekp, G)
                realised_iterations += 1
                if terminate === true
                    @warn(
                        "EKP: scheduler halted before the configured iteration count",
                        forecast_number, realised_iterations, configured = n_iterations,
                    )
                    break
                end
                n_failed > 0 && @info "EKP: candidates failed" forecast_number iteration failed =
                    n_failed of = size(G, 2)
            end

            phi_final = Matrix{Float64}(get_ϕ_final(priors.prior, ekp))
            # The mean of the CONSTRAINED ensemble, `mean(ϕ)` — not `ϕ(mean(u))`. The two differ
            # for bounded parameters, and "constrained ensemble mean" is what is wanted.
            theta_mean[] = NamedTuple{learned_names}(
                Tuple(vec(mean(phi_final, dims = 2)))
            )
            last_unconstrained[] = Matrix{Float64}(get_u_final(ekp))
            last_constrained[] = phi_final
            last_iterations[] = realised_iterations
            # Accumulated algorithmic time: how much of a Bayesian update the adaptive scheduler
            # actually performed. ~1 is a well-conditioned calibration; far below says the
            # ensemble barely moved, far above says it is chasing the data misfit.
            last_algorithmic_time[] = sum(get_Δt(ekp))
            enkf.p = merge(base_hyperparams, theta_mean[])
            @info "EKP: calibration complete" forecast_number realised_iterations algorithmic_time =
                last_algorithmic_time[] estimates = theta_mean[]
        end

        hyperparameters = merge(base_hyperparams, theta_mean[])
        @info "EnKF: filtering observation history" forecast_number observations = T members =
            filter_method.n_ensemble
        # Rebuilt at the current parameters so the initial state matches them; as on the UKF
        # path, `p` is the only thing that persists on the caller's filter across origins.
        kf_live = build_backend(hyperparameters, inner_seed)
        final_ensemble = Ref{Any}(nothing)
        # The CORRECTED ensemble at the forecast origin, captured through LLPF's own callback,
        # which fires after `correct!` and before the state is saved.
        #
        # Copy on the last correction only, counted here rather than inferred from `f.t` (which
        # `predict!`, not `correct!`, increments) — the callback fires exactly once per correction,
        # so a counter is independent of LLPF's internal bookkeeping.
        #
        # It has to be a copy taken at that moment, NOT a reference deep-copied after
        # `forward_trajectory` returns. `predict!` and `correct!` both mutate `f.ensemble` in place
        # (replacing elements of the same outer vector), and the loop's last act is `predict!` — so
        # a reference read afterwards is the PREDICTED ensemble, one step past the origin. Forecast
        # from that and every horizon carries an extra step of process noise and lands a week late.
        corrections = Ref(0)
        solution = forward_trajectory(
            kf_live, us, observations, hyperparameters;
            post_correct_cb = function (f, _p, _ret)
                corrections[] += 1
                emit_forecast && corrections[] == T &&
                    (final_ensemble[] = deepcopy(f.ensemble))
                return nothing
            end,
        )
        corrections[] == T || error(
            "EnKF: expected $T corrections but the filter callback fired $(corrections[]) " *
                "times, so the captured ensemble is not the one at the forecast origin"
        )
        isfinite(solution.ll) || error(
            "EnKF: filtering pass produced a non-finite log-likelihood at forecast origin " *
                "$forecast_number under $(theta_mean[])"
        )

        # Same construction as the UKF path, so the two are directly comparable: the filtered
        # accumulator pushed through each spec's `observation_mean` at its own model time. `xt`
        # is the corrected (post-`correct!`) mean.
        mean_updates = _fitted_observation_means(
            observation_specs, layout.accumulator_indices, stochastic_dynamics.extract,
            solution.xt, hyperparameters, filter_method.dt, update_range,
        )

        hyperparameter_summary = DataFrame(
            parameter = String[], statistic = String[], value = Float64[]
        )
        # The spread of the outer ensemble is a CALIBRATION DIAGNOSTIC — how converged the
        # inversion is — not a posterior credible interval. Ensemble Kalman inversion contracts
        # toward the optimum rather than sampling the posterior, so reading these as intervals
        # would understate uncertainty. Named `ekp_*` to keep that distinction visible in the CSV.
        push!(
            hyperparameter_summary,
            ("ekp", "iterations", Float64(last_iterations[])),
        )
        push!(
            hyperparameter_summary,
            ("ekp", "algorithmic_time", last_algorithmic_time[]),
        )
        spread = last_constrained[]
        for (index, name) in enumerate(learned_names)
            push!(
                hyperparameter_summary,
                (string(name), "estimate", Float64(theta_mean[][name])),
            )
            spread === nothing && continue
            values = collect(view(spread, index, :))
            for (statistic, probability) in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))
                push!(
                    hyperparameter_summary,
                    (string(name), "ekp_" * statistic, quantile(values, probability)),
                )
            end
        end
        if !emit_forecast
            @info "EnKF: forecast suppressed after sequential fit" forecast_number log_likelihood =
                solution.ll
            return nothing, mean_updates, hyperparameter_summary
        end

        @info "EnKF: generating forecast ensemble" forecast_number draws = n_draws horizons =
            n_ahead
        # Uniform resample of the corrected ensemble — an EnKF's members are equally weighted,
        # unlike the PF's cloud.
        corrected = final_ensemble[]
        indices = rand(rng_forecast, eachindex(corrected), Int(n_draws))
        initial_states = [corrected[index] for index in indices]
        forecast_filter = deepcopy(kf_live)
        Random.seed!(forecast_filter.rng, rand(rng_forecast, UInt64))
        samples, latent_samples = forecast_ensemble(
            forecast_filter,
            initial_states,
            hyperparameters;
            n_ahead = Int(n_ahead),
            t0 = float(T - 1) * filter_method.dt,
            dt = filter_method.dt,
            latent_range = layout.latent_range,
            n_obs = n_observations,
        )

        # Shape follows the model, not the filter: a `Vector` over time for one signal (identical to
        # the UKF and PF paths, so a single-signal CSV is unchanged), a `[time, observation]` matrix
        # for several. `run_backtest` checks `size(..., 1)`, which is the same for both.
        # Forecast-spread diagnostic: the realised across-member spread of each latent coefficient
        # per horizon (unconstrained chart), as on the PF path.
        latent_log_sd = [
            _cloud_sd(view(latent_samples, horizon, :, l))
                for horizon in 1:Int(n_ahead), l in axes(latent_samples, 3)
        ]
        append_latent_spread!(hyperparameter_summary, layout.latent_names, latent_log_sd)
        # In-sample counterpart: the filtered ensemble-mean latent path over the assimilated
        # window, which the forecast-spread diagnostic above cannot show.
        append_latent_audit!(
            hyperparameter_summary, layout.latent_names, solution.xt, layout.latent_range;
            dt = filter_method.dt,
        )
        quantile_array = forecast_quantiles(samples)
        sample_callback === nothing || sample_callback(samples)
        @info "EnKF: forecast complete" forecast_number log_likelihood = solution.ll
        return quantile_array, mean_updates, hyperparameter_summary
    end
    return enkf, fit_forecast!
end
