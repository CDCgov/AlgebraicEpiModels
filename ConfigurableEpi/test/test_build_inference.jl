using Test
using ConfigurableEpi
using DataFramesMeta: DataFrame, names
using Dates: Date, Day
using LinearAlgebra: BLAS, diag
using Distributions: NegativeBinomial, Normal, quantile
using LowLevelParticleFilters: AdvancedParticleFilter, UnscentedKalmanFilter, forward_trajectory,
    correct!, covariance, expweights, index, particles, predict!, reset!, state, weights
import Random

function mock_inference_vf!(du, u, p, t)
    hyper, latent = p
    beta = 0.3 * hyper.R0_baseline * latent.Rt / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

@testset "Typed inference assembly" begin
    @test UKF() == UKF(1.0, 2, 1.0)
    @test PF(100) == PF(100, 1.0, 2, 0.0, true)
    @test PF(100).threads                      # particle parallelism is the default
    @test !PF(100; threads = false).threads
    @test PFFilterConfig(n_particles = 100).threads
    @test_throws ArgumentError UKF(dt = 0.0)
    @test_throws ArgumentError UKF(supersample = 0)
    @test_throws ArgumentError UKF(obs_jitter = 0.0)
    @test_throws ArgumentError PF(0)
    @test_throws ArgumentError PF(10; dt = 0.0)
    @test_throws ArgumentError PF(10; supersample = 0)
    @test_throws ArgumentError PF(10; obs_jitter = -1.0)

    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.03))
    stochastic = build_stochastic_update(layout, (rt_spec,))
    obs_model = (
        SignalObservationSpec(
            1,
            NegBinomialNoise(phi = (_latent, hyper, _t) -> hyper.phi);
            mean_modifier = (_latent, hyper, _t) -> hyper.ascertainment,
            name = :reports,
        ),
    )
    base_hyperparams = (
        R0_baseline = 1.5,
        obs_scale = 0.25,
        ascertainment = 0.8,
        phi = 50.0,
    )
    x0_state = [
        900.0,
        50.0,
        10.0,
        stochastic.to_unconstrained((Rt = 1.0,))[1],
    ]
    r0_prior = positive_gaussian(:R0_baseline, 1.5, 0.4)
    asof2 = DataFrame(
        date = Date.(["2024-01-01", "2024-01-08"]),
        counts = [8.0, 10.0],
    )
    asof3 = DataFrame(
        date = Date.(["2024-01-01", "2024-01-08", "2024-01-15"]),
        counts = [8.0, 10.0, 12.0],
    )
    asof4 = DataFrame(
        date = Date("2024-01-01") .+ Day.(7 .* (0:3)),
        counts = [8.0, 10.0, 12.0, 11.0],
    )
    asof6 = DataFrame(
        date = Date("2024-01-01") .+ Day.(7 .* (0:5)),
        counts = [8.0, 10.0, 12.0, 11.0, 13.0, 12.0],
    )

    optimize_method = OptimiseHyperparams(
        (R0_baseline = r0_prior,); options = (maxiters = 1,)
    )
    @test OptimiseConfig(;
        reopt_interval = 2, opt_maxiters = 1, opt_maxiters_burnin = 2,
    ).window_length === nothing
    @test OptimiseConfig(;
        reopt_interval = 2, opt_maxiters = 1, opt_maxiters_burnin = 2,
        window_length = 3,
    ).window_length == 3
    liu_west_method = LiuWest((R0_baseline = r0_prior,); discount = 0.97)
    @test liu_west_method isa HyperparamInferenceMethod
    unsupported = try
        build_inference(PF(10), optimize_method)
    catch error
        error
    end
    @test unsupported isa ArgumentError
    @test occursin("PF", sprint(showerror, unsupported))
    @test occursin("OptimiseHyperparams", sprint(showerror, unsupported))
    @test_throws ArgumentError build_inference(UKF(), liu_west_method)

    @testset "initial latent variance: package default, per-name override, typo caught" begin
        cov = ConfigurableEpi._initial_state_covariance
        li = first(layout.latent_range)

        # Unnamed slots keep the shared default.
        @test cov(layout, x0_state)[li, li] == DEFAULT_LATENT_VARIANCE

        # A model can make the latent's initial spread follow the latent's own prior instead.
        from_prior = prior_unconstrained_variance(positive_gaussian(:Rt, 1.0, 0.1))
        @test from_prior != DEFAULT_LATENT_VARIANCE
        @test cov(layout, x0_state; latent_variance = (Rt = from_prior,))[li, li] ≈ from_prior

        # The learned tail still appends after the latent block, unaffected.
        with_tail = cov(layout, x0_state, [0.25]; latent_variance = (Rt = from_prior,))
        @test size(with_tail, 1) == layout.total_dim + 1
        @test with_tail[end, end] == 0.25

        # A name that is not a latent slot is a typo, not a silent no-op.
        @test_throws ArgumentError cov(layout, x0_state; latent_variance = (Rt_typo = 0.1,))

        # Compartment uncertainty scales with each compartment's OWN initial value, so one
        # matrix works across compartments spanning orders of magnitude. A flat absolute
        # variance cannot: the historical 1e-6 was sd 1e-3 on a compartment of 1e7.
        P0 = cov(layout, x0_state)
        for (i, x) in enumerate(x0_state[1:length(ode_names(layout))])
            @test P0[i, i] ≈ max((DEFAULT_MODEL_RELATIVE_SD * x)^2, 1.0e-6)
        end
        # S = 900 gets a far wider slot than the accumulator seeded at 10.
        @test P0[1, 1] > P0[3, 3]
        # An empty compartment still leaves P0 positive definite.
        empty_x0 = [0.0, 0.0, 0.0, x0_state[4]]
        @test all(diag(cov(layout, empty_x0)) .> 0.0)
        # And the knob is a knob.
        @test cov(layout, x0_state; model_relative_sd = 0.1)[1, 1] ≈ 4 * P0[1, 1]
        @test_throws ArgumentError cov(layout, x0_state; model_relative_sd = -0.1)
    end

    @testset "_observation_vectors: single-signal shape is unchanged" begin
        obsvec = ConfigurableEpi._observation_vectors
        # A numeric column is the historical single-signal case, reproduced exactly.
        @test obsvec([8.0, 10.0, 12.0]) == [[8.0], [10.0], [12.0]]
        @test obsvec([8, 10]) == [[8.0], [10.0]]
        @test eltype(obsvec([8.0])) == Vector{Float64}
        # A column of vectors is the widened multi-location case: one entry per date, each holding
        # the per-location counts in signal order.
        @test obsvec([[8.0, 90.0], [10.0, 91.0]]) == [[8.0, 90.0], [10.0, 91.0]]
        @test eltype(obsvec([[8, 90], [10, 91]])) == Vector{Float64}
    end

    @testset "initial accumulator variance: opt-in, keyed by signal" begin
        cov = ConfigurableEpi._initial_state_covariance
        acc = only(layout.accumulator_indices)
        signal = only(layout.signal_names)   # the mock layout's default name, not `:reports`

        # A reset accumulator seeded at 0 is floored, i.e. effectively pinned. That is fine for a
        # Gaussian filter (exact zero cross-covariance ⇒ zero gain into the compartments) and fatal
        # for an ensemble one, whose sample cross-covariance is noise that a near-zero innovation
        # variance amplifies without bound.
        pinned_x0 = [900.0, 50.0, 0.0, x0_state[4]]
        @test cov(layout, pinned_x0)[acc, acc] == 1.0e-6

        # The override is keyed by SIGNAL name, and only the accumulator slot moves.
        widened = cov(layout, pinned_x0; accumulator_variance = (; signal => 4.0e6))
        @test widened[acc, acc] == 4.0e6
        @test widened[1, 1] == cov(layout, pinned_x0)[1, 1]

        # Opting out leaves P0 exactly as it was — this is what keeps the UKF and PF paths
        # byte-identical while the ensemble path widens.
        @test cov(layout, pinned_x0; accumulator_variance = (;)) == cov(layout, pinned_x0)

        # Still floored, so a zero override cannot make P0 singular.
        @test cov(layout, pinned_x0; accumulator_variance = (; signal => 0.0))[acc, acc] ==
            1.0e-6
        # A signal name that does not exist is a typo, not a silent no-op.
        @test_throws ArgumentError cov(
            layout, pinned_x0; accumulator_variance = (definitely_not_a_signal = 1.0,)
        )
        @test_throws ArgumentError cov(
            layout, pinned_x0; accumulator_variance = (; signal => -1.0)
        )
    end

    @testset "rolling filter checkpoints" begin
        dynamics, measure, n_obs, n_noise =
            ConfigurableEpi.build_dynamics_and_measurement(
            mock_inference_vf!, layout, obs_model, stochastic;
            dt = 1.0, supersample = 1, obs_jitter = 1.0,
        )
        build_ukf(::Type{E}, hp) where {E} = ConfigurableEpi._ukf_backend(
            E,
            dynamics,
            measure,
            layout,
            x0_state,
            1.0,
            n_obs,
            n_noise,
            hp,
            (;),
        )
        observations = ConfigurableEpi._observation_vectors(asof6.counts)
        us6 = fill(Float64[], 6)

        # The public four-argument method keeps the original reset-and-full-replay arithmetic
        # exactly: the ranged helper is only a refactoring of that recursion.
        reference = build_ukf(Float64, base_hyperparams)
        reset!(reference)
        reference_ll = zero(eltype(state(reference)))
        for k in eachindex(observations)
            time = (k - 1) * reference.Ts
            reference_ll += first(
                correct!(reference, us6[k], observations[k], base_hyperparams, time)
            )
            predict!(reference, us6[k], base_hyperparams, time)
        end

        # A filter stopped after the prefix is a complete in-memory checkpoint: resuming it at
        # the same parameters gives the same final state as one continuous pass. Only this one
        # boundary object is copied; `_filter_loglik!` itself performs no copies.
        continuous = build_ukf(Float64, base_hyperparams)
        full_ll = marginal_loglik(continuous, us6, observations, base_hyperparams)
        @test full_ll === reference_ll
        anchor = build_ukf(Float64, base_hyperparams)
        reset!(anchor)
        prefix_ll = ConfigurableEpi._filter_loglik!(
            anchor, us6, observations, base_hyperparams, 1:2
        )
        checkpoint = deepcopy(anchor)
        direct_copy = deepcopy(checkpoint)
        direct_tail_ll = ConfigurableEpi._filter_loglik!(
            direct_copy, us6, observations, base_hyperparams, 3:6
        )
        resumed = ConfigurableEpi._candidate_from_checkpoint(
            checkpoint, build_ukf, Float64, base_hyperparams
        )
        tail_ll = ConfigurableEpi._filter_loglik!(
            resumed, us6, observations, base_hyperparams, 3:6
        )
        @test prefix_ll + tail_ll ≈ full_ll
        @test direct_tail_ll == tail_ll
        @test state(direct_copy) == state(resumed)
        @test covariance(direct_copy) == covariance(resumed)
        @test index(direct_copy) == index(resumed)
        @test state(resumed) == state(continuous)
        @test covariance(resumed) == covariance(continuous)
        @test index(resumed) == index(continuous) == 6
        @test state(checkpoint) == state(anchor) # candidate mutation did not touch the checkpoint

        # The Float64 checkpoint is constant with respect to the new θ. A fresh Dual-typed UKF
        # receives its mean/covariance/index, and only the window contributes derivatives.
        objective = function (values)
            hp = merge(base_hyperparams, (R0_baseline = only(values),))
            candidate = ConfigurableEpi._candidate_from_checkpoint(
                checkpoint, build_ukf, eltype(values), hp
            )
            return ConfigurableEpi._filter_loglik!(
                candidate, us6, observations, hp, 3:6
            )
        end
        gradient = ConfigurableEpi.ForwardDiff.gradient(
            objective, [base_hyperparams.R0_baseline]
        )
        @test all(isfinite, gradient)

        # The checkpoint follows the observation boundary, not the forecast cadence. Here a
        # fixed two-observation window moves from 3 to 5 because T jumps from 4 to 6; the retained
        # filter advances exactly two observations without rebuilding the prefix.
        checkpoint_ref = Ref{Union{Nothing, typeof(anchor)}}(nothing)
        checkpoint_start = Ref(1)
        checkpoint_date = Ref{Union{Nothing, Date}}(nothing)
        builds = Ref(0)
        build_anchor = function ()
            builds[] += 1
            return build_ukf(Float64, base_hyperparams)
        end
        first_checkpoint, mode, steps = ConfigurableEpi._prepare_window_checkpoint!(
            checkpoint_ref,
            checkpoint_start,
            checkpoint_date,
            build_anchor,
            base_hyperparams,
            us6,
            observations,
            asof6.date,
            3:4,
        )
        @test mode == :anchored
        @test steps == 2
        @test checkpoint_start[] == 3
        @test index(first_checkpoint) == 2
        second_checkpoint, mode, steps = ConfigurableEpi._prepare_window_checkpoint!(
            checkpoint_ref,
            checkpoint_start,
            checkpoint_date,
            build_anchor,
            base_hyperparams,
            us6,
            observations,
            asof6.date,
            5:6,
        )
        @test mode == :advanced
        @test steps == 2
        @test builds[] == 1
        @test checkpoint_start[] == 5
        @test index(second_checkpoint) == 4

        # Value revisions already behind the boundary are frozen into the historical checkpoint.
        state_before_revision = copy(state(second_checkpoint))
        revised_observations = deepcopy(observations)
        revised_observations[1][1] += 100.0
        _, mode, steps = ConfigurableEpi._prepare_window_checkpoint!(
            checkpoint_ref,
            checkpoint_start,
            checkpoint_date,
            build_anchor,
            base_hyperparams,
            us6,
            revised_observations,
            asof6.date,
            5:6,
        )
        @test mode == :advanced
        @test steps == 0
        @test state(checkpoint_ref[]) == state_before_revision

        # A structurally changed boundary date or a backwards boundary cannot be reached by forward
        # mutation. Re-running the prefix at the current parameters would silently substitute the
        # constant-θ prefix `window_length = nothing` already computes — and would re-apply
        # `build_x0`'s equilibrium seed away from the real start of the data — so both refuse and
        # say which knob restores full replay.
        state_before_refusal = copy(state(checkpoint_ref[]))
        changed_dates = copy(asof6.date)
        changed_dates[5] += Day(1)
        @test_throws "window_length = nothing" ConfigurableEpi._prepare_window_checkpoint!(
            checkpoint_ref,
            checkpoint_start,
            checkpoint_date,
            build_anchor,
            base_hyperparams,
            us6,
            observations,
            changed_dates,
            5:6,
        )
        @test_throws "cannot move backwards" ConfigurableEpi._prepare_window_checkpoint!(
            checkpoint_ref,
            checkpoint_start,
            checkpoint_date,
            build_anchor,
            base_hyperparams,
            us6,
            observations,
            asof6.date,
            4:6,
        )
        # A refusal is inert: no rebuild, and the retained checkpoint is untouched.
        @test builds[] == 1
        @test checkpoint_start[] == 5
        @test state(checkpoint_ref[]) == state_before_refusal

        # A window that reaches back to the first observation drops the checkpoint entirely and
        # scores the complete history — the honest way to ask for what a rebuild used to fake.
        _, mode, steps = ConfigurableEpi._prepare_window_checkpoint!(
            checkpoint_ref,
            checkpoint_start,
            checkpoint_date,
            build_anchor,
            base_hyperparams,
            us6,
            observations,
            asof6.date,
            1:6,
        )
        @test mode == :full_history
        @test steps == 0
        @test checkpoint_ref[] === nothing
        @test builds[] == 1

        @test ConfigurableEpi._loss_range(6, nothing) == 1:6
        @test ConfigurableEpi._loss_range(6, 2) == 5:6
        @test ConfigurableEpi._loss_range(6, 6) == 1:6
        @test ConfigurableEpi._loss_range(3, 10) == 1:3
    end

    @testset "UKF + optimization replays the complete window" begin
        ukf, fit_forecast! = build_inference(
            UKF(dt = 1.0, supersample = 1),
            optimize_method,
            mock_inference_vf!,
            layout,
            stochastic,
            obs_model,
            base_hyperparams,
            x0_state;
            n_ahead = 2,
            reopt_interval = 2,
            initial_optim_options = (maxiters = 2,),
        )
        @test ukf isa UnscentedKalmanFilter
        us = fill(Float64[], 3)
        qmat, mean_updates, hyperparameters = fit_forecast!(
            ukf, asof3, 1:3, us, 1
        )
        @test size(qmat) == (2, length(DEFAULT_QS))
        @test all(issorted(qmat[h, :]) for h in axes(qmat, 1))
        @test length(mean_updates) == 3
        @test names(hyperparameters) == ["parameter", "statistic", "value"]
        @test_throws ArgumentError fit_forecast!(
            ukf, asof3, 1:3, us, 1; sample_callback = _ -> nothing,
        )
        # The frame carries a learned-parameter estimate per name PLUS the per-horizon
        # forecast-spread rows for each latent coefficient, so select by statistic.
        estimate(df, name) =
            only(df[(df.parameter .== name) .& (df.statistic .== "estimate"), :value])
        @test ukf.p.R0_baseline == estimate(hyperparameters, "R0_baseline")
        spread = hyperparameters[hyperparameters.statistic .== "fc_log_sd_h2", :]
        @test spread.parameter == ["Rt"]           # one row per latent coefficient
        @test only(spread.value) > 0.0
        @test_throws ArgumentError fit_forecast!(ukf, asof3, 2:3, us, 2)

        # A non-reoptimization origin retains the explicit parameter state on the UKF.
        previous_parameter = ukf.p.R0_baseline
        suppressed, suppressed_updates, suppressed_hyper = fit_forecast!(
            ukf, asof3, 1:3, us, 2; emit_forecast = false
        )
        @test suppressed === nothing
        @test length(suppressed_updates) == 3
        @test all(suppressed_hyper.statistic .== "estimate")
        @test ukf.p.R0_baseline == previous_parameter

        # The next cadence point optimizes again using the ordinary options.
        _, _, reoptimized = fit_forecast!(ukf, asof3, 1:3, us, 3)
        @test ukf.p.R0_baseline == estimate(reoptimized, "R0_baseline")

        @testset "window length is independent of re-optimization cadence" begin
            @test_throws ArgumentError build_inference(
                UKF(dt = 1.0, supersample = 1),
                optimize_method,
                mock_inference_vf!,
                layout,
                stochastic,
                obs_model,
                base_hyperparams,
                x0_state;
                n_ahead = 2,
                window_length = 0,
            )
            windowed_ukf, windowed_fit! = build_inference(
                UKF(dt = 1.0, supersample = 1),
                optimize_method,
                mock_inference_vf!,
                layout,
                stochastic,
                obs_model,
                base_hyperparams,
                x0_state;
                n_ahead = 2,
                reopt_interval = 2,
                window_length = 2,
                initial_optim_options = (maxiters = 1,),
            )
            # Re-optimize at origins 1 and 3, while T moves 3 → 4 → 6. The checkpoint boundary
            # therefore moves 2 → 5, independently of the two-origin cadence.
            q3, means3, _ = windowed_fit!(
                windowed_ukf, asof3, 1:3, fill(Float64[], 3), 1
            )
            windowed_fit!(windowed_ukf, asof4, 1:4, fill(Float64[], 4), 2)
            q6, means6, summary6 = windowed_fit!(
                windowed_ukf, asof6, 1:6, fill(Float64[], 6), 3
            )
            @test size(q3) == size(q6) == (2, length(DEFAULT_QS))
            @test length(means3) == 3
            @test length(means6) == 6
            @test all(isfinite, q6)
            @test windowed_ukf.p.R0_baseline == estimate(summary6, "R0_baseline")
        end

        @testset "warm_start controls where each re-optimization begins" begin
            function make_ukf(; warm_start = true)
                return build_inference(
                    UKF(dt = 1.0, supersample = 1),
                    optimize_method,
                    mock_inference_vf!,
                    layout,
                    stochastic,
                    obs_model,
                    base_hyperparams,
                    x0_state;
                    n_ahead = 2,
                    reopt_interval = 2,
                    warm_start = warm_start,
                    initial_optim_options = (maxiters = 2,),
                )
            end
            inputs = fill(Float64[], 3)

            # Two origins that re-optimize (1 and 3) on identical data. Warm-started, the second
            # begins at the first's answer; cold, it begins at `base_hyperparams` again — so with
            # the SAME data it must return to the SAME place the first origin reached.
            cold_ukf, cold_fit! = make_ukf(warm_start = false)
            _, _, cold_first = cold_fit!(cold_ukf, asof3, 1:3, inputs, 1)
            first_estimate = estimate(cold_first, "R0_baseline")
            _, _, cold_second = cold_fit!(cold_ukf, asof3, 1:3, inputs, 3)
            @test estimate(cold_second, "R0_baseline") ≈ first_estimate

            # The warm run is the one whose second origin can differ, because it starts from a
            # different point. Its estimate must still be the one written onto the live filter.
            warm_ukf, warm_fit! = make_ukf()
            warm_fit!(warm_ukf, asof3, 1:3, inputs, 1)
            _, _, warm_second = warm_fit!(warm_ukf, asof3, 1:3, inputs, 3)
            @test warm_ukf.p.R0_baseline == estimate(warm_second, "R0_baseline")
        end
    end

    @testset "EnKF + EKP calibration" begin
        ekp_method(;
            n_ensemble = 6, iterations = 2, burnin_iterations = 3, threads = false,
        ) =
            EKPCalibration(
            (R0_baseline = r0_prior,);
            n_ensemble = n_ensemble, iterations = iterations,
            burnin_iterations = burnin_iterations,
            threads = threads,
        )

        function make_engine(
                seed; reopt_interval = 2, window_length = nothing, warm_start = true,
                inner_threads = false, outer_threads = false, kwargs...,
            )
            return build_inference(
                EnKF(40; dt = 1.0, supersample = 1, threads = inner_threads),
                ekp_method(; threads = outer_threads, kwargs...),
                mock_inference_vf!,
                layout,
                stochastic,
                obs_model,
                base_hyperparams,
                x0_state;
                n_ahead = 2,
                reopt_interval = reopt_interval,
                window_length = window_length,
                warm_start = warm_start,
                n_draws = 50,
                rng = Random.MersenneTwister(seed),
            )
        end

        @test ekp_method() isa HyperparamInferenceMethod
        default_ekp_config = EKPConfig(;
            n_ensemble = 6, reopt_interval = 2, iterations = 2,
            burnin_iterations = 3,
        )
        @test default_ekp_config.window_length === nothing
        @test default_ekp_config.threads === false
        @test EKPConfig(;
            n_ensemble = 6, reopt_interval = 2, iterations = 2,
            burnin_iterations = 3, window_length = 4,
        ).window_length == 4
        @test_throws ArgumentError EKPCalibration((R0_baseline = r0_prior,); n_ensemble = 1, iterations = 2)
        @test_throws ArgumentError EKPCalibration((R0_baseline = r0_prior,); n_ensemble = 6, iterations = 0)
        @test EKPCalibration(
            (R0_baseline = r0_prior,); n_ensemble = 6, iterations = 2,
            threads = true,
        ).threads
        @test_throws "nested ensemble threading" make_engine(
            5; inner_threads = true, outer_threads = true,
        )
        # Unsupported pairings on the new axis members stay ArgumentErrors, not MethodErrors.
        @test_throws ArgumentError build_inference(EnKF(10), optimize_method)
        @test_throws ArgumentError build_inference(UKF(), ekp_method())

        us = fill(Float64[], 3)
        selector(df, name, statistic) =
            only(df[(df.parameter .== name) .& (df.statistic .== statistic), :value])

        @testset "outer candidate scheduling and diagnostics" begin
            @test_logs (:warn, r"only one thread") !ConfigurableEpi._outer_ekp_threads_enabled(
                true; nthreads = 1,
            )
            @test ConfigurableEpi._outer_ekp_threads_enabled(true; nthreads = 2)

            phi = reshape([-2.0, 0.25, 3.0], 1, :)
            score(theta) = theta[1] == 3.0 ? error("deliberate candidate failure") : theta[1]
            serial = ConfigurableEpi._evaluate_ekp_candidates(score, phi; threads = false)

            original_blas_threads = BLAS.get_num_threads()
            BLAS.set_num_threads(2)
            try
                threaded = ConfigurableEpi._evaluate_ekp_candidates(
                    score, phi; threads = true,
                )
                @test isequal(threaded[1], serial[1])
                @test threaded[2] == serial[2]
                @test findall(value -> value !== nothing, threaded[3]) == [3]
                @test BLAS.get_num_threads() == 2
            finally
                BLAS.set_num_threads(original_blas_threads)
            end

            G, positive_logliks, failures = serial
            @test G[1, 1] == 2.0
            @test G[1, 2] == sqrt(eps())
            @test isnan(G[1, 3])
            warned = Ref(false)
            @test_logs (
                :warn, r"positive filter log-likelihood",
            ) (
                :warn, r"outer candidate failed",
            ) ConfigurableEpi._report_ekp_candidate_diagnostics!(
                warned,
                positive_logliks,
                failures;
                forecast_number = 1,
                iteration = 1,
            )
            @test warned[]
        end

        enkf, fit_forecast! = make_engine(11)
        @test enkf isa AugmentedEnsembleKalmanFilter
        # The process-noise dimension is smaller than the state dimension: one latent slot plus
        # one accumulator whisker against four state slots. This is the case stock LLPF's additive
        # ensemble filter cannot express.
        @test enkf.nw == 2
        @test enkf.nx == 4
        @test enkf.nw != enkf.nx

        @testset "deepcopy preserves complete ensemble state and RNG" begin
            observations = ConfigurableEpi._observation_vectors(asof3.counts)
            source, _ = make_engine(23)
            continuous, _ = make_engine(23)
            full_ll = marginal_loglik(
                continuous, us, observations, base_hyperparams
            )
            reset!(source)
            prefix_ll = ConfigurableEpi._filter_loglik!(
                source, us, observations, base_hyperparams, 1:1
            )
            checkpoint = deepcopy(source)
            resumed = ConfigurableEpi._candidate_from_checkpoint(
                checkpoint, nothing, Float64, base_hyperparams
            )
            tail_ll = ConfigurableEpi._filter_loglik!(
                resumed, us, observations, base_hyperparams, 2:3
            )
            @test prefix_ll + tail_ll ≈ full_ll
            @test resumed.ensemble == continuous.ensemble
            @test state(resumed) == state(continuous)
            @test covariance(resumed) == covariance(continuous)
            @test index(resumed) == index(continuous)
            @test Random.rand(copy(resumed.rng), UInt64) ==
                Random.rand(copy(continuous.rng), UInt64)
            @test checkpoint.ensemble == source.ensemble

            # The copy carries the CANDIDATE's parameters, not the checkpoint's. Losses pass `p`
            # explicitly so this cannot move a score, but a filter whose `p` disagrees with the
            # parameters it is being scored at is a trap for anything falling back on
            # `parameters(f)` — `enkf.p` is load-bearing on the live filter.
            shifted = merge(base_hyperparams, (R0_baseline = 3.7,))
            reparameterized = ConfigurableEpi._candidate_from_checkpoint(
                checkpoint, nothing, Float64, shifted
            )
            @test reparameterized.p.R0_baseline == 3.7
            @test checkpoint.p.R0_baseline == base_hyperparams.R0_baseline

            # Every candidate starts from the same checkpoint RNG position. Interleaving a
            # different candidate cannot change a repeated candidate's score.
            score = function (r0)
                hp = merge(base_hyperparams, (R0_baseline = r0,))
                candidate = ConfigurableEpi._candidate_from_checkpoint(
                    checkpoint, nothing, Float64, hp
                )
                return ConfigurableEpi._filter_loglik!(
                    candidate, us, observations, hp, 2:3
                )
            end
            score_a = score(1.3)
            score(1.7)
            @test score(1.3) == score_a
        end

        enkf_samples = Ref{Any}(nothing)
        qmat, mean_updates, hyperparameters = fit_forecast!(
            enkf, asof3, 1:3, us, 1;
            sample_callback = samples -> (enkf_samples[] = copy(samples)),
        )
        @test size(qmat) == (2, length(DEFAULT_QS))
        @test size(enkf_samples[]) == (2, 50)
        @test forecast_quantiles(enkf_samples[]) == qmat
        @test all(issorted(qmat[h, :]) for h in axes(qmat, 1))
        @test all(qmat .>= 0.0)
        @test all(isfinite, qmat)
        @test length(mean_updates) == 3
        @test names(hyperparameters) == ["parameter", "statistic", "value"]
        @test enkf.p.R0_baseline == selector(hyperparameters, "R0_baseline", "estimate")
        # The outer spread is reported as a calibration diagnostic, distinctly named so it cannot
        # be mistaken for a posterior interval.
        @test selector(hyperparameters, "R0_baseline", "ekp_q05") <=
            selector(hyperparameters, "R0_baseline", "ekp_q95")
        # The replay contract matches the UKF path: a partial range is rejected.
        @test_throws ArgumentError fit_forecast!(enkf, asof3, 2:3, us, 2)

        @testset "runs the configured iteration counts" begin
            # burnin_iterations at the first origin, iterations at later cadence points. Asserted
            # on the reported count because EKP's default scheduler can stop early — see `_new_ekp`.
            engine, fit! = make_engine(11; reopt_interval = 1, iterations = 2, burnin_iterations = 4)
            _, _, first_origin = fit!(engine, asof3, 1:3, us, 1)
            @test selector(first_origin, "ekp", "iterations") == 4.0
            _, _, second_origin = fit!(engine, asof3, 1:3, us, 2)
            @test selector(second_origin, "ekp", "iterations") == 2.0
        end

        @testset "recalibration follows reopt_interval and warm-starts" begin
            engine, fit! = make_engine(11; reopt_interval = 2)
            fit!(engine, asof3, 1:3, us, 1)
            after_first = engine.p.R0_baseline
            # Origin 2 is off-cadence: the retained constrained ensemble mean is used unchanged.
            suppressed, updates, held = fit!(
                engine, asof3, 1:3, us, 2; emit_forecast = false
            )
            @test suppressed === nothing
            @test length(updates) == 3
            @test engine.p.R0_baseline == after_first
            @test selector(held, "R0_baseline", "estimate") == after_first
            # Origin 3 recalibrates, warm-started from the previous final ensemble.
            _, _, recalibrated = fit!(engine, asof3, 1:3, us, 3)
            @test engine.p.R0_baseline == selector(recalibrated, "R0_baseline", "estimate")
        end

        @testset "serial and outer-threaded engines agree exactly" begin
            # The whole point of the common-random-number seed: G(θ) depends only on θ, so two
            # engines built with the same seed take the same calibration path even when candidate
            # evaluation order changes. Hold BLAS at one for BOTH complete paths so its own
            # reduction order is not a second variable in this Julia-threading comparison.
            # If the inner filters shared a live RNG, candidate order would still leak in.
            a_enkf, a_fit! = make_engine(7)
            b_enkf, b_fit! = make_engine(7; outer_threads = true)
            original_blas_threads = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
            serial_result = threaded_result = nothing
            try
                serial_result = a_fit!(a_enkf, asof3, 1:3, us, 1)
                threaded_result = b_fit!(b_enkf, asof3, 1:3, us, 1)
            finally
                BLAS.set_num_threads(original_blas_threads)
            end
            qa, ma, ha = serial_result
            qb, mb, hb = threaded_result
            @test qa == qb
            @test ma == mb
            # `isequal`, not `==`: the in-sample audit reports `is_tau_days = NaN` whenever
            # the realised lag-1 autocorrelation is <= 0 (here it is, on three
            # observations), and `NaN == NaN` is false. Identical output must compare
            # identical.
            @test isequal(ha.value, hb.value)
            # A different seed moves the outer ensemble, so the equality above is not vacuous.
            c_enkf, c_fit! = make_engine(8)
            @test c_fit!(c_enkf, asof3, 1:3, us, 1)[1] != qa
        end


        @testset "windowed EKP remains reproducible across a distinct cadence" begin
            @test_throws ArgumentError make_engine(5; window_length = -1)
            a_enkf, a_fit! = make_engine(
                31; reopt_interval = 2, window_length = 2,
                iterations = 1, burnin_iterations = 1,
            )
            b_enkf, b_fit! = make_engine(
                31; reopt_interval = 2, window_length = 2,
                iterations = 1, burnin_iterations = 1,
            )
            a_fit!(a_enkf, asof3, 1:3, fill(Float64[], 3), 1)
            b_fit!(b_enkf, asof3, 1:3, fill(Float64[], 3), 1)
            a_fit!(a_enkf, asof4, 1:4, fill(Float64[], 4), 2)
            b_fit!(b_enkf, asof4, 1:4, fill(Float64[], 4), 2)
            qa, ma, ha = a_fit!(a_enkf, asof6, 1:6, fill(Float64[], 6), 3)
            qb, mb, hb = b_fit!(b_enkf, asof6, 1:6, fill(Float64[], 6), 3)
            @test qa == qb
            @test ma == mb
            @test isequal(ha.value, hb.value)
        end

        @testset "warm_start resumes the outer ensemble; false restarts it cold" begin
            # `iterations` is only defensible BECAUSE the ensemble resumes, so the two travel
            # together: a cold restart must get the burn-in budget or the ablation would compare
            # a converged calibration against a truncated one.
            iters(frame) = only(
                frame[
                    (frame.parameter .== "ekp") .& (frame.statistic .== "iterations"),
                    :value,
                ]
            )
            settings = (; reopt_interval = 2, iterations = 1, burnin_iterations = 3)

            warm_enkf, warm_fit! = make_engine(23; settings...)
            warm_fit!(warm_enkf, asof3, 1:3, fill(Float64[], 3), 1)
            _, _, warm_second = warm_fit!(
                warm_enkf, asof6, 1:6, fill(Float64[], 6), 3
            )

            cold_enkf, cold_fit! = make_engine(
                23; warm_start = false, settings...
            )
            first_cold = cold_fit!(cold_enkf, asof3, 1:3, fill(Float64[], 3), 1)[3]
            _, _, cold_second = cold_fit!(
                cold_enkf, asof6, 1:6, fill(Float64[], 6), 3
            )

            # Origin 1 is cold either way, so both spend the burn-in budget there.
            @test iters(first_cold) == 3
            # The second recalibration is where they part.
            @test iters(warm_second) == 1
            @test iters(cold_second) == 3

            # And it is a genuine restart, not just a different iteration count: same seed, same
            # data, different answer, because the cold run redraws from the prior.
            estimate(frame) = only(
                frame[
                    (frame.parameter .== "R0_baseline") .&
                        (frame.statistic .== "estimate"),
                    :value,
                ]
            )
            @test estimate(warm_second) != estimate(cold_second)
        end
    end

    @testset "PF + Liu-West continues online without forecast RNG leakage" begin
        function make_online(seed; threads = true)
            return build_inference(
                PF(120; dt = 1.0, supersample = 1, threads),
                liu_west_method,
                mock_inference_vf!,
                layout,
                stochastic,
                obs_model,
                base_hyperparams,
                x0_state;
                n_ahead = 2,
                n_draws = 60,
                rng = Random.MersenneTwister(seed),
            )
        end

        online_pf, online_fit! = make_online(19)
        direct_pf, direct_fit! = make_online(19)
        @test !first(make_online(19; threads = false)).threads   # opt-out reaches LLPF
        @testset "deepcopy preserves complete particle state and RNG" begin
            pf_source, _ = make_online(29)
            pf_copy = deepcopy(pf_source)
            observation = ConfigurableEpi._observation_vectors(asof2.counts)[1]
            for filter in (pf_source, pf_copy)
                correct!(filter, Float64[], observation, base_hyperparams, 0.0)
                predict!(filter, Float64[], base_hyperparams, 0.0)
            end
            @test particles(pf_source) == particles(pf_copy)
            @test state(pf_source).xprev == state(pf_copy).xprev
            @test weights(pf_source) == weights(pf_copy)
            @test expweights(pf_source) == expweights(pf_copy)
            @test state(pf_source).j == state(pf_copy).j
            @test state(pf_source).bins == state(pf_copy).bins
            @test state(pf_source).maxw[] == state(pf_copy).maxw[]
            @test index(pf_source) == index(pf_copy)
            @test Random.rand(copy(pf_source.rng), UInt64) ==
                Random.rand(copy(pf_copy.rng), UInt64)
        end
        q2, means2, suppressed_hyper = online_fit!(
            online_pf, asof2, 1:2, fill(Float64[], 2), 1; emit_forecast = false
        )
        @test q2 === nothing
        @test isempty(suppressed_hyper)
        q_online, means3, hyper_online = online_fit!(
            online_pf, asof3, 3:3, fill(Float64[], 3), 2
        )
        direct_samples = Ref{Any}(nothing)
        q_direct, means_direct, hyper_direct = direct_fit!(
            direct_pf, asof3, 1:3, fill(Float64[], 3), 1;
            sample_callback = samples -> (direct_samples[] = copy(samples)),
        )

        @test online_pf isa AdvancedParticleFilter
        # `PF`'s threaded default must reach the LLPF filter — and the equalities in this
        # testset then pin the per-thread RNG pool's guarantee: an online continuation and a
        # from-scratch replay stay identical under threading (for this fixed thread count).
        @test online_pf.threads
        @test index(online_pf) == 4
        @test length(first(particles(online_pf))) == layout.total_dim + 1
        @test vcat(means2, means3) == means_direct
        @test q_online == q_direct
        @test size(direct_samples[]) == (2, 60)
        @test forecast_quantiles(direct_samples[]) == q_direct
        # The learned-parameter rows must be identical — that is what "no RNG leakage" means.
        # The `particle_filter` ESS rows are deliberately excluded: they summarise the
        # observations THIS call assimilated (1 for the online step, 3 for the from-scratch run),
        # so they differ by construction without implying any divergence in the filter.
        # The in-sample audit rows (`is_*`) are excluded for exactly the same reason as the ESS
        # rows: they summarise the filtered path over the observations THIS call assimilated, so
        # a 1-observation online step and a 3-observation replay disagree by construction.
        theta_rows(df) = df[
            (df.parameter .!= "particle_filter") .& .!startswith.(df.statistic, "is_"), :,
        ]
        @test theta_rows(hyper_online) == theta_rows(hyper_direct)
        # The ESS diagnostic is present and is a fraction of the particle count.
        ess = hyper_online[hyper_online.parameter .== "particle_filter", :]
        @test Set(ess.statistic) == Set(["ess_frac_min", "ess_frac_median", "ess_frac_final"])
        @test all(0.0 .<= ess.value .<= 1.0)
        # The cloud-width diagnostic: the filtered cloud's sd over the prior's, per learned
        # parameter. It is a θ row, so the online/direct equality above already covers it.
        sd_ratio = hyper_online[
            (hyper_online.parameter .== "R0_baseline") .& (hyper_online.statistic .== "sd_ratio"),
            :value,
        ]
        @test length(sd_ratio) == 1
        @test 0.0 < only(sd_ratio) < 2.0
        @test all(issorted(q_online[h, :]) for h in axes(q_online, 1))
        @test names(hyper_online) == ["parameter", "statistic", "value"]
    end

    @testset "PF + Liu-West forgetting is stateless under continuation" begin
        forgetful = LiuWest(
            (R0_baseline = r0_prior,); discount = 0.97,
            forgetting_memory_days = (R0_baseline = 5.0,),
        )
        function build(seed; hyper = forgetful)
            return build_inference(
                PF(120; dt = 1.0, supersample = 1, threads = false),
                hyper, mock_inference_vf!, layout, stochastic, obs_model,
                base_hyperparams, x0_state;
                n_ahead = 2, n_draws = 60, rng = Random.MersenneTwister(seed),
            )
        end
        theta_rows(df) = df[
            (df.parameter .!= "particle_filter") .& .!startswith.(df.statistic, "is_"), :,
        ]
        online_pf, online_fit! = build(31)
        direct_pf, direct_fit! = build(31)
        online_fit!(online_pf, asof2, 1:2, fill(Float64[], 2), 1; emit_forecast = false)
        q_online, _, hyper_online = online_fit!(online_pf, asof3, 3:3, fill(Float64[], 3), 2)
        q_direct, _, hyper_direct = direct_fit!(direct_pf, asof3, 1:3, fill(Float64[], 3), 1)
        # The forgetting is part of every Liu-West move and holds no state of its own, so a
        # persistent filter continued across calls (the `forked` policy) equals a from-scratch
        # replay, exactly as it does without forgetting.
        @test q_online == q_direct
        @test theta_rows(hyper_online) == theta_rows(hyper_direct)
        # And it is doing something: the same seed without forgetting gives another answer.
        plain_pf, plain_fit! = build(31; hyper = liu_west_method)
        q_plain, _, _ = plain_fit!(plain_pf, asof3, 1:3, fill(Float64[], 3), 1)
        @test q_plain != q_direct
    end
end

# ============================================================================
# The reporting paths — UKF fitted means and forecast quantiles, EnKF fitted means — must map the
# filtered/predicted accumulator to a count through the observation spec AT EACH TIME, exactly as
# the likelihood did. They used to read `hyper.ascertainment` (and `hyper.phi`) as constants, which
# a time-varying modifier silently contradicts. A modifier that grows with `t` makes that
# disagreement large enough to assert on.
# ============================================================================
@testset "reporting paths follow the observation spec at each observation's own time" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.03))
    stochastic = build_stochastic_update(layout, (rt_spec,))
    growing = (_latent, hyper, t) -> hyper.ascertainment * (1 + t)
    obs_model = (
        SignalObservationSpec(
            1, NegBinomialNoise(phi = (_latent, hyper, _t) -> hyper.phi);
            mean_modifier = growing, name = :reports,
        ),
    )
    base_hyperparams = (R0_baseline = 1.5, obs_scale = 0.25, ascertainment = 0.8, phi = 50.0)
    x0_state = [900.0, 50.0, 10.0, stochastic.to_unconstrained((Rt = 1.0,))[1]]
    asof4 = DataFrame(
        date = Date("2024-01-01") .+ Day.(7 .* (0:3)), counts = [8.0, 10.0, 12.0, 11.0]
    )
    dt = 1.0
    method = OptimiseHyperparams(
        (R0_baseline = positive_gaussian(:R0_baseline, 1.5, 0.4),); options = (maxiters = 1,)
    )
    ukf, fit_forecast! = build_inference(
        UKF(dt = dt, supersample = 1), method, mock_inference_vf!, layout, stochastic,
        obs_model, base_hyperparams, x0_state;
        n_ahead = 2, reopt_interval = 100, initial_optim_options = (maxiters = 1,),
    )
    us = fill(Float64[], 4)
    qmat, mean_updates, _ = fit_forecast!(ukf, asof4, 1:4, us, 1)

    # Replay the same (deterministic) filter at the fitted parameters to recover the filtered
    # state means the reporting path was computed from.
    dynamics, measure, ny, nv = ConfigurableEpi.build_dynamics_and_measurement(
        mock_inference_vf!, layout, obs_model, stochastic;
        dt = dt, supersample = 1, obs_jitter = UKF(dt = dt, supersample = 1).obs_jitter,
    )
    kf = ConfigurableEpi._ukf_backend(
        Float64, dynamics, measure, layout, x0_state, dt, ny, nv, ukf.p, NamedTuple()
    )
    solution = forward_trajectory(
        kf, us, ConfigurableEpi._observation_vectors(asof4.counts), ukf.p
    )
    spec = only(obs_model)
    acc = only(layout.accumulator_indices)
    through_spec = [
        observation_mean(
            spec, max(solution.xt[k][acc], 0.0), stochastic.extract(solution.xt[k]), ukf.p,
            (k - 1) * dt,
        ) for k in 1:4
    ]
    @test mean_updates ≈ through_spec
    # ...and NOT the constant read the old code made, so this test discriminates.
    constant_read = [max(solution.xt[k][acc], 0.0) * ukf.p.ascertainment for k in 1:4]
    @test !(mean_updates ≈ constant_read)

    # Forecast quantiles: horizon h is the modifier at t0 + h·dt, through the NB moments.
    means, covs = forecast_states(
        kf, solution.xt[end], ConfigurableEpi._symmetrize_covariance(solution.Rt[end]);
        n_ahead = 2, t0 = 3 * dt, dt = dt, p = ukf.p,
    )
    for h in 1:2
        m = observation_gaussian_moments(
            spec, means[h][acc], covs[h][acc, acc], stochastic.extract(means[h]), ukf.p,
            3 * dt + h * dt,
        )
        sd = sqrt(max(m.var, 1.0e-12))
        @test qmat[h, 4] ≈ max(0.0, quantile(Normal(m.mean, sd), 0.5))
        @test qmat[h, 7] ≈ max(0.0, quantile(Normal(m.mean, sd), 0.975))
        @test !(qmat[h, 4] ≈ max(0.0, ukf.p.ascertainment * max(means[h][acc], 0.0)))
    end

    @testset "_fitted_observation_means: each signal at each observation's own time" begin
        layout2 = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (); signal_names = (:y1, :y2))
        specs = (
            SignalObservationSpec(
                1, NegBinomialNoise(phi = 10.0); mean_modifier = (_l, h, t) -> h.a * (1 + t)
            ),
            SignalObservationSpec(2, PoissonNoise(); mean_modifier = 0.5, baseline = 3.0),
        )
        xt = [
            [900.0, 50.0, 0.0, 10.0, 20.0],
            [880.0, 60.0, 0.0, -1.0, 40.0],   # a negative filtered accumulator is floored
            [870.0, 65.0, 0.0, 30.0, 60.0],
        ]
        hyper = (a = 0.1,)
        extract = _ -> NamedTuple()
        M = ConfigurableEpi._fitted_observation_means(
            specs, layout2.accumulator_indices, extract, xt, hyper, 7.0, 1:3
        )
        @test size(M) == (3, 2)
        @test M[:, 1] ≈ [10.0 * 0.1 * (1 + 0.0), 0.0, 30.0 * 0.1 * (1 + 14.0)]
        @test M[:, 2] ≈ [20.0 * 0.5 + 3.0, 40.0 * 0.5 + 3.0, 60.0 * 0.5 + 3.0]
        v = ConfigurableEpi._fitted_observation_means(
            specs[1:1], layout2.accumulator_indices, extract, xt, hyper, 7.0, 2:3
        )
        @test v isa Vector
        @test v ≈ [0.0, 30.0 * 0.1 * (1 + 14.0)]
        # A constant modifier is the historical `max(x, 0) * ascertainment`, bit for bit.
        constant = (SignalObservationSpec(1, NegBinomialNoise(phi = 10.0); mean_modifier = (_l, h, _t) -> h.a),)
        c = ConfigurableEpi._fitted_observation_means(
            constant, layout2.accumulator_indices, extract, xt, hyper, 7.0, 1:3
        )
        @test c == [max(x[4], 0.0) * hyper.a for x in xt]
    end
end

# The UKF hyperparameter objective differentiates through the ascertainment path (`expm1` of a
# Dual rate): with the transmission level known and the latent held tight, the optimiser must
# move the rate from "no decline" toward the truth that generated the counts.
@testset "UKF optimisation recovers the ascertainment decline rate" begin
    Random.seed!(20260915)
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.05), sigma_rate = FixedParam(:sigma_Rt, 0.005))
    stochastic = build_stochastic_update(layout, (rt_spec,))
    path = AscertainmentPath(0.2, 0.0)
    obs_model = (
        SignalObservationSpec(
            1, NegBinomialNoise(phi = (_latent, hyper, _t) -> hyper.phi);
            mean_modifier = path, name = :reports,
        ),
    )
    r_true = 6.0
    truth = (
        R0_baseline = 1.2, obs_scale = 1.0, ascertainment = 1.0,
        ascertainment_decline_rate = r_true, phi = 80.0,
    )
    x0_state = [900.0, 50.0, 0.0, stochastic.to_unconstrained((Rt = 1.0,))[1]]
    dt = 1.0
    T = 60

    # Data from the deterministic mean path plus NegBinomial noise.
    dynamics = build_full_dynamics(mock_inference_vf!, stochastic, layout; dt = dt, supersample = 1)
    nw = size(build_R1(layout), 1)
    spec = only(obs_model)
    acc = only(layout.accumulator_indices)
    counts = Float64[]
    x = copy(x0_state)
    for k in 1:T
        t = (k - 1) * dt
        x = dynamics(x, nothing, truth, t, zeros(nw))
        mu = observation_mean(spec, max(x[acc], 0.0), stochastic.extract(x), truth, t)
        push!(counts, float(rand(NegativeBinomial(truth.phi, truth.phi / (truth.phi + mu)))))
    end
    asof = DataFrame(date = Date("2024-01-01") .+ Day.(0:(T - 1)), counts = counts)

    method = OptimiseHyperparams(
        (ascertainment_decline_rate = unconstrained_gaussian(:ascertainment_decline_rate, 0.0, 4.0),);
        options = (maxiters = 200,),
    )
    base_hyperparams = merge(truth, (ascertainment_decline_rate = 0.0,))
    ukf, fit_forecast! = build_inference(
        UKF(dt = dt, supersample = 1), method, mock_inference_vf!, layout, stochastic,
        obs_model, base_hyperparams, x0_state;
        n_ahead = 1, reopt_interval = 1, initial_optim_options = (maxiters = 200,),
    )
    _, _, summary = fit_forecast!(ukf, asof, 1:T, fill(Float64[], T), 1)
    estimate = only(
        summary[
            (summary.parameter .== "ascertainment_decline_rate") .&
                (summary.statistic .== "estimate"),
            :value,
        ]
    )
    @test isfinite(estimate)
    @test estimate > 0.5 * r_true
    @test abs(estimate - r_true) < abs(0.0 - r_true)
    @test ukf.p.ascertainment_decline_rate == estimate
end
