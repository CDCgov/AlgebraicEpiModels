using Test
using ConfigurableEpi
using DataFramesMeta: DataFrame, names
using Dates: Date, Day
using LinearAlgebra: BLAS, diag
using Distributions: NegativeBinomial, Normal, quantile
using LowLevelParticleFilters: AdvancedParticleFilter, UnscentedKalmanFilter, forward_trajectory,
    correct!, covariance, expweights, index, particles, predict!, reset!, state, weights
import Random

const CE = ConfigurableEpi

function mock_inference_vf!(du, u, p, t)
    hyper, latent = p
    beta = 0.3 * hyper.R0_baseline * latent.Rt / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

estimate(df, name, statistic = "estimate") =
    only(df[(df.parameter .== name) .& (df.statistic .== statistic), :value])

@testset "Inference engines" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.03))
    stochastic = build_stochastic_update(layout, (rt_spec,))
    obs_model = (
        SignalObservationSpec(
            1, NegBinomialNoise(phi = (_l, h, _t) -> h.phi);
            mean_modifier = (_l, h, _t) -> h.ascertainment, name = :reports,
        ),
    )
    base_hyperparams = (R0_baseline = 1.5, obs_scale = 0.25, ascertainment = 0.8, phi = 50.0)
    x0_state = [900.0, 50.0, 10.0, stochastic.to_unconstrained((Rt = 1.0,))[1]]
    r0_prior = positive_gaussian(:R0_baseline, 1.5, 0.4)
    model = EpiModel(;
        vectorfield! = mock_inference_vf!, layout, stochastic, observation = obs_model,
        hyperparams = base_hyperparams, priors = (R0_baseline = r0_prior,), initial_state = x0_state,
    )
    counts2 = [8.0, 10.0]
    counts3 = [8.0, 10.0, 12.0]
    counts4 = [8.0, 10.0, 12.0, 11.0]
    counts6 = [8.0, 10.0, 12.0, 11.0, 13.0, 12.0]
    ukf_kwargs = (; dt = 1.0, supersample = 1, n_ahead = 2)

    @testset "EpiModel" begin
        @test initial_state(model) == x0_state
        @test learned_names(model) == (:R0_baseline,)
        scaled = EpiModel(;
            vectorfield! = mock_inference_vf!, layout, stochastic, observation = obs_model,
            hyperparams = base_hyperparams, priors = (R0_baseline = r0_prior,),
            initial_state = hp -> x0_state .* hp.R0_baseline,
        )
        @test initial_state(scaled, merge(base_hyperparams, (R0_baseline = 2.0,))) == 2 .* x0_state
        bad(; kwargs...) = EpiModel(;
            vectorfield! = mock_inference_vf!, layout, stochastic, observation = obs_model,
            hyperparams = base_hyperparams, priors = (R0_baseline = r0_prior,), initial_state = x0_state, kwargs...,
        )
        @test_throws ArgumentError bad(; priors = (phi_typo = positive_gaussian(:phi_typo, 1.0, 1.0),))
        @test_throws ArgumentError bad(; priors = (R0_baseline = positive_gaussian(:other, 1.0, 1.0),))
        @test_throws ArgumentError bad(; priors = (;))
        @test_throws ArgumentError bad(; observation = ())
        @test_throws DimensionMismatch bad(; initial_state = x0_state[1:3])
    end

    @testset "config axes and pairings" begin
        @test UKF().obs_jitter == 1.0
        @test PF(n_particles = 100).threads
        @test !PF(n_particles = 100, threads = false).threads
        @test Optimise().window_length === nothing
        @test Optimise(window_length = 3).window_length == 3
        @test option_alias(UKF()) == "ukf" && option_alias(LiuWest()) == "liu_west" && option_alias(EKP(n_ensemble = 2, iterations = 1, burnin_iterations = 1)) == "ekp"
        @test_throws ArgumentError build_inference(UKF(obs_jitter = 0.0), Optimise(), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(UKF(), Optimise(maxiters = 0), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(PF(n_particles = 0), LiuWest(), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(UKF(), Optimise(), model; dt = 0.0, n_ahead = 2)
        @test_throws ArgumentError build_inference(UKF(), Optimise(), model; dt = 1.0, supersample = 0, n_ahead = 2)
        @test_throws ArgumentError build_inference(UKF(), Optimise(), model; dt = 1.0, n_ahead = 0)
        unsupported = try
            build_inference(PF(n_particles = 10), Optimise(), model; ukf_kwargs...)
        catch error
            error
        end
        @test unsupported isa ArgumentError
        @test occursin("PF", sprint(showerror, unsupported))
        @test occursin("Optimise", sprint(showerror, unsupported))
        @test_throws ArgumentError build_inference(UKF(), LiuWest(), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(EnKF(n_ensemble = 10), Optimise(), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(UKF(), EKP(n_ensemble = 6, iterations = 2, burnin_iterations = 3), model; ukf_kwargs...)
    end

    @testset "initial latent variance: package default, per-name override, typo caught" begin
        cov = CE._initial_state_covariance
        li = first(layout.latent_range)
        @test cov(layout, x0_state)[li, li] == DEFAULT_LATENT_VARIANCE
        from_prior = prior_unconstrained_variance(positive_gaussian(:Rt, 1.0, 0.1))
        @test from_prior != DEFAULT_LATENT_VARIANCE
        @test cov(layout, x0_state; latent_variance = (Rt = from_prior,))[li, li] ≈ from_prior
        with_tail = cov(layout, x0_state, [0.25]; latent_variance = (Rt = from_prior,))
        @test size(with_tail, 1) == layout.total_dim + 1
        @test with_tail[end, end] == 0.25
        @test_throws ArgumentError cov(layout, x0_state; latent_variance = (Rt_typo = 0.1,))
        # Compartment uncertainty scales with each compartment's own initial value.
        P0 = cov(layout, x0_state)
        for (i, x) in enumerate(x0_state[1:length(ode_names(layout))])
            @test P0[i, i] ≈ max((DEFAULT_MODEL_RELATIVE_SD * x)^2, 1.0e-6)
        end
        @test P0[1, 1] > P0[3, 3]
        @test all(diag(cov(layout, [0.0, 0.0, 0.0, x0_state[4]])) .> 0.0)
        @test cov(layout, x0_state; model_relative_sd = 0.1)[1, 1] ≈ 4 * P0[1, 1]
        @test_throws ArgumentError cov(layout, x0_state; model_relative_sd = -0.1)
    end

    @testset "_observation_vectors: single-signal shape is unchanged" begin
        obsvec = CE._observation_vectors
        @test obsvec([8.0, 10.0, 12.0]) == [[8.0], [10.0], [12.0]]
        @test obsvec([8, 10]) == [[8.0], [10.0]]
        @test eltype(obsvec([8.0])) == Vector{Float64}
        @test obsvec([[8.0, 90.0], [10.0, 91.0]]) == [[8.0, 90.0], [10.0, 91.0]]
        @test eltype(obsvec([[8, 90], [10, 91]])) == Vector{Float64}
    end

    @testset "initial accumulator variance: opt-in, keyed by signal" begin
        cov = CE._initial_state_covariance
        acc = only(layout.accumulator_indices)
        signal = only(layout.signal_names)
        pinned_x0 = [900.0, 50.0, 0.0, x0_state[4]]
        @test cov(layout, pinned_x0)[acc, acc] == 1.0e-6
        widened = cov(layout, pinned_x0; accumulator_variance = (; signal => 4.0e6))
        @test widened[acc, acc] == 4.0e6
        @test widened[1, 1] == cov(layout, pinned_x0)[1, 1]
        @test cov(layout, pinned_x0; accumulator_variance = (;)) == cov(layout, pinned_x0)
        @test cov(layout, pinned_x0; accumulator_variance = (; signal => 0.0))[acc, acc] == 1.0e-6
        @test_throws ArgumentError cov(layout, pinned_x0; accumulator_variance = (definitely_not_a_signal = 1.0,))
        @test_throws ArgumentError cov(layout, pinned_x0; accumulator_variance = (; signal => -1.0))
    end

    @testset "rolling filter checkpoints" begin
        settings = EngineSettings(; dt = 1.0, supersample = 1, n_ahead = 2)
        assembly = CE.Assembly(model, settings, 1.0)
        build_ukf(::Type{E}, hp) where {E} = CE._ukf(E, assembly, model, hp, 1.0)
        ys = CE._observation_vectors(counts6)

        # The reset-and-full-replay recursion, spelled out.
        reference = build_ukf(Float64, base_hyperparams)
        reset!(reference)
        reference_ll = zero(eltype(state(reference)))
        for k in eachindex(ys)
            time = (k - 1) * reference.Ts
            reference_ll += first(correct!(reference, Float64[], ys[k], base_hyperparams, time))
            predict!(reference, Float64[], base_hyperparams, time)
        end
        continuous = build_ukf(Float64, base_hyperparams)
        full_ll = marginal_loglik(continuous, ys, base_hyperparams)
        @test full_ll === reference_ll

        # A filter stopped after the prefix is a complete checkpoint: resuming it gives the same
        # final state as one continuous pass.
        anchor = build_ukf(Float64, base_hyperparams)
        reset!(anchor)
        prefix_ll = CE._filter_loglik!(anchor, ys, base_hyperparams, 1:2)
        checkpoint = deepcopy(anchor)
        direct_copy = deepcopy(checkpoint)
        direct_tail_ll = CE._filter_loglik!(direct_copy, ys, base_hyperparams, 3:6)
        resumed = CE._candidate_from_checkpoint(checkpoint, build_ukf, Float64, base_hyperparams)
        tail_ll = CE._filter_loglik!(resumed, ys, base_hyperparams, 3:6)
        @test prefix_ll + tail_ll ≈ full_ll
        @test direct_tail_ll == tail_ll
        @test state(direct_copy) == state(resumed)
        @test covariance(direct_copy) == covariance(resumed)
        @test index(direct_copy) == index(resumed)
        @test state(resumed) == state(continuous)
        @test covariance(resumed) == covariance(continuous)
        @test index(resumed) == index(continuous) == 6
        @test state(checkpoint) == state(anchor)

        # The Float64 checkpoint is constant with respect to θ; only the window carries derivatives.
        objective = function (values)
            hp = merge(base_hyperparams, (R0_baseline = only(values),))
            candidate = CE._candidate_from_checkpoint(checkpoint, build_ukf, eltype(values), hp)
            return CE._filter_loglik!(candidate, ys, hp, 3:6)
        end
        @test all(isfinite, CE.ForwardDiff.gradient(objective, [base_hyperparams.R0_baseline]))

        # The checkpoint follows the observation boundary: a fixed two-observation window moves
        # from 3 to 5 as T goes 4 -> 6, and the retained filter advances without a rebuild.
        cp = CE.WindowCheckpoint()
        builds = Ref(0)
        build_anchor = () -> (builds[] += 1; build_ukf(Float64, base_hyperparams))
        first_checkpoint, mode, steps = CE._prepare_window_checkpoint!(cp, build_anchor, base_hyperparams, ys, 3:4)
        @test (mode, steps, cp.start, index(first_checkpoint)) == (:anchored, 2, 3, 2)
        second_checkpoint, mode, steps = CE._prepare_window_checkpoint!(cp, build_anchor, base_hyperparams, ys, 5:6)
        @test (mode, steps, builds[], cp.start, index(second_checkpoint)) == (:advanced, 2, 1, 5, 4)
        # Value revisions behind the boundary are frozen into the checkpoint.
        state_before = copy(state(second_checkpoint))
        revised = deepcopy(ys)
        revised[1][1] += 100.0
        _, mode, steps = CE._prepare_window_checkpoint!(cp, build_anchor, base_hyperparams, revised, 5:6)
        @test (mode, steps) == (:advanced, 0)
        @test state(cp.filter) == state_before
        # A backwards boundary is refused (rebuilding the prefix would silently substitute the
        # constant-θ prefix that `window_length = nothing` computes honestly), and inertly.
        @test_throws "cannot move backwards" CE._prepare_window_checkpoint!(cp, build_anchor, base_hyperparams, ys, 4:6)
        @test (builds[], cp.start) == (1, 5)
        @test state(cp.filter) == state_before
        # A window reaching the first observation drops the checkpoint and scores everything.
        _, mode, steps = CE._prepare_window_checkpoint!(cp, build_anchor, base_hyperparams, ys, 1:6)
        @test (mode, steps, cp.filter, builds[]) == (:full_history, 0, nothing, 1)

        @test CE._loss_range(6, nothing) == 1:6
        @test CE._loss_range(6, 2) == 5:6
        @test CE._loss_range(6, 6) == 1:6
        @test CE._loss_range(3, 10) == 1:3
    end

    @testset "UKF + Optimise replays the complete window" begin
        engine = build_inference(UKF(), Optimise(reopt_interval = 2, maxiters = 1, maxiters_burnin = 2), model; ukf_kwargs...)
        @test engine isa UKFOptimiseEngine
        @test engine.filter isa UnscentedKalmanFilter
        result = fit_forecast!(engine, counts3, 1)
        @test size(result.quantiles) == (2, length(DEFAULT_QS))
        @test all(issorted(result.quantiles[h, :]) for h in axes(result.quantiles, 1))
        @test length(result.fitted_means) == 3
        @test result.samples === nothing
        @test names(result.summary) == ["parameter", "statistic", "value"]
        @test engine.hyperparams.R0_baseline == estimate(result.summary, "R0_baseline")
        @test engine.filter.p.R0_baseline == engine.hyperparams.R0_baseline
        spread = result.summary[result.summary.statistic .== "fc_log_sd_h2", :]
        @test spread.parameter == ["Rt"]
        @test only(spread.value) > 0.0
        @test_throws ArgumentError fit_forecast!(engine, counts3, 2; update_range = 2:3)
        @test_throws ArgumentError fit_forecast!(engine, counts3, 0)

        # An off-cadence origin keeps the previous optimum.
        previous = engine.hyperparams.R0_baseline
        suppressed = fit_forecast!(engine, counts3, 2; emit_forecast = false)
        @test suppressed.quantiles === nothing && suppressed.samples === nothing
        @test length(suppressed.fitted_means) == 3
        @test all(suppressed.summary.statistic .== "estimate")
        @test engine.hyperparams.R0_baseline == previous
        reoptimized = fit_forecast!(engine, counts3, 3)
        @test engine.hyperparams.R0_baseline == estimate(reoptimized.summary, "R0_baseline")

        @testset "window length is independent of re-optimization cadence" begin
            @test_throws ArgumentError build_inference(UKF(), Optimise(window_length = 0), model; ukf_kwargs...)
            windowed = build_inference(
                UKF(), Optimise(reopt_interval = 2, maxiters = 1, maxiters_burnin = 1, window_length = 2), model; ukf_kwargs...,
            )
            # Re-optimise at origins 1 and 3 while T moves 3 -> 4 -> 6: the boundary moves 2 -> 5.
            r3 = fit_forecast!(windowed, counts3, 1)
            fit_forecast!(windowed, counts4, 2)
            r6 = fit_forecast!(windowed, counts6, 3)
            @test size(r3.quantiles) == size(r6.quantiles) == (2, length(DEFAULT_QS))
            @test length(r3.fitted_means) == 3
            @test length(r6.fitted_means) == 6
            @test all(isfinite, r6.quantiles)
            @test windowed.hyperparams.R0_baseline == estimate(r6.summary, "R0_baseline")
        end

        @testset "warm_start controls where each re-optimization begins" begin
            make(; warm_start = true) = build_inference(
                UKF(), Optimise(; reopt_interval = 2, maxiters = 1, maxiters_burnin = 2, warm_start), model; ukf_kwargs...,
            )
            # Cold: origin 3 restarts from the configured values on the same data, so it returns
            # to the same place origin 1 reached.
            cold = make(warm_start = false)
            first_estimate = estimate(fit_forecast!(cold, counts3, 1).summary, "R0_baseline")
            @test estimate(fit_forecast!(cold, counts3, 3).summary, "R0_baseline") ≈ first_estimate
            warm = make()
            fit_forecast!(warm, counts3, 1)
            second = fit_forecast!(warm, counts3, 3)
            @test warm.hyperparams.R0_baseline == estimate(second.summary, "R0_baseline")
        end

        @testset "custom optimiser schedule and the RunConfig form" begin
            single_stage = build_inference(
                UKF(), Optimise(maxiters = 3, maxiters_burnin = 3), model; ukf_kwargs..., optimiser = CE.LBFGS(),
            )
            @test isfinite(estimate(fit_forecast!(single_stage, counts3, 1).summary, "R0_baseline"))
            cfg = from_dict(
                RunConfig, Dict(
                    "io" => Dict("data" => "d", "model_id" => "m", "forecast_df" => "f.csv", "loc" => "ny"),
                    "n_ahead" => 2, "step_days" => 1, "supersample" => 1, "burnin_observations" => 2, "n_draws" => 7,
                    "input" => Dict("counts" => Dict{String, Any}()), "filter" => Dict("ukf" => Dict{String, Any}()),
                    "hyper" => Dict("optimise" => Dict("maxiters" => 1, "maxiters_burnin" => 1)),
                    "epi" => Dict("mock" => Dict{String, Any}()),
                ),
            )
            from_cfg = build_inference(validate_run_config(cfg), model)
            @test from_cfg isa UKFOptimiseEngine
            @test (from_cfg.settings.dt, from_cfg.settings.supersample, from_cfg.settings.n_ahead, from_cfg.settings.n_draws) == (1.0, 1, 2, 7)
        end
    end

    @testset "EnKF + EKP calibration" begin
        ekp(; n_ensemble = 6, iterations = 2, burnin_iterations = 3, kwargs...) =
            EKP(; n_ensemble, iterations, burnin_iterations, kwargs...)
        function make_engine(
                seed; reopt_interval = 2, window_length = nothing, warm_start = true,
                inner_threads = false, outer_threads = false, kwargs...,
            )
            return build_inference(
                EnKF(n_ensemble = 40, threads = inner_threads),
                ekp(; reopt_interval, window_length, warm_start, threads = outer_threads, kwargs...), model;
                dt = 1.0, supersample = 1, n_ahead = 2, n_draws = 50, rng = Random.MersenneTwister(seed),
            )
        end

        @test ekp().window_length === nothing
        @test !ekp().threads
        @test ekp(window_length = 4).window_length == 4
        @test_throws ArgumentError build_inference(EnKF(n_ensemble = 1), ekp(), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(EnKF(n_ensemble = 40), ekp(iterations = 0), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(EnKF(n_ensemble = 40), ekp(n_ensemble = 1), model; ukf_kwargs...)
        @test_throws "nested ensemble threading" make_engine(5; inner_threads = true, outer_threads = true)
        # Rank guards: the ensemble must span the state, the outer ensemble the learned set.
        @test_throws ArgumentError build_inference(EnKF(n_ensemble = 3), ekp(), model; ukf_kwargs...)
        @test_throws ArgumentError build_inference(EnKF(n_ensemble = 40), ekp(n_ensemble = 1), model; ukf_kwargs...)
        @test_throws ArgumentError make_engine(5; window_length = -1)

        @testset "outer candidate scheduling and diagnostics" begin
            @test_logs (:warn, r"only one thread") !CE._outer_ekp_threads_enabled(true; nthreads = 1)
            @test CE._outer_ekp_threads_enabled(true; nthreads = 2)
            phi = reshape([-2.0, 0.25, 3.0], 1, :)
            score(theta) = theta[1] == 3.0 ? error("deliberate candidate failure") : theta[1]
            serial = CE._evaluate_ekp_candidates(score, phi; threads = false)
            original_blas_threads = BLAS.get_num_threads()
            BLAS.set_num_threads(2)
            try
                threaded = CE._evaluate_ekp_candidates(score, phi; threads = true)
                @test isequal(threaded[1], serial[1])
                @test threaded[2] == serial[2]
                @test findall(!isnothing, threaded[3]) == [3]
                @test BLAS.get_num_threads() == 2
            finally
                BLAS.set_num_threads(original_blas_threads)
            end
            G, positive_logliks, failures = serial
            @test G[1, 1] == 2.0
            @test G[1, 2] == sqrt(eps())
            @test isnan(G[1, 3])
            warned = Ref(false)
            @test_logs (:warn, r"positive filter log-likelihood") (:warn, r"outer candidate failed") CE._report_ekp_candidate_diagnostics!(
                warned, positive_logliks, failures; forecast_number = 1, iteration = 1,
            )
            @test warned[]
        end

        engine = make_engine(11)
        @test engine isa EnKFEKPEngine
        @test engine.filter isa AugmentedEnsembleKalmanFilter
        # One latent slot plus one accumulator whisker against four state slots: `nw != nx`.
        @test (engine.filter.nw, engine.filter.nx) == (2, 4)

        @testset "deepcopy preserves complete ensemble state and RNG" begin
            ys = CE._observation_vectors(counts3)
            source = make_engine(23).filter
            continuous = make_engine(23).filter
            full_ll = marginal_loglik(continuous, ys, base_hyperparams)
            reset!(source)
            prefix_ll = CE._filter_loglik!(source, ys, base_hyperparams, 1:1)
            checkpoint = deepcopy(source)
            resumed = CE._candidate_from_checkpoint(checkpoint, nothing, Float64, base_hyperparams)
            tail_ll = CE._filter_loglik!(resumed, ys, base_hyperparams, 2:3)
            @test prefix_ll + tail_ll ≈ full_ll
            @test resumed.ensemble == continuous.ensemble
            @test state(resumed) == state(continuous)
            @test covariance(resumed) == covariance(continuous)
            @test index(resumed) == index(continuous)
            @test Random.rand(copy(resumed.rng), UInt64) == Random.rand(copy(continuous.rng), UInt64)
            @test checkpoint.ensemble == source.ensemble
            # The copy carries the candidate's parameters, not the checkpoint's.
            shifted = merge(base_hyperparams, (R0_baseline = 3.7,))
            @test CE._candidate_from_checkpoint(checkpoint, nothing, Float64, shifted).p.R0_baseline == 3.7
            @test checkpoint.p.R0_baseline == base_hyperparams.R0_baseline
            # Every candidate starts from the same checkpoint RNG position.
            score = function (r0)
                hp = merge(base_hyperparams, (R0_baseline = r0,))
                return CE._filter_loglik!(CE._candidate_from_checkpoint(checkpoint, nothing, Float64, hp), ys, hp, 2:3)
            end
            score_a = score(1.3)
            score(1.7)
            @test score(1.3) == score_a
        end

        result = fit_forecast!(engine, counts3, 1)
        @test size(result.quantiles) == (2, length(DEFAULT_QS))
        @test size(result.samples) == (2, 50)
        @test forecast_quantiles(result.samples) == result.quantiles
        @test all(issorted(result.quantiles[h, :]) for h in axes(result.quantiles, 1))
        @test all(result.quantiles .>= 0.0)
        @test all(isfinite, result.quantiles)
        @test length(result.fitted_means) == 3
        @test names(result.summary) == ["parameter", "statistic", "value"]
        @test engine.hyperparams.R0_baseline == estimate(result.summary, "R0_baseline")
        @test engine.filter.p.R0_baseline == engine.hyperparams.R0_baseline
        # The outer spread is a calibration diagnostic, named so it is not read as a posterior.
        @test estimate(result.summary, "R0_baseline", "ekp_q05") <= estimate(result.summary, "R0_baseline", "ekp_q95")
        @test_throws ArgumentError fit_forecast!(engine, counts3, 2; update_range = 2:3)

        @testset "runs the configured iteration counts" begin
            e = make_engine(11; reopt_interval = 1, iterations = 2, burnin_iterations = 4)
            @test estimate(fit_forecast!(e, counts3, 1).summary, "ekp", "iterations") == 4.0
            @test estimate(fit_forecast!(e, counts3, 2).summary, "ekp", "iterations") == 2.0
        end

        @testset "recalibration follows reopt_interval and warm-starts" begin
            e = make_engine(11; reopt_interval = 2)
            fit_forecast!(e, counts3, 1)
            after_first = e.hyperparams.R0_baseline
            held = fit_forecast!(e, counts3, 2; emit_forecast = false)
            @test held.quantiles === nothing && held.samples === nothing
            @test length(held.fitted_means) == 3
            @test e.hyperparams.R0_baseline == after_first
            @test estimate(held.summary, "R0_baseline") == after_first
            recalibrated = fit_forecast!(e, counts3, 3)
            @test e.hyperparams.R0_baseline == estimate(recalibrated.summary, "R0_baseline")
        end

        @testset "serial and outer-threaded engines agree exactly" begin
            # Common random numbers: G(θ) depends only on θ, so candidate evaluation order cannot
            # change the calibration path. BLAS is held at one thread for both complete paths.
            a, b = make_engine(7), make_engine(7; outer_threads = true)
            original_blas_threads = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
            ra = rb = nothing
            try
                ra = fit_forecast!(a, counts3, 1)
                rb = fit_forecast!(b, counts3, 1)
            finally
                BLAS.set_num_threads(original_blas_threads)
            end
            @test ra.quantiles == rb.quantiles
            @test ra.fitted_means == rb.fitted_means
            @test isequal(ra.summary.value, rb.summary.value)   # `is_tau_days` may be NaN
            @test fit_forecast!(make_engine(8), counts3, 1).quantiles != ra.quantiles
        end

        @testset "windowed EKP remains reproducible across a distinct cadence" begin
            settings = (; reopt_interval = 2, window_length = 2, iterations = 1, burnin_iterations = 1)
            a, b = make_engine(31; settings...), make_engine(31; settings...)
            for e in (a, b)
                fit_forecast!(e, counts3, 1)
                fit_forecast!(e, counts4, 2)
            end
            ra, rb = fit_forecast!(a, counts6, 3), fit_forecast!(b, counts6, 3)
            @test ra.quantiles == rb.quantiles
            @test ra.fitted_means == rb.fitted_means
            @test isequal(ra.summary.value, rb.summary.value)
        end

        @testset "warm_start resumes the outer ensemble; false restarts it cold" begin
            iters(frame) = estimate(frame, "ekp", "iterations")
            settings = (; reopt_interval = 2, iterations = 1, burnin_iterations = 3)
            warm = make_engine(23; settings...)
            fit_forecast!(warm, counts3, 1)
            warm_second = fit_forecast!(warm, counts6, 3).summary
            cold = make_engine(23; warm_start = false, settings...)
            first_cold = fit_forecast!(cold, counts3, 1).summary
            cold_second = fit_forecast!(cold, counts6, 3).summary
            @test iters(first_cold) == 3          # origin 1 is cold either way
            @test iters(warm_second) == 1
            @test iters(cold_second) == 3
            @test estimate(warm_second, "R0_baseline") != estimate(cold_second, "R0_baseline")
        end
    end

    @testset "PF + Liu-West continues online without forecast RNG leakage" begin
        make_online(seed; threads = true, hyper = LiuWest(discount = 0.97)) = build_inference(
            PF(; n_particles = 120, threads), hyper, model;
            dt = 1.0, supersample = 1, n_ahead = 2, n_draws = 60, rng = Random.MersenneTwister(seed),
        )
        online, direct = make_online(19), make_online(19)
        @test online isa PFLiuWestEngine
        @test !make_online(19; threads = false).filter.threads

        @testset "deepcopy preserves complete particle state and RNG" begin
            pf_source = make_online(29).filter
            pf_copy = deepcopy(pf_source)
            observation = CE._observation_vectors(counts2)[1]
            for filter in (pf_source, pf_copy)
                correct!(filter, Float64[], observation, base_hyperparams, 0.0)
                predict!(filter, Float64[], base_hyperparams, 0.0)
            end
            @test particles(pf_source) == particles(pf_copy)
            @test state(pf_source).xprev == state(pf_copy).xprev
            @test weights(pf_source) == weights(pf_copy)
            @test expweights(pf_source) == expweights(pf_copy)
            @test index(pf_source) == index(pf_copy)
            @test Random.rand(copy(pf_source.rng), UInt64) == Random.rand(copy(pf_copy.rng), UInt64)
        end

        r2 = fit_forecast!(online, counts2, 1; emit_forecast = false)
        @test r2.quantiles === nothing && r2.samples === nothing
        @test isempty(r2.summary)
        r_online = fit_forecast!(online, counts3, 2; update_range = 3:3)
        r_direct = fit_forecast!(direct, counts3, 1)
        @test online.filter isa AdvancedParticleFilter
        @test online.filter.threads
        @test index(online.filter) == 4
        @test length(first(particles(online.filter))) == layout.total_dim + 1
        # An online continuation and a from-scratch replay are identical under threading (for
        # this thread count): the per-thread RNG pool's guarantee.
        @test vcat(r2.fitted_means, r_online.fitted_means) == r_direct.fitted_means
        @test r_online.quantiles == r_direct.quantiles
        @test size(r_direct.samples) == (2, 60)
        @test forecast_quantiles(r_direct.samples) == r_direct.quantiles
        # The ESS and in-sample audit rows summarise the observations THIS call assimilated, so
        # only the θ rows are compared.
        theta_rows(df) = df[(df.parameter .!= "particle_filter") .& .!startswith.(df.statistic, "is_"), :]
        @test theta_rows(r_online.summary) == theta_rows(r_direct.summary)
        ess = r_online.summary[r_online.summary.parameter .== "particle_filter", :]
        @test Set(ess.statistic) == Set(["ess_frac_min", "ess_frac_median", "ess_frac_final"])
        @test all(0.0 .<= ess.value .<= 1.0)
        @test 0.0 < estimate(r_online.summary, "R0_baseline", "sd_ratio") < 2.0
        @test all(issorted(r_online.quantiles[h, :]) for h in axes(r_online.quantiles, 1))
        @test names(r_online.summary) == ["parameter", "statistic", "value"]

        @testset "derived hyperparameters are summarised like learned ones" begin
            derived_model = EpiModel(;
                vectorfield! = mock_inference_vf!, layout, stochastic, observation = obs_model,
                hyperparams = base_hyperparams, priors = (R0_baseline = r0_prior,), initial_state = x0_state,
                derived_hyperparameters = hp -> (R0_doubled = 2 * hp.R0_baseline,),
            )
            e = build_inference(PF(n_particles = 60, threads = false), LiuWest(), derived_model; dt = 1.0, supersample = 1, n_ahead = 1, n_draws = 20)
            summary = fit_forecast!(e, counts3, 1).summary
            @test estimate(summary, "R0_doubled", "mean") ≈ 2 * estimate(summary, "R0_baseline", "mean")
        end

        @testset "forgetting is stateless under continuation" begin
            forgetful = LiuWest(discount = 0.97, forgetting_memory_days = Dict("R0_baseline" => 5.0))
            a = make_online(31; threads = false, hyper = forgetful)
            b = make_online(31; threads = false, hyper = forgetful)
            fit_forecast!(a, counts2, 1; emit_forecast = false)
            r_a = fit_forecast!(a, counts3, 2; update_range = 3:3)
            r_b = fit_forecast!(b, counts3, 1)
            @test r_a.quantiles == r_b.quantiles
            @test theta_rows(r_a.summary) == theta_rows(r_b.summary)
            plain = make_online(31; threads = false, hyper = LiuWest(discount = 0.97))
            @test fit_forecast!(plain, counts3, 1).quantiles != r_b.quantiles
            # Model defaults are merged under the run config's forgetting; unknown names are rejected.
            with_defaults = EpiModel(;
                vectorfield! = mock_inference_vf!, layout, stochastic, observation = obs_model,
                hyperparams = base_hyperparams, priors = (R0_baseline = r0_prior,), initial_state = x0_state,
                forgetting_memory_days = (R0_baseline = 5.0,),
            )
            c = build_inference(PF(n_particles = 120, threads = false), LiuWest(discount = 0.97), with_defaults; dt = 1.0, supersample = 1, n_ahead = 2, n_draws = 60, rng = Random.MersenneTwister(31))
            @test fit_forecast!(c, counts3, 1).quantiles == r_b.quantiles
            @test_throws ArgumentError build_inference(
                PF(n_particles = 10), LiuWest(forgetting_memory_days = Dict("nope" => 5.0)), model; ukf_kwargs...,
            )
        end
    end
end

# The reporting paths (fitted means, forecast quantiles) must map accumulators to counts through
# the observation spec at each time, exactly as the likelihood did; a modifier that grows with
# `t` makes the difference from a constant read large enough to assert on.
@testset "reporting paths follow the observation spec at each observation's own time" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.03))
    stochastic = build_stochastic_update(layout, (rt_spec,))
    growing = (_l, h, t) -> h.ascertainment * (1 + t)
    obs_model = (SignalObservationSpec(1, NegBinomialNoise(phi = (_l, h, _t) -> h.phi); mean_modifier = growing, name = :reports),)
    base_hyperparams = (R0_baseline = 1.5, obs_scale = 0.25, ascertainment = 0.8, phi = 50.0)
    x0_state = [900.0, 50.0, 10.0, stochastic.to_unconstrained((Rt = 1.0,))[1]]
    counts4 = [8.0, 10.0, 12.0, 11.0]
    dt = 1.0
    model = EpiModel(;
        vectorfield! = mock_inference_vf!, layout, stochastic, observation = obs_model, hyperparams = base_hyperparams,
        priors = (R0_baseline = positive_gaussian(:R0_baseline, 1.5, 0.4),), initial_state = x0_state,
    )
    engine = build_inference(UKF(), Optimise(reopt_interval = 100, maxiters = 1, maxiters_burnin = 1), model; dt, supersample = 1, n_ahead = 2)
    result = fit_forecast!(engine, counts4, 1)

    # Replay the same deterministic filter at the fitted parameters.
    hp = engine.hyperparams
    kf = CE._ukf(Float64, CE.Assembly(model, engine.settings, UKF().obs_jitter), model, hp, dt)
    solution = forward_trajectory(kf, fill(Float64[], 4), CE._observation_vectors(counts4), hp)
    spec = only(obs_model)
    acc = only(layout.accumulator_indices)
    through_spec = [
        observation_mean(spec, max(solution.xt[k][acc], 0.0), stochastic.extract(solution.xt[k]), hp, (k - 1) * dt) for k in 1:4
    ]
    @test result.fitted_means ≈ through_spec
    @test !(result.fitted_means ≈ [max(solution.xt[k][acc], 0.0) * hp.ascertainment for k in 1:4])

    means, covs = forecast_states(kf, solution.xt[end], CE._symmetrize_covariance(solution.Rt[end]); n_ahead = 2, t0 = 3 * dt, dt, p = hp)
    for h in 1:2
        m = observation_gaussian_moments(spec, means[h][acc], covs[h][acc, acc], stochastic.extract(means[h]), hp, 3 * dt + h * dt)
        sd = sqrt(max(m.var, 1.0e-12))
        @test result.quantiles[h, 4] ≈ max(0.0, quantile(Normal(m.mean, sd), 0.5))
        @test result.quantiles[h, 7] ≈ max(0.0, quantile(Normal(m.mean, sd), 0.975))
        @test !(result.quantiles[h, 4] ≈ max(0.0, hp.ascertainment * max(means[h][acc], 0.0)))
    end

    @testset "_fitted_observation_means: each signal at each observation's own time" begin
        layout2 = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (); signal_names = (:y1, :y2))
        specs = (
            SignalObservationSpec(1, NegBinomialNoise(phi = 10.0); mean_modifier = (_l, h, t) -> h.a * (1 + t)),
            SignalObservationSpec(2, PoissonNoise(); mean_modifier = 0.5, baseline = 3.0),
        )
        xt = [[900.0, 50.0, 0.0, 10.0, 20.0], [880.0, 60.0, 0.0, -1.0, 40.0], [870.0, 65.0, 0.0, 30.0, 60.0]]
        hyper = (a = 0.1,)
        extract = _ -> NamedTuple()
        M = CE._fitted_observation_means(specs, layout2.accumulator_indices, extract, xt, hyper, 7.0, 1:3)
        @test size(M) == (3, 2)
        @test M[:, 1] ≈ [10.0 * 0.1 * (1 + 0.0), 0.0, 30.0 * 0.1 * (1 + 14.0)]   # a negative accumulator is floored
        @test M[:, 2] ≈ [20.0 * 0.5 + 3.0, 40.0 * 0.5 + 3.0, 60.0 * 0.5 + 3.0]
        v = CE._fitted_observation_means(specs[1:1], layout2.accumulator_indices, extract, xt, hyper, 7.0, 2:3)
        @test v isa Vector
        @test v ≈ [0.0, 30.0 * 0.1 * (1 + 14.0)]
        constant = (SignalObservationSpec(1, NegBinomialNoise(phi = 10.0); mean_modifier = (_l, h, _t) -> h.a),)
        c = CE._fitted_observation_means(constant, layout2.accumulator_indices, extract, xt, hyper, 7.0, 1:3)
        @test c == [max(x[4], 0.0) * hyper.a for x in xt]
    end
end

# The UKF objective differentiates through the ascertainment path (`expm1` of a Dual rate): with
# the transmission level known and the latent held tight, the optimiser must move the rate from
# "no decline" toward the truth that generated the counts.
@testset "UKF optimisation recovers the ascertainment decline rate" begin
    Random.seed!(20260915)
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.05), sigma_rate = FixedParam(:sigma_Rt, 0.005))
    stochastic = build_stochastic_update(layout, (rt_spec,))
    path = AscertainmentPath(0.2, 0.0)
    obs_model = (SignalObservationSpec(1, NegBinomialNoise(phi = (_l, h, _t) -> h.phi); mean_modifier = path, name = :reports),)
    r_true = 6.0
    truth = (R0_baseline = 1.2, obs_scale = 1.0, ascertainment = 1.0, ascertainment_decline_rate = r_true, phi = 80.0)
    x0_state = [900.0, 50.0, 0.0, stochastic.to_unconstrained((Rt = 1.0,))[1]]
    dt = 1.0
    T = 60
    dynamics = build_full_dynamics(mock_inference_vf!, stochastic, layout; dt, supersample = 1)
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
    model = EpiModel(;
        vectorfield! = mock_inference_vf!, layout, stochastic, observation = obs_model,
        hyperparams = merge(truth, (ascertainment_decline_rate = 0.0,)),
        priors = (ascertainment_decline_rate = unconstrained_gaussian(:ascertainment_decline_rate, 0.0, 4.0),),
        initial_state = x0_state,
    )
    engine = build_inference(UKF(), Optimise(maxiters = 200, maxiters_burnin = 200), model; dt, supersample = 1, n_ahead = 1)
    summary = fit_forecast!(engine, counts, 1).summary
    est = estimate(summary, "ascertainment_decline_rate")
    @test isfinite(est)
    @test est > 0.5 * r_true
    @test abs(est - r_true) < abs(0.0 - r_true)
    @test engine.hyperparams.ascertainment_decline_rate == est
end
