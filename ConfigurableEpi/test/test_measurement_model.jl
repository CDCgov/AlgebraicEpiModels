using Test
using ConfigurableEpi
using StaticArrays
using LabelledArrays

# Latents stored under an identity constraint, so constrained == unconstrained in these checks.
_sto(layout) = build_stochastic_update(
    layout, Tuple(RWParamSpec(n; init = unconstrained_gaussian(n, 0.0, 1.0), sigma_rate = 0.1) for n in layout.latent_names)
)


@testset "Measurement Model" begin

    @testset "Reporting fractions scale only fitted observation times" begin
        modifier = ReportingFractionModifier((latent, hyper, t) -> 2.0, [1.0, 0.5, missing])
        spec = SignalObservationSpec(1, NegBinomialNoise(phi = 50.0); mean_modifier = modifier)
        @test observation_mean(spec, 100.0, nothing, nothing, 0.0) == 200.0
        @test observation_mean(spec, 100.0, nothing, nothing, 1.0) == 100.0
        @test observation_mean(spec, 100.0, nothing, nothing, 2.0) == 200.0
        @test observation_mean(spec, 100.0, nothing, nothing, 3.0) == 200.0

        set_reporting_fractions!(modifier, [0.25, 1.0])
        @test observation_mean(spec, 100.0, nothing, nothing, 0.0) == 50.0
        @test observation_mean(spec, 100.0, nothing, nothing, 2.0) == 200.0
        @test_throws ArgumentError set_reporting_fractions!(modifier, [1.1])
        @test_throws ArgumentError ReportingFractionModifier((l, h, t) -> 1.0, [-0.1])

        inner = (latent, hyper, t) -> 3.0
        full = build_reporting_observation(CorrectedObservations(), inner, [0.5, 0.5])
        @test full isa FullReporting
        @test full(nothing, nothing, 0.0) == 3.0
        @test set_reporting_fractions!(full, [0.1]) === full
        thinned = build_reporting_observation(ThinnedObservations(), inner, [0.5, 1.0])
        @test thinned isa ReportingFractionModifier
        @test thinned(nothing, nothing, 0.0) == 1.5
        @test_throws ArgumentError build_reporting_observation(ThinnedObservations(), inner, nothing)
        @test option_alias(ThinnedObservations()) == "thinned" && ThinnedObservations() isa ReportingMode
    end

    # ========================================================================
    # ObservationNoiseSpec types
    #
    # Noise parameters are now plain `Real` literals or `(latent, hyper, t)`
    # functions (mirroring the rate-function convention) — no ParamSpec.
    # ========================================================================
    @testset "NegBinomialNoise construction" begin
        @testset "with fixed (literal) params" begin
            noise = NegBinomialNoise(0.1, 10.0)
            @test noise.sigma_mult == 0.1
            @test noise.phi == 10.0
        end

        @testset "keyword constructor defaults sigma_mult to 0" begin
            noise = NegBinomialNoise(phi = 10.0)
            @test noise.sigma_mult == 0.0
            @test noise.phi == 10.0
        end

        @testset "with (latent, hyper, t) functions" begin
            noise = NegBinomialNoise(
                (latent, hyper, t) -> hyper.sigma_obs,
                (latent, hyper, t) -> hyper.phi
            )
            @test noise.sigma_mult isa Function
            @test noise.phi isa Function
        end
    end

    @testset "PoissonNoise construction" begin
        @testset "pure Poisson (no multiplicative)" begin
            noise = PoissonNoise()
            @test isnothing(noise.sigma_mult)
        end

        @testset "with multiplicative noise" begin
            noise = PoissonNoise(0.1)
            @test noise.sigma_mult == 0.1
        end

        @testset "explicit nothing" begin
            noise = PoissonNoise(nothing)
            @test isnothing(noise.sigma_mult)
        end
    end

    @testset "LogNormalNoise construction" begin
        noise = LogNormalNoise(0.2)
        @test noise.sigma == 0.2
    end

    # ========================================================================
    # n_noise_terms
    # ========================================================================
    @testset "n_noise_terms" begin
        @testset "NegBinomialNoise" begin
            noise = NegBinomialNoise(0.1, 10.0)
            @test n_noise_terms(noise) == 1  # single Gaussian term carries the full variance
        end

        @testset "PoissonNoise pure" begin
            noise = PoissonNoise()
            @test n_noise_terms(noise) == 1
        end

        @testset "PoissonNoise with mult" begin
            noise = PoissonNoise(0.1)
            @test n_noise_terms(noise) == 2
        end

        @testset "LogNormalNoise" begin
            noise = LogNormalNoise(0.1)
            @test n_noise_terms(noise) == 1
        end
    end

    # ========================================================================
    # SignalObservationSpec
    # ========================================================================
    @testset "SignalObservationSpec construction" begin
        @testset "basic construction" begin
            noise = NegBinomialNoise(0.1, 10.0)
            spec = SignalObservationSpec(1, noise)
            @test spec isa ObservationSpec{ConfigurableEpi.Signal}
            @test spec.raw.index == 1
            @test spec.noise_spec === noise
            @test spec.name == :obs_1  # default name
        end

        @testset "custom name" begin
            noise = PoissonNoise()
            spec = SignalObservationSpec(2, noise; name = :hospitalizations)
            @test spec.raw.index == 2
            @test spec.name == :hospitalizations
        end

        @testset "invalid signal_idx throws" begin
            noise = PoissonNoise()
            @test_throws ArgumentError SignalObservationSpec(0, noise)
            @test_throws ArgumentError SignalObservationSpec(-1, noise)
        end
    end

    # ========================================================================
    # AggregatedSignalSpec
    # ========================================================================
    @testset "AggregatedSignalSpec construction" begin
        @testset "basic construction" begin
            noise = NegBinomialNoise(0.1, 10.0)
            spec = AggregatedSignalSpec([1, 2, 3], noise; name = :total_hosp)
            @test spec.raw.indices == [1, 2, 3]
            @test spec.noise_spec === noise
            @test spec.name == :total_hosp
        end

        @testset "default name" begin
            noise = PoissonNoise()
            spec = AggregatedSignalSpec([1, 2], noise)
            @test spec.name == :obs_aggregated
        end

        @testset "empty indices for 'all signals'" begin
            noise = PoissonNoise()
            spec = AggregatedSignalSpec(noise; name = :obs_total)
            @test spec.raw isa ConfigurableEpi.AllSignals  # expanded at build time
            @test spec.name == :obs_total
            layout = StateLayout((:S,), (:O_a, :O_b), (); signal_names = (:a, :b))
            @test resolve_observation(spec, layout).raw.indices == [1, 2]
            @test_throws ArgumentError ConfigurableEpi.compute_true_mean(spec, zeros(3), layout.accumulator_indices)
        end

        @testset "invalid indices throw" begin
            noise = PoissonNoise()
            # An explicit empty list is not "all signals".
            @test_throws ArgumentError AggregatedSignalSpec(Int[], noise)
            @test_throws ArgumentError AggregatedSignalSpec([0], noise)
            @test_throws ArgumentError AggregatedSignalSpec([-1, 2], noise)
        end
    end

    @testset "n_noise_terms for observation specs" begin
        noise_nb = NegBinomialNoise(0.1, 10.0)
        noise_pois = PoissonNoise()

        @testset "SignalObservationSpec" begin
            spec = SignalObservationSpec(1, noise_nb)
            @test n_noise_terms(spec) == 1
        end

        @testset "AggregatedSignalSpec" begin
            spec = AggregatedSignalSpec([1, 2], noise_pois)
            @test n_noise_terms(spec) == 1
        end
    end

    # ========================================================================
    # apply_noise — now takes (spec, true_mean, v, latent, hyper, t)
    # ========================================================================
    @testset "apply_noise" begin
        latent = NamedTuple()
        hyper = (sigma_obs = 0.1, phi = 10.0)

        @testset "NegBinomialNoise with zero noise" begin
            noise = NegBinomialNoise(0.0, 100.0)  # large phi → low variance
            v = [0.0]  # one Gaussian term carries the full variance
            result = apply_noise(noise, 100.0, v, latent, hyper, 0.0)
            # With v=0, should be close to true mean
            @test result ≈ 100.0 atol = 1.0e-4
        end

        @testset "NegBinomialNoise is non-negative" begin
            noise = NegBinomialNoise(0.1, 1.0)
            v = [-5.0]  # large negative noise
            result = apply_noise(noise, 10.0, v, latent, hyper, 0.0)
            @test result >= 0.0
        end

        @testset "NegBinomialNoise variance formula μ + μ²/φ + (σμ)²" begin
            μ, φ, σ = 100.0, 100.0, 0.5
            noise = NegBinomialNoise(σ, φ)
            std = sqrt(μ + μ^2 / φ + (σ * μ)^2)

            # One additive Gaussian term: y = μ + std * v[1]
            @test apply_noise(noise, μ, [1.0], latent, hyper, 0.0) ≈ μ + std atol = 1.0e-4
            @test apply_noise(noise, μ, [-1.0], latent, hyper, 0.0) ≈ μ - std atol = 1.0e-4

            # Reporting noise σ strictly inflates the spread vs σ = 0
            noise0 = NegBinomialNoise(0.0, φ)
            @test apply_noise(noise, μ, [1.0], latent, hyper, 0.0) >
                apply_noise(noise0, μ, [1.0], latent, hyper, 0.0)
        end

        @testset "PoissonNoise pure" begin
            noise = PoissonNoise()
            v = [0.0]
            result = apply_noise(noise, 100.0, v, latent, hyper, 0.0)
            @test result ≈ 100.0 atol = 1.0e-4
        end

        @testset "PoissonNoise pure variance scaling" begin
            noise = PoissonNoise()
            v = [1.0]  # one std dev
            result = apply_noise(noise, 100.0, v, latent, hyper, 0.0)
            # μ + sqrt(μ) * 1 = 100 + 10 = 110
            @test result ≈ 110.0 atol = 1.0e-4
        end

        @testset "PoissonNoise with mult" begin
            noise = PoissonNoise(0.0)
            v = [0.0, 0.0]
            result = apply_noise(noise, 100.0, v, latent, hyper, 0.0)
            @test result ≈ 100.0 atol = 1.0e-4
        end

        @testset "LogNormalNoise with zero noise" begin
            noise = LogNormalNoise(0.1)
            v = [0.0]
            result = apply_noise(noise, 100.0, v, latent, hyper, 0.0)
            @test result ≈ 100.0 atol = 1.0e-4
        end

        @testset "LogNormalNoise multiplicative" begin
            noise = LogNormalNoise(0.5)
            v = [1.0]
            result = apply_noise(noise, 100.0, v, latent, hyper, 0.0)
            # 100 * exp(0.5 * 1) ≈ 164.87
            @test result ≈ 100.0 * exp(0.5) atol = 1.0e-4
        end

        @testset "noise param can read a hyperparameter via a function" begin
            # sigma_mult reads hyper.sigma_obs (the function form replaces the
            # old HyperParam lookup; the optimizer tunes hyper.sigma_obs)
            noise = NegBinomialNoise(
                (latent, hyper, t) -> hyper.sigma_obs,
                100.0
            )
            v = [1.0]
            test_hyper = (sigma_obs = 0.5,)
            result = apply_noise(noise, 100.0, v, NamedTuple(), test_hyper, 0.0)
            # σ = hyper.sigma_obs = 0.5 → y = 100 + sqrt(100 + 100 + 2500) ≈ 151.96
            @test result ≈ 100.0 + sqrt(100.0 + 100.0 + (0.5 * 100.0)^2) atol = 1.0e-4
        end
    end

    # ========================================================================
    # build_measurement_model - single signal
    # ========================================================================
    @testset "build_measurement_model single signal" begin
        # Create a simple single-signal layout
        core_names = (:S, :I, :R)
        obs_names = (:O_y1, :O_y2)
        latent_names = (:Rt,)
        layout = StateLayout(core_names, obs_names, latent_names; signal_names = (:y,))

        @testset "with NegBinomialNoise" begin
            noise = NegBinomialNoise(0.1, 10.0)
            result = build_measurement_model(layout, noise, _sto(layout))

            @test haskey(result, :measure)
            @test haskey(result, :n_obs)
            @test haskey(result, :n_noise)
            @test result.n_obs == 1
            @test result.n_noise == 1
        end

        @testset "with PoissonNoise pure" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, noise, _sto(layout))

            @test result.n_obs == 1
            @test result.n_noise == 1
        end

        @testset "measurement function works" begin
            noise = NegBinomialNoise(0.0, 100.0)
            result = build_measurement_model(layout, noise, _sto(layout))

            # Build state vector
            # Layout: [S, I, R, O_y1, O_y2 (accumulator), Rt]
            total_dim = layout.total_dim
            @test total_dim == 6

            x = zeros(total_dim)
            x[5] = 100.0  # accumulator: this step's incidence
            x[6] = 1.5    # Rt (latent)

            v = [0.0]  # NB now uses a single noise term
            hyperparams = NamedTuple()
            u = SVector(50.0)  # ignored: the accumulator is read directly

            y = result.measure(x, u, hyperparams, 0.0, v)

            @test y isa SVector{1}
            @test y[1] ≈ 100.0 atol = 1.0e-4  # accumulator value, no u subtraction
        end
    end

    # ========================================================================
    # build_measurement_model - multiple signals
    # ========================================================================
    @testset "build_measurement_model multiple signals" begin
        core_names = (:S, :I, :R)
        obs_names = (:O_y1, :O_y2)
        latent_names = (:Rt,)
        layout = StateLayout(core_names, obs_names, latent_names; signal_names = (:y1, :y2))

        @testset "construction with tuple of specs" begin
            obs_specs = (
                SignalObservationSpec(1, NegBinomialNoise(0.1, 10.0)),
                SignalObservationSpec(2, PoissonNoise()),
            )

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            @test result.n_obs == 2
            @test result.n_noise == 2  # 1 for NB + 1 for Poisson
        end

        @testset "measurement function with multiple signals" begin
            obs_specs = (
                SignalObservationSpec(1, NegBinomialNoise(0.0, 100.0)),
                SignalObservationSpec(2, PoissonNoise()),
            )

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            # Layout: [S, I, R, O_y1, O_y2, Rt]
            total_dim = layout.total_dim
            @test total_dim == 6

            x = zeros(total_dim)
            x[4] = 30.0  # signal 1 accumulator
            x[5] = 25.0  # signal 2 accumulator
            x[6] = 1.5

            v = [0.0, 0.0]  # 1 for signal 1 (NB), 1 for signal 2 (Poisson)
            hyperparams = NamedTuple()
            u = SVector(10.0, 5.0)  # ignored

            y = result.measure(x, u, hyperparams, 0.0, v)

            @test y isa SVector{2}
            @test y[1] ≈ 30.0 atol = 1.0e-4  # signal 1 accumulator, read directly
            @test y[2] ≈ 25.0 atol = 1.0e-4  # signal 2 accumulator, read directly
        end

        @testset "invalid signal_idx throws" begin
            obs_specs = (
                SignalObservationSpec(3, PoissonNoise()),  # signal 3 doesn't exist
            )
            @test_throws ArgumentError build_measurement_model(layout, obs_specs, _sto(layout))
        end
    end

    # ========================================================================
    # build_measurement_model - aggregated (sum all signals)
    # ========================================================================
    @testset "build_measurement_model aggregated" begin
        core_names = (:S, :I, :R)
        obs_names = (:O_child, :O_adult, :O_elderly)
        latent_names = (:Rt,)
        layout = StateLayout(
            core_names, obs_names, latent_names;
            signal_names = (:child, :adult, :elderly)
        )

        @testset "explicit aggregation with AggregatedSignalSpec" begin
            noise = NegBinomialNoise(0.0, 100.0)
            obs_specs = (AggregatedSignalSpec([1, 2, 3], noise; name = :total_hosp),)

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            @test result.n_obs == 1  # one aggregated observation
            @test result.n_noise == 1  # NB has 1 noise term
        end

        @testset "Val(:aggregated) convenience" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, (AggregatedSignalSpec(noise; name = :obs_total),), _sto(layout))

            @test result.n_obs == 1
            @test result.n_noise == 1
        end

        @testset "aggregated measurement sums signals" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, (AggregatedSignalSpec(noise; name = :obs_total),), _sto(layout))

            # Layout: [S, I, R, O_child, O_adult, O_elderly, Rt]
            total_dim = layout.total_dim
            @test total_dim == 7

            x = zeros(total_dim)
            x[4] = 10.0
            x[5] = 25.0
            x[6] = 30.0

            v = [0.0]  # 1 noise term for Poisson
            y = result.measure(x, SVector(0.0, 5.0, 0.0), NamedTuple(), 0.0, v)

            @test y isa SVector{1}
            @test y[1] ≈ 65.0 atol = 1.0e-4  # sum of accumulators (u ignored)
        end

        @testset "partial aggregation" begin
            noise = PoissonNoise()
            # Only aggregate signals 1 and 2 (child + adult)
            obs_specs = (AggregatedSignalSpec([1, 2], noise; name = :under65),)

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            total_dim = layout.total_dim
            x = zeros(total_dim)
            x[4] = 10.0
            x[5] = 25.0
            x[6] = 30.0

            v = [0.0]
            y = result.measure(x, SVector(0.0, 5.0, 0.0), NamedTuple(), 0.0, v)

            @test y[1] ≈ 35.0 atol = 1.0e-4  # only signals 1+2 (10 + 25)
        end

        @testset "invalid signal in aggregation throws" begin
            noise = PoissonNoise()
            obs_specs = (AggregatedSignalSpec([1, 4], noise),)  # signal 4 doesn't exist
            @test_throws ArgumentError build_measurement_model(layout, obs_specs, _sto(layout))
        end
    end

    # ========================================================================
    # build_measurement_model - mixed (some aggregated, some individual)
    # ========================================================================
    @testset "build_measurement_model mixed" begin
        core_names = (:S, :I, :R)
        obs_names = (:O_child, :O_adult, :O_elderly)
        latent_names = ()
        layout = StateLayout(
            core_names, obs_names, latent_names;
            signal_names = (:child, :adult, :elderly)
        )

        @testset "mixed aggregated + individual" begin
            # Aggregate child + adult, individual elderly
            noise = PoissonNoise()
            obs_specs = (
                AggregatedSignalSpec([1, 2], noise; name = :under65_hosp),
                SignalObservationSpec(3, noise; name = :over65_hosp),
            )

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            @test result.n_obs == 2
            @test result.n_noise == 2  # 1 for aggregated + 1 for individual
        end

        @testset "mixed measurement values" begin
            noise = PoissonNoise()
            obs_specs = (
                AggregatedSignalSpec([1, 2], noise; name = :under65),
                SignalObservationSpec(3, noise; name = :elderly),
            )

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            # Layout: [S, I, R, O_child, O_adult, O_elderly]
            total_dim = layout.total_dim
            @test total_dim == 6

            x = zeros(total_dim)
            x[4] = 15.0
            x[5] = 25.0
            x[6] = 40.0

            v = [0.0, 0.0]
            y = result.measure(x, SVector(0.0, 0.0, 0.0), NamedTuple(), 0.0, v)

            @test y isa SVector{2}
            @test y[1] ≈ 40.0 atol = 1.0e-4  # child + adult = 15 + 25
            @test y[2] ≈ 40.0 atol = 1.0e-4  # elderly alone
        end
    end

    # ========================================================================
    # Edge cases
    # ========================================================================
    @testset "edge cases" begin
        core_names = (:S, :I)
        obs_names = (:O_I,)
        latent_names = ()
        layout = StateLayout(core_names, obs_names, latent_names; signal_names = (:y,))

        @testset "accumulator read directly (u ignored)" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, noise, _sto(layout))

            x = zeros(layout.total_dim)
            x[3] = 50.0

            v = [0.0]
            y = result.measure(x, SVector(50.0), NamedTuple(), 0.0, v)

            @test y[1] ≈ 50.0 atol = 1.0e-4  # accumulator, not 50 - u = 0
        end

        @testset "negative accumulator clamped to zero" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, noise, _sto(layout))

            x = zeros(layout.total_dim)
            x[3] = -5.0  # whisker noise can push an accumulator negative

            v = [0.0]
            y = result.measure(x, SVector(0.0), NamedTuple(), 0.0, v)

            @test y[1] >= 0.0  # compute_true_mean clamps to zero
        end

        @testset "single-signal convenience rejects multi-signal layout" begin
            core_names = (:S, :I)
            obs_names = (:O1, :O2)
            latent_names = ()
            layout_multi = StateLayout(
                core_names, obs_names, latent_names; signal_names = (
                    :y1, :y2,
                )
            )

            noise = PoissonNoise()
            @test_throws ArgumentError build_measurement_model(layout_multi, noise, _sto(layout_multi))
        end
    end

    # ========================================================================
    # mean_modifier - ascertainment as a Real or (latent, hyper, t) function
    # ========================================================================
    @testset "mean_modifier for ascertainment" begin
        core_names = (:S, :I, :R)
        obs_names = (:O_I,)
        latent_names = (:Rt, :ascertainment)
        layout = StateLayout(core_names, obs_names, latent_names; signal_names = (:y,))

        @testset "SignalObservationSpec with literal modifier" begin
            noise = PoissonNoise()
            spec = SignalObservationSpec(1, noise; mean_modifier = 0.5)

            @test spec.mean_modifier == 0.5
        end

        @testset "SignalObservationSpec with function modifier" begin
            noise = PoissonNoise()
            modifier = (latent, hyper, t) -> hyper.detection_rate
            spec = SignalObservationSpec(1, noise; mean_modifier = modifier)

            @test spec.mean_modifier isa Function
        end

        @testset "measurement with literal modifier scales mean" begin
            noise = PoissonNoise()
            obs_specs = (SignalObservationSpec(1, noise; mean_modifier = 0.25),)  # 25% detection

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            total_dim = layout.total_dim
            @test total_dim == 6

            x = zeros(total_dim)
            x[4] = 100.0
            x[5] = 1.2
            x[6] = 0.5

            v = [0.0]
            y = result.measure(x, SVector(0.0), NamedTuple(), 0.0, v)

            @test y[1] ≈ 25.0 atol = 1.0e-4  # 100 * 0.25
        end

        @testset "measurement with function modifier reads hyperparams" begin
            noise = PoissonNoise()
            modifier = (latent, hyper, t) -> hyper.detect_rate
            obs_specs = (SignalObservationSpec(1, noise; mean_modifier = modifier),)

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            x = zeros(layout.total_dim)
            x[4] = 200.0

            v = [0.0]
            p = (detect_rate = 0.3,)  # 30% detection from hyperparams

            y = result.measure(x, SVector(0.0), p, 0.0, v)

            @test y[1] ≈ 60.0 atol = 1.0e-4  # 200 * 0.3
        end

        @testset "measurement with function modifier reads latent state" begin
            noise = PoissonNoise()
            # ascertainment is at index 2 in latent_names (:Rt, :ascertainment)
            modifier = (latent, hyper, t) -> latent.ascertainment
            obs_specs = (SignalObservationSpec(1, noise; mean_modifier = modifier),)

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            x = zeros(layout.total_dim)
            x[4] = 400.0
            x[5] = 1.5
            x[6] = 0.75

            v = [0.0]
            y = result.measure(x, SVector(0.0), NamedTuple(), 0.0, v)

            @test y[1] ≈ 300.0 atol = 1.0e-4  # 400 * 0.75
        end

        @testset "AggregatedSignalSpec with mean_modifier" begin
            core_names = (:S, :I, :R)
            obs_names = (:O_child, :O_adult)
            latent_names = ()
            layout2 = StateLayout(
                core_names, obs_names, latent_names;
                signal_names = (:child, :adult)
            )

            noise = PoissonNoise()
            obs_specs = (
                AggregatedSignalSpec([1, 2], noise; mean_modifier = 0.5, name = :total),
            )

            result = build_measurement_model(layout2, obs_specs, _sto(layout2))

            x = zeros(layout2.total_dim)
            x[4] = 60.0
            x[5] = 50.0

            v = [0.0]
            y = result.measure(x, SVector(0.0, 10.0), NamedTuple(), 0.0, v)

            @test y[1] ≈ 55.0 atol = 1.0e-4  # (60 + 50) * 0.5, u ignored
        end

        @testset "no modifier (nothing) preserves raw mean" begin
            noise = PoissonNoise()
            obs_specs = (SignalObservationSpec(1, noise),)  # no mean_modifier

            result = build_measurement_model(layout, obs_specs, _sto(layout))

            x = zeros(layout.total_dim)
            x[4] = 123.0

            v = [0.0]
            y = result.measure(x, SVector(0.0), NamedTuple(), 0.0, v)

            @test y[1] ≈ 123.0 atol = 1.0e-4  # unchanged
        end
    end
    @testset "observation baseline: additive, un-ascertained, survives index resolution" begin
        noise = NegBinomialNoise(phi = (l, h, t) -> 100.0)

        # The baseline is added AFTER the multiplicative modifier, because `ascertainment` scales
        # INFECTIONS while the baseline is a property of the signal and is not ascertained.
        plain = SignalObservationSpec(1, noise; mean_modifier = (l, h, t) -> 0.005)
        floored = SignalObservationSpec(
            1, noise; mean_modifier = (l, h, t) -> 0.005, baseline = 7.0
        )
        @test observation_mean(plain, 1000.0, (;), (;), 0.0) ≈ 5.0
        @test observation_mean(floored, 1000.0, (;), (;), 0.0) ≈ 12.0
        # At zero incidence the floor is the whole mean — the point of it. This is what keeps the
        # NB variance `mu + mu^2/phi` away from zero in a deep trough, where an unfloored model
        # becomes arbitrarily confident about a near-zero prediction.
        @test observation_mean(floored, 0.0, (;), (;), 0.0) ≈ 7.0

        # A function baseline may read hyperparameters, like the modifier.
        dynamic = SignalObservationSpec(1, noise; baseline = (l, h, t) -> h.obs_floor)
        @test observation_mean(dynamic, 10.0, (;), (obs_floor = 3.0,), 0.0) ≈ 13.0

        # Resolving "all signals" into concrete indices rebuilds the spec, and must keep its
        # modifier, baseline and name.
        aggregated = AggregatedSignalSpec(
            noise; mean_modifier = (l, h, t) -> 0.005, baseline = 7.0, name = :total
        )
        four = StateLayout((:S,), (:O_a, :O_b, :O_c, :O_d), (); signal_names = (:a, :b, :c, :d))
        resolved = resolve_observation(aggregated, four)
        @test resolved.raw.indices == [1, 2, 3, 4]
        @test resolved.baseline == 7.0
        @test resolved.name == :total
        @test observation_mean(resolved, 1000.0, (;), (;), 0.0) ≈ 12.0

        # Default is no baseline, so every existing model is untouched.
        @test SignalObservationSpec(1, noise).baseline === nothing
        @test AggregatedSignalSpec(noise).baseline === nothing
    end
end

# ============================================================================
# observation_scale / observation_baseline — the helpers the reporting paths go through so a
# time-varying modifier is evaluated at each observation's own time, never read as a constant.
# ============================================================================
@testset "observation_scale and observation_baseline" begin
    latent = (Rt = 1.2, ascertainment = 0.75)
    hyper = (detect = 0.3, phi = 40.0, ascertainment = 0.005, ascertainment_decline_rate = 0.5)
    path = AscertainmentPath(0.2, 10.0)
    modifiers = (
        nothing, 0.25, (l, h, t) -> h.detect, (l, h, t) -> l.ascertainment, path,
    )
    baselines = (nothing, 3.0, (l, h, t) -> 2.0 + t)

    @testset "mean == scale * raw + baseline for every modifier and baseline shape" begin
        for m in modifiers, b in baselines, t in (0.0, 30.0)
            spec = SignalObservationSpec(1, PoissonNoise(); mean_modifier = m, baseline = b)
            raw = 200.0
            @test observation_mean(spec, raw, latent, hyper, t) ≈
                observation_scale(spec, latent, hyper, t) * raw +
                observation_baseline(spec, latent, hyper, t)
        end
        @test observation_scale(SignalObservationSpec(1, PoissonNoise()), latent, hyper, 0.0) == 1.0
        @test observation_scale(
            SignalObservationSpec(1, PoissonNoise(); mean_modifier = 0.25), latent, hyper, 0.0
        ) == 0.25
        @test observation_scale(
            SignalObservationSpec(1, PoissonNoise(); mean_modifier = path), latent, hyper, 375.25
        ) ≈ path(hyper, 375.25)
        @test observation_baseline(
            SignalObservationSpec(1, PoissonNoise(); baseline = (l, h, t) -> 2.0 + t),
            latent, hyper, 5.0,
        ) == 7.0
        @test observation_baseline(SignalObservationSpec(1, PoissonNoise()), latent, hyper, 5.0) == 0.0
    end
end

@testset "AscertainmentPath as a mean_modifier is evaluated at the observation's time" begin
    layout = StateLayout((:S, :I, :R), (:O_I,), (:Rt,); signal_names = (:y,))
    path = AscertainmentPath(0.2, 0.0)
    specs = (SignalObservationSpec(1, PoissonNoise(); mean_modifier = path),)
    result = build_measurement_model(layout, specs, _sto(layout))
    x = zeros(layout.total_dim)
    x[4] = 1000.0
    hyper = (ascertainment = 0.005, ascertainment_decline_rate = 0.3)
    at_start = result.measure(x, SVector(0.0), hyper, 0.0, [0.0])[1]
    a_year_on = result.measure(x, SVector(0.0), hyper, 365.25, [0.0])[1]
    @test at_start ≈ 1000.0 * 0.005 atol = 1.0e-6
    @test a_year_on ≈ 1000.0 * path(hyper, 365.25) atol = 1.0e-6
    @test a_year_on < at_start
    # A zero rate is the plain constant, so an existing `(l, h, t) -> h.ascertainment` model and
    # the path agree exactly.
    flat = (ascertainment = 0.005, ascertainment_decline_rate = 0.0)
    @test result.measure(x, SVector(0.0), flat, 365.25, [0.0])[1] ==
        result.measure(x, SVector(0.0), flat, 0.0, [0.0])[1]
end

# A `WeightedStateSpec` observes a linear functional of occupancy (a shedding chain's stages),
# resolved by name against the layout, beside an ordinary accumulator signal.
@testset "WeightedStateSpec observes a weighted occupancy" begin
    layout = StateLayout(
        (:S, :I), (:O_1, :W_1, :W_2, :W_3), (:Rt,);
        signal_names = (:reports, :wastewater), accumulator_indices = (1, 4),
    )
    spec = WeightedStateSpec([:W_1, :W_2], [0.25, 0.75], LogNormalNoise(0.5); mean_modifier = 2.0, name = :wastewater)
    @test n_noise_terms(spec) == 1
    resolved = resolve_observation(spec, layout)
    @test resolved.raw.indices == [4, 5]
    @test resolved.name == :wastewater
    x = [100.0, 10.0, 3.0, 8.0, -2.0, 1.0, 0.0]
    # A negative occupancy is floored at zero, as an accumulator is.
    @test ConfigurableEpi.compute_true_mean(resolved, x, layout.accumulator_indices) == 0.25 * 8.0
    @test observation_mean(resolved, 2.0, (Rt = 1.0,), (;), 0.0) == 4.0
    @test_throws ArgumentError WeightedStateSpec([:W_1, :W_1], [1.0, 1.0], LogNormalNoise(0.5))
    @test_throws DimensionMismatch WeightedStateSpec([:W_1], [1.0, 1.0], LogNormalNoise(0.5))
    @test_throws ArgumentError WeightedStateSpec([:W_1], [-1.0], LogNormalNoise(0.5))
    @test_throws ArgumentError WeightedStateSpec(Symbol[], Float64[], LogNormalNoise(0.5))
    @test_throws ArgumentError resolve_observation(WeightedStateSpec([:nope], [1.0], LogNormalNoise(0.5)), layout)
    # An unresolved spec cannot be read: the measurement builders resolve every spec first.
    @test_throws ArgumentError ConfigurableEpi.compute_true_mean(spec, x, layout.accumulator_indices)

    specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 50.0); name = :reports), spec)
    @test observation_names(build_observation_schema(specs)) == (:reports, :wastewater)
    measure, n_obs, n_noise = build_measurement_model(layout, specs, _sto(layout))
    @test (n_obs, n_noise) == (2, 2)
    y = measure(x, nothing, (;), 0.0, zeros(2))
    @test y[1] ≈ 3.0                                  # the count accumulator
    @test y[2] ≈ 4.0                                  # 2 × (0.25 × 8 + 0.75 × 0)
end

# A share of incidence scaled to counts by a known per-slot total: the typed specimens of one
# subtype out of all typed that week.
@testset "ShareSignalSpec and SlotSeries" begin
    layout = StateLayout((:S, :I), (:O_a, :O_b, :O_c), (:Rt,); signal_names = (:a, :b, :c))
    noise = NegBinomialNoise(phi = 50.0)
    acc = layout.accumulator_indices
    x = [100.0, 10.0, 6.0, 3.0, 1.0, 0.0]   # S, I, three accumulators, one latent

    share = ShareSignalSpec([1], [1, 2, 3], noise; name = :share_a)
    @test share.name == :share_a
    @test n_noise_terms(share) == 1
    true_mean(spec, x) = ConfigurableEpi.compute_true_mean(spec, x, acc)
    @test true_mean(share, x) ≈ 0.6
    @test true_mean(ShareSignalSpec([1, 2], [1, 2, 3], noise), x) ≈ 0.9
    @test true_mean(ShareSignalSpec([2], [2, 3], noise), x) ≈ 0.75
    # Negative accumulators count as empty, and an empty denominator gives a zero share.
    @test true_mean(share, [100.0, 10.0, -2.0, 3.0, 1.0, 0.0]) == 0.0
    @test true_mean(share, [100.0, 10.0, 0.0, 0.0, 0.0, 0.0]) == 0.0

    @test_throws ArgumentError ShareSignalSpec(Int[], [1, 2], noise)
    @test_throws ArgumentError ShareSignalSpec([1, 1], [1, 2], noise)
    @test_throws ArgumentError ShareSignalSpec([3], [1, 2], noise)
    @test_throws ArgumentError ShareSignalSpec([0], [0, 1], noise)
    @test_throws ArgumentError resolve_observation(ShareSignalSpec([1], [1, 4], noise), layout)
    @test resolve_observation(share, layout).raw === share.raw

    totals = SlotSeries([200.0, 50.0, 0.0]; step = 7.0)
    @test totals(nothing, nothing, 0.0) == 200.0
    @test totals(nothing, nothing, 7.0) == 50.0
    @test totals(nothing, nothing, 14.0) == 0.0
    # Beyond the series (a forecast horizon) the last value holds; before it, the first.
    @test totals(nothing, nothing, 70.0) == 0.0
    @test totals(nothing, nothing, -7.0) == 200.0
    # On the grid up to rounding (0.7 / 0.1 is just under 7), and an error between slots.
    @test SlotSeries(collect(0.0:9.0); step = 0.1)(nothing, nothing, 0.7) == 7.0
    @test_throws ArgumentError totals(nothing, nothing, 3.5)
    @test_throws ArgumentError totals(nothing, nothing, 6.0)
    @test_throws ArgumentError SlotSeries(Float64[])
    @test_throws ArgumentError SlotSeries([1.0]; step = 0.0)
    @test_throws ArgumentError SlotSeries([1.0, -1.0])
    @test_throws ArgumentError SlotSeries([1.0, NaN])

    typed = ShareSignalSpec([1], [1, 2, 3], noise; mean_modifier = totals, name = :typed_a)
    @test observation_mean(typed, 0.6, nothing, nothing, 0.0) ≈ 120.0
    @test observation_mean(typed, 0.6, nothing, nothing, 7.0) ≈ 30.0

    specs = (
        SignalObservationSpec(1, noise; name = :count_a), typed,
        ShareSignalSpec([2], [1, 2, 3], noise; mean_modifier = totals, name = :typed_b),
    )
    @test observation_names(build_observation_schema(specs)) == (:count_a, :typed_a, :typed_b)
    measure, n_obs, n_noise = build_measurement_model(layout, specs, _sto(layout))
    @test (n_obs, n_noise) == (3, 3)
    @test collect(measure(x, nothing, (;), 0.0, zeros(3))) ≈ [6.0, 120.0, 60.0]
    @test collect(measure(x, nothing, (;), 7.0, zeros(3))) ≈ [6.0, 30.0, 15.0]
    @test isfinite(build_measurement_logpdf(layout, specs, _sto(layout))(x, nothing, [6.0, 120.0, 60.0], (;), 0.0))
end

# An observation that reads no compartment, with its mean scaled linearly by a latent.
@testset "LatentObservationSpec and LinearScale" begin
    layout = StateLayout((:S, :I), (:O_1,), (:scale,))
    stochastic = _sto(layout)
    scaled = LinearScale(:scale, (l, h, t) -> h.base; link = exp)
    @test scaled((scale = log(2.0),), (base = 10.0,), 0.0) ≈ 20.0
    @test LinearScale(:scale, 3.0)((scale = 2.0,), (;), 0.0) == 6.0

    spec = LatentObservationSpec(scaled, LogNormalNoise(0.2); name = :clade)
    @test spec.raw isa ConfigurableEpi.LatentOnly && spec.baseline === nothing
    x = [100.0, 10.0, 4.0, log(2.0)]
    @test ConfigurableEpi.compute_true_mean(spec, x, layout.accumulator_indices) == 1.0
    measure, n_obs, n_noise = build_measurement_model(layout, (spec,), stochastic)
    @test (n_obs, n_noise) == (1, 1)
    @test only(measure(x, nothing, (base = 10.0,), 0.0, zeros(1))) ≈ 20.0
    g = build_measurement_logpdf(layout, (spec,), stochastic)
    @test g(x, nothing, [18.0], (base = 10.0,), 0.0) ≈
        observation_logpdf(LogNormalNoise(0.2), 18.0, 20.0, stochastic.extract(x), (base = 10.0,), 0.0)
end
