using Test
using ConfigurableEpi
using StaticArrays
using LabelledArrays

@testset "Measurement Model" begin

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
            @test spec.signal_idx == 1
            @test spec.noise_spec === noise
            @test spec.name == :obs_1  # default name
        end

        @testset "custom name" begin
            noise = PoissonNoise()
            spec = SignalObservationSpec(2, noise; name = :hospitalizations)
            @test spec.signal_idx == 2
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
            @test spec.signal_indices == [1, 2, 3]
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
            @test isempty(spec.signal_indices)  # resolved at build time
            @test spec.name == :obs_total
            @test spec.is_all_signals == true
        end

        @testset "explicit construction sets is_all_signals false" begin
            noise = PoissonNoise()
            spec = AggregatedSignalSpec([1, 2], noise; name = :explicit)
            @test spec.is_all_signals == false
        end

        @testset "invalid indices throw" begin
            noise = PoissonNoise()
            # Explicit empty (not "all signals") should throw via inner constructor
            @test_throws ArgumentError AggregatedSignalSpec(
                Int[], noise, nothing, nothing, :x, false
            )
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
            result = build_measurement_model(layout, noise)

            @test haskey(result, :measure)
            @test haskey(result, :n_obs)
            @test haskey(result, :n_noise)
            @test result.n_obs == 1
            @test result.n_noise == 1
        end

        @testset "with PoissonNoise pure" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, noise)

            @test result.n_obs == 1
            @test result.n_noise == 1
        end

        @testset "measurement function works" begin
            noise = NegBinomialNoise(0.0, 100.0)
            result = build_measurement_model(layout, noise)

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

            result = build_measurement_model(layout, obs_specs)

            @test result.n_obs == 2
            @test result.n_noise == 2  # 1 for NB + 1 for Poisson
        end

        @testset "measurement function with multiple signals" begin
            obs_specs = (
                SignalObservationSpec(1, NegBinomialNoise(0.0, 100.0)),
                SignalObservationSpec(2, PoissonNoise()),
            )

            result = build_measurement_model(layout, obs_specs)

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
            @test_throws ArgumentError build_measurement_model(layout, obs_specs)
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

            result = build_measurement_model(layout, obs_specs)

            @test result.n_obs == 1  # one aggregated observation
            @test result.n_noise == 1  # NB has 1 noise term
        end

        @testset "Val(:aggregated) convenience" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, noise, Val(:aggregated))

            @test result.n_obs == 1
            @test result.n_noise == 1
        end

        @testset "aggregated measurement sums signals" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, noise, Val(:aggregated))

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

            result = build_measurement_model(layout, obs_specs)

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
            @test_throws ArgumentError build_measurement_model(layout, obs_specs)
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

            result = build_measurement_model(layout, obs_specs)

            @test result.n_obs == 2
            @test result.n_noise == 2  # 1 for aggregated + 1 for individual
        end

        @testset "mixed measurement values" begin
            noise = PoissonNoise()
            obs_specs = (
                AggregatedSignalSpec([1, 2], noise; name = :under65),
                SignalObservationSpec(3, noise; name = :elderly),
            )

            result = build_measurement_model(layout, obs_specs)

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
            result = build_measurement_model(layout, noise)

            x = zeros(layout.total_dim)
            x[3] = 50.0

            v = [0.0]
            y = result.measure(x, SVector(50.0), NamedTuple(), 0.0, v)

            @test y[1] ≈ 50.0 atol = 1.0e-4  # accumulator, not 50 - u = 0
        end

        @testset "negative accumulator clamped to zero" begin
            noise = PoissonNoise()
            result = build_measurement_model(layout, noise)

            x = zeros(layout.total_dim)
            x[3] = -5.0  # whisker noise can push an accumulator negative

            v = [0.0]
            y = result.measure(x, SVector(0.0), NamedTuple(), 0.0, v)

            @test y[1] >= 0.0  # compute_true_mean clamps to zero
        end

        @testset "single-signal convenience rejects multi-signal layout" begin
            core_names = (:S, :I)
            obs_names = (:O,)
            latent_names = ()
            layout_multi = StateLayout(
                core_names, obs_names, latent_names; signal_names = (
                    :y1, :y2,
                )
            )

            noise = PoissonNoise()
            @test_throws ArgumentError build_measurement_model(layout_multi, noise)
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

            result = build_measurement_model(layout, obs_specs)

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

            result = build_measurement_model(layout, obs_specs)

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

            result = build_measurement_model(layout, obs_specs)

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

            result = build_measurement_model(layout2, obs_specs)

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

            result = build_measurement_model(layout, obs_specs)

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

        # REGRESSION. `resolve_signal_indices` rebuilds an `AggregatedSignalSpec` to turn "all
        # signals" into concrete indices. That rebuild used to be duplicated verbatim at three call
        # sites, and adding `baseline` dropped it at all three — silently, and only on the
        # aggregated path, which is the one `two_strain_escape` uses.
        aggregated = AggregatedSignalSpec(
            noise; mean_modifier = (l, h, t) -> 0.005, baseline = 7.0, name = :total
        )
        resolved = ConfigurableEpi.resolve_signal_indices(aggregated, 4)
        @test resolved.signal_indices == [1, 2, 3, 4]
        @test resolved.baseline == 7.0
        @test resolved.name == :total
        @test observation_mean(resolved, 1000.0, (;), (;), 0.0) ≈ 12.0

        # Default is no baseline, so every existing model is untouched.
        @test SignalObservationSpec(1, noise).baseline === nothing
        @test AggregatedSignalSpec(noise).baseline === nothing
    end
end

# ============================================================================
# observation_scale / observation_baseline / observation_gaussian_moments — the helpers the
# reporting paths (UKF fitted means and forecast quantiles, EnKF fitted means) go through so a
# time-varying modifier is evaluated at each observation's own time, never read as a constant.
# ============================================================================
@testset "observation_scale, observation_baseline and the Gaussian forecast moments" begin
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

    @testset "Gaussian moments of a NegBinomial signal, at the requested time" begin
        nb = SignalObservationSpec(
            1, NegBinomialNoise(sigma_mult = 0.1, phi = (l, h, t) -> h.phi); mean_modifier = path
        )
        raw_mean, raw_var = 150.0, 900.0
        t = 375.25
        m = observation_gaussian_moments(nb, raw_mean, raw_var, latent, hyper, t)
        scale = path(hyper, t)
        mu = scale * raw_mean
        @test m.mean ≈ mu
        @test m.var ≈ scale^2 * raw_var + mu + mu^2 / 40.0 + (0.1 * mu)^2
        # The path declines, so the same accumulator maps to a smaller count later on.
        @test observation_gaussian_moments(nb, raw_mean, raw_var, latent, hyper, 0.0).mean > m.mean
        # A negative accumulator mean is floored at zero, as `compute_true_mean` floors the state.
        @test observation_gaussian_moments(nb, -5.0, raw_var, latent, hyper, 0.0).mean == 0.0

        # With a constant modifier and no multiplicative noise this IS the historical
        # `ascertainment·accumulator ± NB` formula, bit for bit.
        constant = SignalObservationSpec(
            1, NegBinomialNoise(phi = (l, h, t) -> h.phi);
            mean_modifier = (l, h, t) -> h.ascertainment,
        )
        mc = observation_gaussian_moments(constant, raw_mean, raw_var, latent, hyper, 0.0)
        @test mc.mean == hyper.ascertainment * raw_mean
        @test mc.var == hyper.ascertainment^2 * raw_var + mc.mean + mc.mean^2 / hyper.phi

        @test_throws ArgumentError observation_gaussian_moments(
            SignalObservationSpec(1, PoissonNoise()), raw_mean, raw_var, latent, hyper, 0.0
        )
    end
end

@testset "AscertainmentPath as a mean_modifier is evaluated at the observation's time" begin
    layout = StateLayout((:S, :I, :R), (:O_I,), (:Rt,); signal_names = (:y,))
    path = AscertainmentPath(0.2, 0.0)
    specs = (SignalObservationSpec(1, PoissonNoise(); mean_modifier = path),)
    result = build_measurement_model(layout, specs)
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
