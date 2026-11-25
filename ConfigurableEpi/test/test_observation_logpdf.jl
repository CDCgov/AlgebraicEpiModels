using Test
using ConfigurableEpi
using Distributions: Poisson, NegativeBinomial, LogNormal, logpdf
using Statistics: mean, var
import Random

# The PF observation methods reuse the same noise specs as the UKF, but evaluate
# / sample the TRUE distribution (exact discrete counts) rather than a Gaussian
# approximation. We check them against Distributions.jl directly.
@testset "Observation log-likelihood + sampling" begin
    latent = NamedTuple()
    hyper = NamedTuple()

    @testset "observation_logpdf matches Distributions.jl" begin
        @testset "NegBinomial" begin
            for (μ, φ, y) in ((40.0, 50.0, 42), (10.0, 5.0, 7), (100.0, 200.0, 95))
                spec = NegBinomialNoise(0.0, φ)
                expected = logpdf(NegativeBinomial(φ, φ / (φ + μ)), y)
                @test observation_logpdf(spec, y, μ, latent, hyper, 0.0) ≈ expected
            end
            # mean μ and variance μ + μ²/φ for the chosen parameterization
            μ, φ = 40.0, 50.0
            d = NegativeBinomial(φ, φ / (φ + μ))
            @test mean(d) ≈ μ
            @test var(d) ≈ μ + μ^2 / φ
        end

        @testset "Poisson" begin
            for (μ, y) in ((10.0, 8), (50.0, 55), (3.0, 0))
                spec = PoissonNoise()
                @test observation_logpdf(spec, y, μ, latent, hyper, 0.0) ≈ logpdf(Poisson(μ), y)
            end
        end

        @testset "LogNormal" begin
            μ, σ, y = 100.0, 0.2, 87.0
            spec = LogNormalNoise(σ)
            @test observation_logpdf(spec, y, μ, latent, hyper, 0.0) ≈
                logpdf(LogNormal(log(μ), σ), y)
        end

        @testset "noise params resolve via (latent, hyper, t) functions" begin
            spec = NegBinomialNoise(0.0, (l, h, t) -> h.phi)
            μ, y = 30.0, 33
            @test observation_logpdf(spec, y, μ, NamedTuple(), (phi = 25.0,), 0.0) ≈
                logpdf(NegativeBinomial(25.0, 25.0 / (25.0 + μ)), y)
        end
    end

    @testset "sample_observation recovers the right moments" begin
        rng = Random.MersenneTwister(20260617)
        n = 50_000  # tolerances are loose (2–5%); 50k is plenty and keeps the suite fast

        @testset "NegBinomial" begin
            μ, φ = 50.0, 20.0
            s = [sample_observation(NegBinomialNoise(0.0, φ), μ, latent, hyper, 0.0, rng) for _ in 1:n]
            @test mean(s) ≈ μ rtol = 0.02
            @test var(s) ≈ μ + μ^2 / φ rtol = 0.05
            @test all(>=(0.0), s)
        end

        @testset "Poisson" begin
            μ = 25.0
            s = [sample_observation(PoissonNoise(), μ, latent, hyper, 0.0, rng) for _ in 1:n]
            @test mean(s) ≈ μ rtol = 0.02
            @test var(s) ≈ μ rtol = 0.05
        end
    end

    @testset "NegBinomialNoise with sigma_mult: simulated, never scored" begin
        # The (σμ)² term is drawn as a mean-one log-normal perturbation of the mean; the mixture
        # has no closed-form density, so the PF refuses to weight with it. With σ = 0 no extra
        # randomness is consumed.
        rng = Random.MersenneTwister(20261030)
        n = 50_000
        μ, φ, σ = 50.0, 20.0, 0.1
        spec = NegBinomialNoise(σ, φ)
        s = [sample_observation(spec, μ, latent, hyper, 0.0, rng) for _ in 1:n]
        @test mean(s) ≈ μ rtol = 0.02
        @test var(s) ≈ μ + μ^2 / φ + (σ * μ)^2 rtol = 0.05
        @test var(s) > 1.05 * (μ + μ^2 / φ)
        # The log-normal spread is calibrated so the variance is exact, not first order in σ.
        wide = NegBinomialNoise(1.0, φ)
        w = [sample_observation(wide, μ, latent, hyper, 0.0, rng) for _ in 1:200_000]
        @test mean(w) ≈ μ rtol = 0.02
        @test var(w) ≈ μ + μ^2 / φ + μ^2 rtol = 0.05
        @test_throws ArgumentError observation_logpdf(spec, 50, μ, latent, hyper, 0.0)
        resolved = NegBinomialNoise((l, h, t) -> h.sigma, (l, h, t) -> h.phi)
        @test_throws ArgumentError observation_logpdf(resolved, 50, μ, latent, (phi = φ, sigma = σ), 0.0)
        @test observation_logpdf(resolved, 50, μ, latent, (phi = φ, sigma = 0.0), 0.0) ≈ logpdf(NegativeBinomial(φ, φ / (φ + μ)), 50)
        a, b = Random.MersenneTwister(1), Random.MersenneTwister(1)
        @test sample_observation(NegBinomialNoise(0.0, φ), μ, latent, hyper, 0.0, a) ==
            sample_observation(NegBinomialNoise(phi = φ), μ, latent, hyper, 0.0, b)
        @test rand(a) == rand(b)
    end

    @testset "PoissonNoise with sigma_mult is rejected by the PF" begin
        # Over-dispersed Poisson has no closed-form exact density/sampler — the PF
        # rejects it rather than silently scoring/sampling a pure Poisson.
        @test_throws ArgumentError observation_logpdf(PoissonNoise(0.1), 5, 5.0, latent, hyper, 0.0)
        @test_throws ArgumentError sample_observation(
            PoissonNoise(0.1), 5.0, latent, hyper, 0.0, Random.MersenneTwister(0)
        )
        # pure Poisson still works
        @test observation_logpdf(PoissonNoise(), 5, 5.0, latent, hyper, 0.0) ≈ logpdf(Poisson(5.0), 5)
    end

    # ========================================================================
    # build_measurement_logpdf — sums per-signal logpdf, honors mean_modifier
    # ========================================================================
    @testset "build_measurement_logpdf" begin
        rwspec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = 0.1)

        @testset "single signal matches a direct logpdf" begin
            layout = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (:Rt,); signal_names = (:y,))
            ld = build_stochastic_update(layout, (rwspec,))
            obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 50.0); name = :y),)
            g = build_measurement_logpdf(layout, obs_specs, ld)

            x = zeros(layout.total_dim)
            x[5] = 40.0  # accumulator for signal 1
            ll = g(x, nothing, [42.0], NamedTuple(), 0.0)
            @test ll ≈ logpdf(NegativeBinomial(50.0, 50.0 / (50.0 + 40.0)), 42)
        end

        @testset "two signals sum, mixed Poisson + NegBinomial" begin
            layout = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (:Rt,); signal_names = (:y1, :y2))
            ld = build_stochastic_update(layout, (rwspec,))
            obs_specs = (
                SignalObservationSpec(1, PoissonNoise()),
                SignalObservationSpec(2, NegBinomialNoise(phi = 30.0)),
            )
            g = build_measurement_logpdf(layout, obs_specs, ld)

            x = zeros(layout.total_dim)
            x[4] = 10.0  # signal 1 accumulator
            x[5] = 20.0  # signal 2 accumulator
            ll = g(x, nothing, [12.0, 18.0], NamedTuple(), 0.0)
            @test ll ≈ logpdf(Poisson(10.0), 12) +
                logpdf(NegativeBinomial(30.0, 30.0 / (30.0 + 20.0)), 18)
            # An absent (NaN) component contributes nothing.
            @test g(x, nothing, [12.0, NaN], NamedTuple(), 0.0) ≈ logpdf(Poisson(10.0), 12)
            @test g(x, nothing, [NaN, NaN], NamedTuple(), 0.0) == 0.0
        end

        @testset "mean_modifier scales the mean" begin
            layout = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (:Rt,); signal_names = (:y,))
            ld = build_stochastic_update(layout, (rwspec,))
            obs_specs = (SignalObservationSpec(1, PoissonNoise(); mean_modifier = 0.5),)
            g = build_measurement_logpdf(layout, obs_specs, ld)

            x = zeros(layout.total_dim)
            x[5] = 100.0
            ll = g(x, nothing, [40.0], NamedTuple(), 0.0)
            @test ll ≈ logpdf(Poisson(50.0), 40)  # mean = 100 * 0.5
        end

        @testset "single-spec convenience overload" begin
            layout = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (:Rt,); signal_names = (:y,))
            ld = build_stochastic_update(layout, (rwspec,))
            g = build_measurement_logpdf(layout, PoissonNoise(), ld)

            x = zeros(layout.total_dim)
            x[5] = 7.0
            @test g(x, nothing, [9.0], NamedTuple(), 0.0) ≈ logpdf(Poisson(7.0), 9)
        end

        @testset "type stability" begin
            layout = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (:Rt,); signal_names = (:y,))
            ld = build_stochastic_update(layout, (rwspec,))
            g = build_measurement_logpdf(layout, (SignalObservationSpec(1, PoissonNoise()),), ld)
            x = zeros(layout.total_dim)
            x[5] = 7.0
            @test (@inferred g(x, nothing, [9.0], NamedTuple(), 0.0)) isa Float64
        end
    end
end
