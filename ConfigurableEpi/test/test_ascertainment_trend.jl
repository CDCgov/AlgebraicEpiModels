using Test
using ConfigurableEpi
using Statistics: mean, std
import Random

# The trend ascertainment (src/ascertainment_trend.jl): log-ascertainment as an integrated Brownian
# motion, a noise-free level advanced by a random-walk decline rate. These tests pin the driver
# pair's arithmetic and, above all, that the one hyperparameter MEANS what its name says.
@testset "Trend ascertainment (integrated Brownian motion)" begin
    rate_prior = unconstrained_gaussian(ASCERTAINMENT_TREND_RATE, 0.1943, 0.1181)

    @testset "IntegratedParamSpec: validation and the noise-free step" begin
        level_prior = positive_gaussian(:level, 0.005, 0.0005)
        spec = IntegratedParamSpec(:level; init = level_prior, rate = :slope, per_days = -365.25)
        @test carries_state(spec)
        @test spec isa ProcessParamSpec
        @test_throws ArgumentError IntegratedParamSpec(:level; init = level_prior, rate = :level)
        @test_throws ArgumentError IntegratedParamSpec(
            :level; init = level_prior, rate = :slope, per_days = 0.0,
        )
        @test_throws ArgumentError IntegratedParamSpec(
            :level; init = level_prior, rate = :slope, per_days = Inf,
        )
        # The init prior carries the bijector, so its name must be the latent's.
        @test_throws ArgumentError IntegratedParamSpec(:other; init = level_prior, rate = :slope)

        step = ConfigurableEpi.update_single
        params = (slope = 0.2,)
        @test step(spec, 1.0, 0.0, params, 1.0) ≈ 1.0 - 0.2 / 365.25
        # Noise-free: the unit-noise draw is ignored.
        @test step(spec, 1.0, 0.0, params, 1.0) == step(spec, 1.0, 7.5, params, 1.0)
        # dt-invariant under a constant rate: seven daily steps are one weekly step.
        daily = foldl((u, _) -> step(spec, u, 0.0, params, 1.0), 1:7; init = 1.0)
        @test daily ≈ step(spec, 1.0, 0.0, params, 7.0)
        # A positive `per_days` integrates the rate with its own sign.
        rising = IntegratedParamSpec(:level; init = level_prior, rate = :slope)
        @test step(rising, 1.0, 0.0, params, 2.0) ≈ 1.4
    end

    specs = ascertainment_trend_specs(; rate_prior, level0 = 0.005)
    layout = StateLayout(
        (:S, :I), (:O_I_1,), (ASCERTAINMENT_TREND_RATE, ASCERTAINMENT_TREND_LEVEL),
    )
    stochastic = build_stochastic_update(layout, specs)
    hyper = NamedTuple{(ASCERTAINMENT_TREND_WANDER,)}((0.1,))
    latent0 = NamedTuple{(ASCERTAINMENT_TREND_RATE, ASCERTAINMENT_TREND_LEVEL)}((0.2, 0.005))
    x0 = vcat([990.0, 10.0, 0.0], collect(stochastic.to_unconstrained(latent0)))

    @testset "the pair is wired through the layout and the stochastic update" begin
        @test layout.latent_names == (ASCERTAINMENT_TREND_RATE, ASCERTAINMENT_TREND_LEVEL)
        @test stochastic.extract(x0)[ASCERTAINMENT_TREND_LEVEL] ≈ 0.005
        @test x0[end] ≈ log(0.005)                  # the level's slot IS log-ascertainment

        # No noise: the rate is static and log-ascertainment falls along a straight line.
        still = stochastic.advance(x0, hyper, zeros(2), nothing, 0.0, 7.0)
        @test stochastic.extract(still)[ASCERTAINMENT_TREND_RATE] == 0.2
        @test still[end] ≈ log(0.005) - 0.2 * 7 / 365.25
        @test still[1:3] == x0[1:3]                 # compartments untouched by the drivers

        # With noise: the RATE moves by sigma_rate * sqrt(dt) * w; the LEVEL ignores its own draw
        # and integrates the PRE-advance rate (every driver reads the state at the step's start).
        kicked = stochastic.advance(x0, hyper, [1.0, 123.0], nothing, 0.0, 7.0)
        @test stochastic.extract(kicked)[ASCERTAINMENT_TREND_RATE] ≈
            0.2 + ascertainment_trend_sigma_rate(0.1) * sqrt(7.0)
        @test kicked[end] == still[end]

        # The wander is read from the hyperparameters on every step, so Liu-West can carry it.
        stiffer = stochastic.advance(
            x0, NamedTuple{(ASCERTAINMENT_TREND_WANDER,)}((0.05,)), [1.0, 0.0], nothing, 0.0, 7.0,
        )
        @test stochastic.extract(stiffer)[ASCERTAINMENT_TREND_RATE] - 0.2 ≈
            (stochastic.extract(kicked)[ASCERTAINMENT_TREND_RATE] - 0.2) / 2
    end

    @testset "the wander is the 1-sd departure from a straight line after one year" begin
        rng = Random.MersenneTwister(11)
        n_paths, n_steps = 3000, 365
        dt = 365.25 / n_steps
        straight_line = log(0.005) - 0.2               # one year at the starting rate
        departures = map(1:n_paths) do _
            x = x0
            for _ in 1:n_steps
                x = stochastic.advance(x, hyper, randn(rng, 2), nothing, 0.0, dt)
            end
            x[end] - straight_line
        end
        @test std(departures) ≈ DEFAULT_ASCERTAINMENT_TREND_WANDER rtol = 0.06
        @test mean(departures) ≈ 0.0 atol = 0.008      # unbiased: the line is the conditional mean
        @test ascertainment_trend_sigma_rate(0.1) ≈ 0.1 * sqrt(3 / 365.25)
    end

    @testset "observation function, seed, and the weekday wrapper" begin
        latent = NamedTuple{(:Rt, ASCERTAINMENT_TREND_LEVEL)}((1.0, 0.004))
        @test TrendAscertainment()(latent, (;), 3.0) == 0.004

        seed = TrendAscertainmentSeed(0.005, 0.2)
        @test seed(0.0) == 0.005
        @test seed(-365.25) ≈ 0.005 * exp(0.2)         # higher before the window: it is a decline
        @test seed(365.25) ≈ 0.005 * exp(-0.2)

        @test_throws ArgumentError ascertainment_trend_specs(; rate_prior, level0 = 0.0)
        @test_throws ArgumentError ascertainment_trend_specs(;
            rate_prior = unconstrained_gaussian(:not_the_rate, 0.2, 0.1), level0 = 0.005,
        )

        # The weekday modifier passes the latent THROUGH to the inner ascertainment function. The
        # declining path ignores it, so that model is unchanged; the trend model needs it.
        doubled = ConfigurableEpi.DayOfWeekModifier(TrendAscertainment(), 1, ntuple(_ -> 2.0, 7))
        @test doubled(latent, (;), 0.0) == 0.008
    end
end
