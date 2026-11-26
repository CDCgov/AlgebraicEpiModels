using Test
using ConfigurableEpi

# Two groups with their own contact rates, a shared recovery and one accumulator counting
# infections. `hyper.infect = false` is the same field with the infection loop open.
function _warm_vf!(du, u, p, t)
    hyper, latent = p
    on = hyper.infect ? 1.0 : 0.0
    force_a = hyper.beta * latent.Rt * (3.0 * u[:Ia] + 1.0 * u[:Ib]) / 1.0e6 + hyper.importation
    force_b = hyper.beta * latent.Rt * (1.0 * u[:Ia] + 0.5 * u[:Ib]) / 1.0e6 + hyper.importation
    into_a, into_b = on * force_a * u[:Sa], on * force_b * u[:Sb]
    du[:Sa] = -into_a
    du[:Sb] = -into_b
    du[:Ia] = into_a - 0.25 * u[:Ia]
    du[:Ib] = into_b - 0.25 * u[:Ib]
    du[:R] = 0.25 * (u[:Ia] + u[:Ib])
    du[:O_I_1] = into_a + into_b
    return nothing
end

@testset "warm start" begin
    @testset "growth rate of a signal" begin
        @test log_growth_rate([100.0, 80.0, 64.0, 51.2]; interval = 7.0) ≈ log(0.8) / 7
        @test log_growth_rate([5.0]) == 0.0
        @test log_growth_rate([5.0, 0.0, 3.0]) == 0.0
    end

    @testset "the infections a signal implies" begin
        ends, counts = [-21.0, -14.0, -7.0], [400.0, 320.0, 256.0]
        per_infection, split = [0.01, 0.02], [0.75, 0.25]
        incidence = signal_incidence(ends, counts, per_infection, split; delay = 7.0, growth_points = 3)
        @test incidence.edges == [-35.0, -28.0, -21.0, -14.0, -7.0, 0.0]
        # Each reported interval is reproduced at the observation level, a delay earlier.
        @test [sum(per_infection .* incidence.rates[:, k]) * 7 for k in 1:3] ≈ counts
        @test incidence.rates[1, 1] / incidence.rates[2, 1] ≈ 3.0
        # Infections not yet reported continue the signal's growth.
        @test incidence.rates[:, 4] ≈ 0.8 .* incidence.rates[:, 3]
        @test incidence.rates[:, 5] ≈ 0.64 .* incidence.rates[:, 3]
        held = signal_incidence(ends, counts, per_infection, split; delay = 7.0, growth_points = 0)
        @test held.rates[:, 5] == held.rates[:, 3]
        by_week = signal_incidence(ends, counts, per_infection, [0.75 0.5 0.25; 0.25 0.5 0.75]; delay = 7.0)
        @test [sum(per_infection .* by_week.rates[:, k]) * 7 for k in 1:3] ≈ counts
        @test by_week.rates[1, 3] / by_week.rates[2, 3] ≈ 1 / 3
        @test_throws ArgumentError signal_incidence([-21.0, -10.0], [1.0, 1.0], per_infection, split; delay = 7.0)
        @test_throws ArgumentError signal_incidence(ends, counts, [0.0, 0.01], [1.0, 0.0]; delay = 7.0)
        @test_throws DimensionMismatch signal_incidence(ends, counts, per_infection, [1.0]; delay = 7.0)

        closing = closing_incidence(incidence)
        @test closing.growth ≈ log(0.8) / 7
        @test closing.rate ≈ sum(incidence.rates[:, 5]) * 0.8^0.5
        @test closing_incidence(incidence; per_infection).rate * 7 ≈ 256.0 * 0.8^2.5
        @test closing_incidence(held).growth == 0.0

        history = BinnedIncidence(collect(-70.0:7.0:-7.0), fill(5.0, 2, 9))
        spliced = splice_incidence(history, incidence)
        @test spliced.edges == collect(-70.0:7.0:0.0)
        @test spliced.rates[:, 1:5] == fill(5.0, 2, 5)
        @test spliced.rates[:, 6:10] == incidence.rates
        @test splice_incidence(BinnedIncidence([-30.0, -20.0], fill(1.0, 2, 1)), incidence) === incidence

        # The same history pushed back through the observation level closes on the signal.
        closure = history_closure(incidence, per_infection, ends, counts; delay = 7.0)
        @test closure.implied ≈ counts
        @test closure.ratio ≈ 1.0
        @test history_closure(incidence, per_infection ./ 2, ends, counts; delay = 7.0).ratio ≈ 2.0
    end

    layout = StateLayout((:Sa, :Sb, :Ia, :Ib, :R), (:O_I_1,), (:Rt,))
    routes = InfectionRoutes([:a, :b], [1, 2], [[3, 6], [4, 6]], [1, 2], [1.0, 1.0])
    hyper = (; beta = 0.12, importation = 1.0e-7, infect = true)
    open_loop = merge(hyper, (; infect = false))
    replay_vf!(du, u, p, t) = _warm_vf!(du, u, (open_loop, p[2]), t)
    model = ReplayModel(_warm_vf!, replay_vf!, layout, routes)
    rate(x, m; kwargs...) = total_infection_rate(model, x, hyper, (; Rt = m), 0.0; kwargs...)

    @testset "the model's infection rate and the multiplier that matches it" begin
        x = [6.0e5, 3.0e5, 2000.0, 500.0, 9.75e4, 0.0]
        into_a = (0.12 * 0.7 * (3 * 2000.0 + 500.0) / 1.0e6 + 1.0e-7) * 6.0e5
        into_b = (0.12 * 0.7 * (2000.0 + 0.5 * 500.0) / 1.0e6 + 1.0e-7) * 3.0e5
        @test rate(x, 0.7) ≈ into_a + into_b
        @test rate(x, 0.7; per_infection = [0.01, 0.03]) ≈ 0.01 * into_a + 0.03 * into_b
        @test rate(x, 0.0) ≈ 1.0e-7 * 9.0e5
        @test matched_transmission_multiplier(m -> rate(x, m), into_a + into_b) ≈ 0.7
        @test matched_transmission_multiplier(m -> rate(x, m), 1.0e9) == 10.0
        @test matched_transmission_multiplier(Returns(3.0), 5.0) == 1.0
        shared = ReplayModel(_warm_vf!, replay_vf!, layout, InfectionRoutes([:a, :b], [1, 1], [[3, 6], [4, 6]], [1, 2], [1.0, 1.0]))
        @test_throws ArgumentError total_infection_rate(shared, x, hyper, (; Rt = 1.0), 0.0; per_infection = [0.01, 0.03])
        @test_throws DimensionMismatch rate(x, 0.7; per_infection = [0.01])
    end

    @testset "the model runs the last days itself" begin
        truth_multiplier = 0.8
        latent = Returns((; Rt = truth_multiplier))
        run(x, from, to) = deterministic_spinup(_warm_vf!, layout, x, hyper, latent, from, to)
        month_before = run([6.0e5, 3.0e5, 20000.0, 5000.0, 7.5e4, 0.0], -48.0, -28.0)
        fortnight_before = run(month_before, -28.0, -14.0)
        week_before = run(fortnight_before, -14.0, -7.0)
        truth = deterministic_spinup(
            _warm_vf!, layout, week_before, hyper, latent, -7.0, 0.0; dt = 7.0, supersample = 14, keep_accumulators = true,
        )
        closing = (; rate = rate(truth, truth_multiplier), growth = log(rate(truth, truth_multiplier) / rate(week_before, truth_multiplier)) / 7)
        @test closing.growth < -0.03

        # Started on the model's own distribution, the closed loop returns the truth.
        exact = closed_loop_start(
            model, fortnight_before, hyper, m -> (; Rt = m), closing;
            window = 14.0, accumulate_days = 7.0, fast = [3, 4, 6], first_guess = 1.3,
        )
        @test exact.multiplier ≈ truth_multiplier rtol = 1.0e-3
        @test exact.scale ≈ 1.0 rtol = 1.0e-2
        @test exact.state ≈ truth rtol = 1.0e-2

        # A history that shares the same infections out differently between the groups: the rate
        # matched at one instant gives the wrong multiplier, the closed loop does not.
        skewed = copy(month_before)
        infected = skewed[3] + skewed[4]
        skewed[3], skewed[4] = 0.2 * infected, 0.8 * infected
        at_instant = matched_transmission_multiplier(m -> rate(skewed, m), rate(month_before, truth_multiplier))
        @test abs(at_instant - truth_multiplier) > 0.2
        closed = closed_loop_start(
            model, skewed, hyper, m -> (; Rt = m), closing;
            window = 28.0, accumulate_days = 7.0, fast = [3, 4, 6],
        )
        @test closed.multiplier ≈ truth_multiplier rtol = 0.02
        @test closed.growth ≈ closing.growth atol = 1.0e-3
        @test rate(closed.state, closed.multiplier) ≈ closing.rate rtol = 1.0e-8
        # The groups are weakly coupled here, so four weeks do not quite finish the settling.
        @test closed.state[3] / closed.state[4] ≈ truth[3] / truth[4] rtol = 0.1
        @test skewed[3] / skewed[4] == 0.25
        @test closed.state[6] ≈ truth[6] rtol = 0.03

        # The result differentiates in a hyperparameter.
        slope = ConfigurableEpi.ForwardDiff.derivative(0.12) do beta
            closed_loop_start(
                model, skewed, merge(hyper, (; beta)), m -> (; Rt = m), closing;
                window = 28.0, accumulate_days = 7.0, fast = [3, 4, 6],
            ).multiplier
        end
        @test slope ≈ -truth_multiplier / 0.12 rtol = 0.05

        @test_throws ArgumentError closed_loop_start(
            model, skewed, hyper, m -> (; Rt = m), closing;
            window = 7.0, accumulate_days = 7.0, fast = [3, 4, 6],
        )
    end
end
