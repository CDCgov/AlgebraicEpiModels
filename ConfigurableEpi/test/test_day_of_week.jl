using Test
using ConfigurableEpi
using Dates: Date, Day, dayofweek
using Distributions: NegativeBinomial, LogNormal
using ForwardDiff
using LinearAlgebra: I
import Random

# The weekday observation effect (src/day_of_week.jl): a mean-1 multiplier on the observation mean
# and a per-weekday widening of the negative-binomial dispersion.

const _DOW_START = Date(2024, 9, 20)   # a Friday
const _DOW_TRUE = (1.15, 1.05, 1.0, 1.0, 0.98, 0.85, 0.97)   # Mon … Sun
const _DOW_W = _DOW_TRUE ./ (sum(_DOW_TRUE) / 7)

_dow_hyper(; kw...) = merge(
    (ascertainment = 0.005, ascertainment_decline_rate = 0.0, phi = 140.0),
    (; (n => 0.0 for n in DOW_LEARNED_NAMES)...), (; kw...),
)

# `n` days of exp(trend) × weekday weight, optionally with NB noise and a per-weekday lognormal
# wobble of variance `s2[d]` (mean 1) on the multiplier.
function _dow_series(n; level = 400.0, growth = 0.01, phi = Inf, s2 = ntuple(_ -> 0.0, 7), rng = nothing)
    dates = _DOW_START .+ Day.(0:(n - 1))
    counts = map(enumerate(dates)) do (i, date)
        d = dayofweek(date)
        mu = level * exp(growth * (i - 1)) * _DOW_W[d]
        rng === nothing && return mu
        if s2[d] > 0
            sigma2 = log1p(s2[d])
            mu *= rand(rng, LogNormal(-sigma2 / 2, sqrt(sigma2)))
        end
        return float(isfinite(phi) ? rand(rng, NegativeBinomial(phi, phi / (phi + mu))) : mu)
    end
    return dates, counts
end

@testset "Day-of-week observation effect" begin
    @testset "config validation and learned names" begin
        for cfg in (NoDayOfWeekConfig(), PluginDayOfWeekConfig(), LearnedDayOfWeekConfig())
            @test validate_day_of_week(cfg) === cfg
        end
        @test_throws ErrorException validate_day_of_week(PluginDayOfWeekConfig(window_days = 28))
        @test_throws ErrorException validate_day_of_week(PluginDayOfWeekConfig(exclude_recent_days = -1))
        @test validate_day_of_week(PluginDayOfWeekConfig(fit_policy = "first_vintage")).fit_policy ==
            "first_vintage"
        @test_throws ErrorException validate_day_of_week(PluginDayOfWeekConfig(fit_policy = "weekly"))

        @test day_of_week_learned_names(NoDayOfWeekConfig()) == ()
        @test day_of_week_learned_names(PluginDayOfWeekConfig()) == ()
        @test day_of_week_learned_names(LearnedDayOfWeekConfig()) == DOW_LEARNED_NAMES

        learned = LearnedDayOfWeekConfig()
        @test assert_day_of_week_learnable(learned, [:R0_baseline, DOW_LEARNED_NAMES...]) === nothing
        @test_throws ErrorException assert_day_of_week_learnable(learned, [:R0_baseline, :dow_z1])
        @test_throws ErrorException assert_day_of_week_learnable(PluginDayOfWeekConfig(), [:dow_z1])
        @test_throws ErrorException assert_day_of_week_learnable(NoDayOfWeekConfig(), [:dow_z1])
        @test assert_day_of_week_learnable(NoDayOfWeekConfig(), [:R0_baseline]) === nothing
    end

    @testset "weekday indexing follows start_date + t" begin
        inner = AscertainmentPath(0.2, 0.0)
        weights = ntuple(d -> Float64(d), 7)   # the multiplier IS the weekday number
        m = DayOfWeekModifier(inner, dayofweek(_DOW_START), weights)
        h = _dow_hyper()
        for t in 0:20
            @test m(h, Float64(t)) ≈ 0.005 * dayofweek(_DOW_START + Day(t))
            @test m(nothing, h, Float64(t)) == m(h, Float64(t))
        end
    end

    @testset "Helmert basis: orthonormal, zero-sum, exchangeable" begin
        H = [DOW_HELMERT[j][k] for j in 1:7, k in 1:6]
        @test H' * H ≈ Matrix(1.0I, 6, 6)
        @test all(k -> abs(sum(H[:, k])) < 1.0e-12, 1:6)
        # Equal row norms ⇒ an isotropic prior on z gives every weekday the same marginal variance
        # (6/7 of the coordinate variance) and pairwise correlation −1/6: no day is the reference.
        C = H * H'
        @test all(j -> C[j, j] ≈ 6 / 7, 1:7)
        @test all(C[i, j] ≈ -1 / 7 for i in 1:7, j in 1:7 if i != j)
        @test dow_helmert_prior_sd(0.3)^2 * 6 / 7 ≈ 0.3^2
    end

    @testset "learned weights: zero coordinates are no effect; mean 1; any pattern reachable; ForwardDiff" begin
        inner = AscertainmentPath(0.2, 0.0)
        m = DayOfWeekModifier(inner, 1, nothing)   # t = 0 is a Monday
        h = _dow_hyper()
        @test all(t -> m(h, Float64(t)) ≈ 0.005, 0:6)
        # Any zero-sum log pattern `c` is reached exactly by `z = Hᵀ c`.
        H = [DOW_HELMERT[j][k] for j in 1:7, k in 1:6]
        c = [0.2, 0.1, 0.0, -0.05, 0.0, -0.3, 0.05]
        c .-= sum(c) / 7
        hl = _dow_hyper(; (n => v for (n, v) in zip(DOW_LEARNED_NAMES, H' * c))...)
        @test collect(dow_log_effects(hl)) ≈ c
        weekly = [m(hl, Float64(t)) / 0.005 for t in 0:6]
        @test sum(weekly) / 7 ≈ 1.0
        @test weekly ≈ 7 .* exp.(c) ./ sum(exp.(c))
        @test collect(day_of_week_multipliers(hl)) ≈ weekly
        g = ForwardDiff.derivative(x -> m(_dow_hyper(dow_z1 = x), 0.0), 0.0)
        @test isfinite(g) && g != 0
    end

    @testset "learned softmax does not overflow for large coordinates" begin
        # Raw exponentials overflow near exp(710); the max-shifted softmax must still give finite
        # weights of mean 1, identical to the small-coordinate pattern scaled up.
        H = [DOW_HELMERT[j][k] for j in 1:7, k in 1:6]
        c = [800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        c .-= sum(c) / 7
        huge = _dow_hyper(; (n => v for (n, v) in zip(DOW_LEARNED_NAMES, H' * c))...)
        w = collect(day_of_week_multipliers(huge))
        @test all(isfinite, w)
        @test sum(w) / 7 ≈ 1.0
        @test w[1] ≈ 7.0   # all mass on Monday
        g = ForwardDiff.derivative(x -> day_of_week_multipliers(_dow_hyper(dow_z1 = x))[1], 900.0)
        @test isfinite(g)
    end

    @testset "remove_day_of_week_effect divides each count at its own observation date" begin
        inner = AscertainmentPath(0.2, 0.0)
        h = _dow_hyper()
        m = DayOfWeekModifier(inner, dayofweek(_DOW_START), Tuple(_DOW_W))
        times = Float64.(0:27)
        history = (; times, counts = [100.0 * _DOW_W[dayofweek(_DOW_START + Day(Int(t)))] for t in times])
        cleaned = remove_day_of_week_effect(history, m, h)
        @test cleaned.times === history.times
        @test all(≈(100.0), cleaned.counts)
        # No weekday effect, or no history: returned unchanged (bit-exact step-1 path).
        @test remove_day_of_week_effect(history, inner, h) === history
        @test remove_day_of_week_effect(nothing, m, h) === nothing
        @test day_of_week_weight(inner, h, 3.0) === true
    end

    @testset "plugin recovers known weights under growth and ignores the excluded tail" begin
        dates, counts = _dow_series(182; growth = 0.02)
        counts[(end - 13):end] .*= 3.0   # a nowcast-inflated tail the estimator must not see
        est = estimate_day_of_week_effects(dates, counts; phi = 140.0, window_days = 182, exclude_recent_days = 14)
        @test sum(est.weights) / 7 ≈ 1.0
        @test all(isapprox.(est.weights, _DOW_W; atol = 1.0e-2))
        @test all(<(1.0e-4), est.extra_var)   # deterministic series: nothing beyond the NB floor
        @test !est.fallback
        @test est.n_used > 0
    end

    @testset "plugin extra variance: sustained growth adds nothing (recentring)" begin
        # Without recentring, the second-order baseline bias (7/3)r² adds its square, ≈ 5.5e-4 at
        # r = 0.1, to every weekday. Noise-free counts with a huge phi isolate that bias.
        for r in (0.1, -0.1)
            dates, counts = _dow_series(182; level = 1.0e9, growth = r)
            est = estimate_day_of_week_effects(dates, counts; phi = 1.0e12, window_days = 182, exclude_recent_days = 0)
            @test all(<(1.0e-8), est.extra_var)
        end
    end

    @testset "plugin extra variance: recovers s_d², no double counting" begin
        rng = Random.MersenneTwister(20260916)
        s2 = (0.02, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0)
        dates, counts = _dow_series(2100; growth = 0.0, phi = 140.0, s2, rng)
        est = estimate_day_of_week_effects(dates, counts; phi = 140.0, window_days = 2100, exclude_recent_days = 0)
        @test all(isapprox.(est.weights, _DOW_W; atol = 3.0e-2))
        @test est.extra_var[1] ≈ 0.02 atol = 0.01
        @test est.extra_var[6] ≈ 0.04 atol = 0.015
        for d in (2, 3, 4, 5, 7)
            @test est.extra_var[d] < 0.006   # NB + Poisson already carry these days
        end
    end

    @testset "short history and gaps" begin
        dates, counts = _dow_series(20)
        est = @test_logs (:warn,) match_mode = :any estimate_day_of_week_effects(dates, counts; phi = 140.0)
        @test est.weights == ntuple(_ -> 1.0, 7)
        @test est.extra_var == ntuple(_ -> 0.0, 7)
        @test est.fallback
        dates, counts = _dow_series(100)
        @test_throws ArgumentError estimate_day_of_week_effects(dates[[1:50; 52:100]], counts[[1:50; 52:100]]; phi = 140.0)
        # A weekday with no counts would get weight 0, a zero observation mean on that weekday.
        counts[dayofweek.(dates) .== 7] .= 0.0
        est = @test_logs (:warn,) match_mode = :any estimate_day_of_week_effects(dates, counts; phi = 140.0, exclude_recent_days = 0)
        @test est.weights == ntuple(_ -> 1.0, 7)
    end

    @testset "operational history matches the model calendar" begin
        dates, counts = _dow_series(60)
        report_date = last(dates) + Day(1)
        prepared = prepare_day_of_week_history(dates, counts, report_date)
        @test prepared.dates == dates[1:(end - 1)]
        @test prepared.counts == counts[1:(end - 1)]
        @test_throws ArgumentError prepare_day_of_week_history(
            dates[1:(end - 1)], counts[1:(end - 1)], report_date,
        )
        @test_throws ArgumentError prepare_day_of_week_history(
            dates[[1:20; 22:end]], counts[[1:20; 22:end]], report_date,
        )
    end

    @testset "dispersion is phi when the extra variance is zero" begin
        h = _dow_hyper()
        @test DayOfWeekDispersion(3, ntuple(_ -> 0.0, 7))(nothing, h, 5.0) == 140.0
        p = DayOfWeekDispersion(1, (0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        @test p(nothing, h, 0.0) ≈ 1 / (1 / 140 + 0.01)
        @test p(nothing, h, 1.0) ≈ 140.0
    end

    @testset "build_day_of_week_observation dispatches on the config type" begin
        inner = AscertainmentPath(0.2, 0.0)
        h = _dow_hyper()
        none = build_day_of_week_observation(NoDayOfWeekConfig(), inner, _DOW_START, nothing; phi = 140.0)
        @test none.modifier === inner   # the step-1 model, bit-exact
        @test none.dispersion(nothing, h, 3.0) === h.phi
        @test none.effects === nothing
        @test none.derived_hyperparameters === nothing
        @test day_of_week_report_rows(none.effects) == ()

        dates, counts = _dow_series(120)
        history = (times = Float64.(0:119), counts = counts)
        plugin = build_day_of_week_observation(PluginDayOfWeekConfig(), inner, _DOW_START, history; phi = 140.0)
        @test plugin.effects.weights isa NTuple{7, Float64}
        @test plugin.modifier(h, 0.0) ≈ 0.005 * plugin.effects.weights[dayofweek(_DOW_START)]
        @test plugin.derived_hyperparameters === nothing
        rows = day_of_week_report_rows(plugin.effects)
        @test length(rows) == 14
        @test ("dow_multiplier_mon", "plugin", plugin.effects.weights[1]) in rows

        fixed_cfg = PluginDayOfWeekConfig(fit_policy = "first_vintage")
        fixed = build_day_of_week_observation(
            fixed_cfg, inner, _DOW_START, nothing;
            phi = 140.0, precomputed = plugin.effects,
        )
        @test fixed.effects === plugin.effects
        @test fixed.modifier(h, 0.0) == plugin.modifier(h, 0.0)
        @test_throws ErrorException build_day_of_week_observation(
            fixed_cfg, inner, _DOW_START, nothing;
            phi = 140.0, precomputed = ConfigurableEpi._NO_DOW_EFFECTS,
        )
        @test_throws ErrorException build_day_of_week_observation(
            NoDayOfWeekConfig(), inner, _DOW_START, nothing;
            phi = 140.0, precomputed = plugin.effects,
        )

        # Learned: weights from θ, no history read, no extra variance (the ensemble carries it).
        learned = build_day_of_week_observation(LearnedDayOfWeekConfig(), inner, _DOW_START, nothing; phi = 140.0)
        @test learned.dispersion(nothing, h, 3.0) === h.phi
        @test learned.effects === nothing
        @test day_of_week_report_rows(learned.effects) == ()
        @test learned.modifier(h, 0.0) ≈ 0.005   # zero coordinates ⇒ no effect
        derived = learned.derived_hyperparameters(h)
        @test keys(derived) == Tuple(Symbol("dow_multiplier_", d) for d in DOW_DAY_ABBREVIATIONS)
        @test all(≈(1.0), values(derived))

        @test_logs (:warn,) match_mode = :any build_day_of_week_observation(
            PluginDayOfWeekConfig(), inner, _DOW_START, nothing; phi = 140.0,
        )
    end
end
