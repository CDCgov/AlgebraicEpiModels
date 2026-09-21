using Test
using ConfigurableEpi
using Dates: Date, Dates
using ForwardDiff

# The declining ascertainment path (src/ascertainment.jl): the report's
#     alpha(t) = alpha_min + (alpha_ref - alpha_min) * exp(-r (t - t_ref) / 365.25)
# with the level and the rate read from `hyper`, the floor a fraction of the level, and the
# reference date anchored on the calendar once at build time.

_hyper(level, rate) = (ascertainment = level, ascertainment_decline_rate = rate)

@testset "Ascertainment path" begin
    @testset "the prior constants are the report's Table 4 (NSSP ED visits)" begin
        phi_obs = 0.678
        @test ASCERTAINMENT_DECLINE_RATE_PRIOR_MEAN ≈ -0.5 * log(phi_obs) atol = 5.0e-4
        @test ASCERTAINMENT_DECLINE_RATE_PRIOR_SD ≈ -log(phi_obs) / (2 * 1.6449) atol = 5.0e-4
        @test exp(-ASCERTAINMENT_DECLINE_RATE_PRIOR_MEAN) ≈ 0.824 atol = 1.0e-3
        @test DEFAULT_ASCERTAINMENT_DECLINE_RATE == ASCERTAINMENT_DECLINE_RATE_PRIOR_MEAN
        @test 0 < DEFAULT_ASCERTAINMENT_FLOOR_FRACTION < 1
        @test Date(DEFAULT_ASCERTAINMENT_REFERENCE_DATE) == Date(2024, 7, 1)
    end

    @testset "a zero rate is the constant level, bit for bit" begin
        path = AscertainmentPath(0.2, 100.0)
        for level in (0.005, 0.0018, 1.0), t in (-1000.0, 0.0, 100.0, 5000.0)
            @test path(_hyper(level, 0.0), t) === level
        end
        # ...and so is the kernel it is built from.
        @test ascertainment_at(0.005, 0.2, 0.0, 731.0) === 0.005
    end

    @testset "the level is pinned at the reference date whatever the rate or window start" begin
        ref = Date(2024, 7, 1)
        late = AscertainmentPath(; floor_fraction = 0.2, reference_date = ref, start_date = Date(2023, 12, 30))
        early = AscertainmentPath(; floor_fraction = 0.2, reference_date = ref, start_date = Date(2022, 10, 1))
        @test late.t_ref_days == 184.0
        @test early.t_ref_days == Float64(Dates.value(ref - Date(2022, 10, 1)))
        for rate in (-0.3, 0.0, 0.1943, 1.0)
            h = _hyper(0.005, rate)
            @test late(h, late.t_ref_days) ≈ 0.005 rtol = 1.0e-14
            @test early(h, early.t_ref_days) ≈ 0.005 rtol = 1.0e-14
            # Two windows, one calendar: the same calendar day gives the same ascertainment.
            for offset in (-400.0, -30.0, 0.0, 45.5, 800.0)
                @test late(h, late.t_ref_days + offset) ≈ early(h, early.t_ref_days + offset) rtol = 1.0e-13
            end
        end
    end

    @testset "matches the report's closed form and its annual factor" begin
        level, f, rate = 0.005, 0.2, 0.25
        path = AscertainmentPath(f, 50.0)
        alpha_min = f * level
        for t in (-500.0, 0.0, 50.0, 400.0, 3000.0)
            expected = alpha_min + (level - alpha_min) * exp(-rate * (t - 50.0) / 365.25)
            @test path(_hyper(level, rate), t) ≈ expected rtol = 1.0e-12
        end
        one_year_later = path(_hyper(level, rate), 50.0 + 365.25)
        @test (one_year_later - alpha_min) / (level - alpha_min) ≈ exp(-rate) rtol = 1.0e-12
    end

    @testset "floor, monotonicity, and the sign of the rate" begin
        level, f = 0.005, 0.2
        path = AscertainmentPath(f, 0.0)
        ts = range(-1000.0, 20_000.0; length = 500)
        declining = [path(_hyper(level, 0.4), t) for t in ts]
        @test all(declining .> f * level)
        @test issorted(declining; rev = true)
        @test path(_hyper(level, 0.4), 1.0e7) ≈ f * level rtol = 1.0e-9
        rising = [path(_hyper(level, -0.4), t) for t in ts]
        @test issorted(rising)
        @test rising[end] > level
        # A floor fraction of one is the constant level whatever the rate.
        flat = AscertainmentPath(1.0, 0.0)
        @test all(flat(_hyper(level, 0.7), t) == level for t in ts)
        # Before the reference date the path sits ABOVE the level for a positive rate.
        @test path(_hyper(level, 0.4), -365.25) > level
    end

    @testset "differentiates through the rate and the level (ForwardDiff)" begin
        level, f, rate, tau = 0.005, 0.2, 0.3, 200.0
        d_rate = ForwardDiff.derivative(r -> ascertainment_at(level, f, r, tau), rate)
        @test d_rate ≈ -level * (1 - f) * (tau / 365.25) * exp(-rate * tau / 365.25) rtol = 1.0e-10
        d_level = ForwardDiff.derivative(l -> ascertainment_at(l, f, rate, tau), level)
        @test d_level ≈ 1 + (1 - f) * expm1(-rate * tau / 365.25) rtol = 1.0e-12
        path = AscertainmentPath(f, 100.0)
        g = ForwardDiff.gradient(
            v -> path((ascertainment = v[1], ascertainment_decline_rate = v[2]), 300.0),
            [level, rate],
        )
        @test g[2] ≈ -level * (1 - f) * (200.0 / 365.25) * exp(-rate * 200.0 / 365.25) rtol = 1.0e-10
    end

    @testset "config validation names the field" begin
        good = (
            ascertainment = 0.005, ascertainment_decline_rate = 0.19,
            ascertainment_floor_fraction = 0.2, ascertainment_reference_date = "2024-07-01",
            ascertainment_rate_bound = 1.0,
        )
        @test validate_ascertainment(good) === nothing
        built = build_ascertainment_path(good, Date(2023, 12, 30))
        @test built isa AscertainmentPath
        @test built.t_ref_days == 184.0
        @test built.floor_fraction == 0.2

        bad_date = merge(good, (ascertainment_reference_date = "2024/07/01",))
        err = try
            validate_ascertainment(bad_date)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("ascertainment_reference_date", sprint(showerror, err))
        @test_throws ArgumentError build_ascertainment_path(bad_date, Date(2023, 12, 30))

        for (field, value) in (
                (:ascertainment_floor_fraction, 1.5),
                (:ascertainment_floor_fraction, -0.1),
                (:ascertainment, 0.0),
                (:ascertainment, NaN),
                (:ascertainment_decline_rate, Inf),
            )
            broken = merge(good, NamedTuple{(field,)}((value,)))
            err = try
                validate_ascertainment(broken)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(string(field), sprint(showerror, err))
        end
        @test_throws ArgumentError AscertainmentPath(1.2, 0.0)
        @test_throws ArgumentError AscertainmentPath(0.2, NaN)
    end

    @testset "a floor fraction of one makes the rate unlearnable" begin
        fixed = (
            ascertainment = 0.005, ascertainment_decline_rate = 0.19,
            ascertainment_floor_fraction = 1.0, ascertainment_reference_date = "2024-07-01",
            ascertainment_rate_bound = 1.0,
        )
        @test_throws ErrorException assert_ascertainment_learnable(fixed, [:ascertainment_decline_rate])
        @test_throws ErrorException assert_ascertainment_learnable(fixed, ["ascertainment_decline_rate"])
        @test assert_ascertainment_learnable(fixed, [:R0_baseline]) === nothing
        lowered = merge(fixed, (ascertainment_floor_fraction = 0.2,))
        @test assert_ascertainment_learnable(lowered, [:ascertainment_decline_rate]) === nothing
    end
end

@testset "the rate bound clamps what the path uses, and nothing inside it" begin
    level = 0.005
    bounded = AscertainmentPath(0.2, 100.0, 1.0)
    free = AscertainmentPath(0.2, 100.0)          # positional default: no bound
    @test free.rate_bound == Inf
    @test DEFAULT_ASCERTAINMENT_RATE_BOUND == 1.0
    for rate in (-0.9, -0.3, 0.0, 0.1943, 0.39, 0.99), t in (-400.0, 0.0, 100.0, 800.0)
        @test bounded(_hyper(level, rate), t) === free(_hyper(level, rate), t)   # identity inside
    end
    @test bounded(_hyper(level, 0.0), 800.0) === level
    # Outside the bound the path is the path at the bound, so the absurd pre-reference growth an
    # unbounded rate offers (`r ≈ 23`/yr ⇒ hundreds of observations per infection) is unreachable.
    @test bounded(_hyper(level, 23.0), 0.0) == bounded(_hyper(level, 1.0), 0.0)
    @test bounded(_hyper(level, -7.0), 900.0) == bounded(_hyper(level, -1.0), 900.0)
    @test bounded(_hyper(level, 23.0), 0.0) < 2 * level
    @test free(_hyper(level, 23.0), 0.0) > 100 * level
    # ...and the likelihood gradient with respect to the rate is zero out there, so only the prior
    # pulls it back; inside the bound the derivative is the unbounded path's.
    g_out = ForwardDiff.derivative(r -> bounded((ascertainment = level, ascertainment_decline_rate = r), 0.0), 23.0)
    g_in = ForwardDiff.derivative(r -> bounded((ascertainment = level, ascertainment_decline_rate = r), 0.0), 0.2)
    g_free = ForwardDiff.derivative(r -> free((ascertainment = level, ascertainment_decline_rate = r), 0.0), 0.2)
    @test g_out == 0.0
    @test g_in == g_free

    good = (
        ascertainment = 0.005, ascertainment_decline_rate = 0.19, ascertainment_floor_fraction = 0.2,
        ascertainment_reference_date = "2024-07-01", ascertainment_rate_bound = 1.0,
    )
    built = build_ascertainment_path(good, Date(2023, 12, 30))
    @test built.rate_bound == 1.0
    @test build_ascertainment_path(merge(good, (ascertainment_rate_bound = Inf,)), Date(2023, 12, 30)).rate_bound == Inf
    for bad in (0.0, -1.0, NaN)
        err = try
            validate_ascertainment(merge(good, (ascertainment_rate_bound = bad,)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("ascertainment_rate_bound", sprint(showerror, err))
    end
    @test_throws ArgumentError AscertainmentPath(0.2, 0.0, 0.0)
end
