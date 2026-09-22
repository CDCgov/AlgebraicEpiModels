using Test
using Dates
using ForwardDiff
using Statistics: mean
using ConfigurableEpi

# ----------------------------------------------------------------------------
# Transmission seasonality forcing (src/seasonality.jl).
#
# The properties that matter downstream are: the opt-in legacy cosine is reproduced bit-for-bit,
# and the empirical curve is exactly periodic with unit annual mean (so it reshapes transmission
# without shifting its level).
#
# ConfigurableEpi ships NO data: the climatology is injected by the caller. These tests therefore
# run on a SYNTHETIC climatology with the structural properties of a real one — annual-mean-1
# curves, a near-sinusoidal northern location (`ny`), a southern location with a second, summer
# peak (`az`), and a national curve (`us`). Tests of a real artifact's content (location coverage,
# the harmonic-R² contrast) belong with the artifact, beside the run script that owns it.
# ----------------------------------------------------------------------------

const _CLIM = let u = [(k - 0.5) / 52 for k in 1:52]
    unit_mean(knots) = knots ./ mean(knots)
    Dict(
        "ny" => unit_mean([1 + 0.3 * cospi(2x) for x in u]),
        "az" => unit_mean([1 + 0.12 * cospi(2x) + 0.18 * cospi(4x) for x in u]),
        "us" => unit_mean([1 + 0.25 * cospi(2x) for x in u]),
    )
end

function _write_climatology(path, climatology; header = "location,knot,value")
    open(path, "w") do io
        println(io, header)
        for loc in sort!(collect(keys(climatology))), (k, v) in enumerate(climatology[loc])
            println(io, loc, ',', k, ',', repr(v))
        end
    end
    return path
end

# A hyperparameter NamedTuple carrying every seasonality parameter, as base_hp does.
_hp(; amp = 0.1, phase = 15.0, kappa = 1.0) =
    (seasonal_amp = amp, seasonal_phase = phase, seasonal_kappa = kappa)

# Dense sample of one year, used for the mean/periodicity properties. 97 is coprime with the
# 52 knots, so samples never align with knot positions.
_year_grid(n = 52 * 97) = (365.25 * i / n for i in 0:(n - 1))

@testset "Seasonality" begin
    @testset "UnitForcing" begin
        f = UnitForcing()
        @test all(f(_hp(), t) == 1.0 for t in 0.0:7.0:1400.0)
        @test @inferred(f(_hp(), 42.0)) === 1.0
    end

    @testset "CosineForcing reproduces the legacy expression" begin
        # The literal expression that lived inline in basic_seir.jl/two_strain_escape.jl before
        # the shared forcing existed. Bit-identical, not approximate: existing configurations
        # must produce byte-identical forecasts.
        legacy(amp, phase, day0, t) =
            1.0 + amp * cospi(2 * (t + day0 - phase) / 365.25)
        for day0 in (0.0, 1.0, 247.0), phase in (-30.0, 15.0, 200.0), amp in (0.0, 0.1, 0.15)
            f = CosineForcing(day0)
            hp = _hp(amp = amp, phase = phase)
            @test all(f(hp, t) === legacy(amp, phase, day0, t) for t in 0.0:1.0:800.0)
        end
        @test @inferred(CosineForcing(1.0)(_hp(), 10.0)) isa Float64
    end

    @testset "build_periodic_curve" begin
        knots = [1 + 0.3 * cospi(2 * (k - 0.5) / 52) for k in 1:52]
        curve = build_periodic_curve(knots)

        @testset "interpolates the knots" begin
            # Knot values are recovered up to the unit-mean renormalisation.
            scale = mean(knots)
            @test all(
                isapprox(curve((k - 0.5) / 52), knots[k] / scale; atol = 1.0e-12) for k in 1:52
            )
        end

        @testset "periodic and continuous at the year boundary" begin
            @test all(isapprox(curve(u), curve(u + 1); atol = 1.0e-12) for u in 0.0:0.005:0.999)
            # The seam is the one place a naive spline fit shows a jump.
            @test curve(1.0) ≈ curve(0.0) atol = 1.0e-12
            left = curve(1.0 - 1.0e-9)
            @test abs(left - curve(0.0)) < 1.0e-6
        end

        @testset "unit annual mean, exactly" begin
            @test mean(curve(u) for u in range(0, 1; length = 5001)[1:(end - 1)]) ≈ 1.0 atol = 1.0e-9
        end

        @testset "smooth (no piecewise-linear kinks)" begin
            vals = [curve(i / 2000) for i in 0:1999]
            d2 = [abs(vals[i - 1] - 2vals[i] + vals[i + 1]) for i in 2:1999]
            # A linear interpolant through 52 knots spikes at every knot; a cubic spline's
            # second difference stays O(h^2).
            @test maximum(d2) < 1.0e-3
        end
    end

    @testset "year_fraction" begin
        @test year_fraction(Date(2023, 1, 1)) == 0.0
        @test year_fraction(Date(2023, 12, 31)) ≈ 364 / 365
        @test year_fraction(Date(2024, 12, 31)) ≈ 365 / 366   # leap-aware
        # 1 March sits at the same point of the year either side of a leap day.
        @test year_fraction(Date(2024, 3, 1)) ≈ year_fraction(Date(2023, 3, 1)) atol = 1 / 365
        @test all(0 <= year_fraction(Date(2023, 1, 1) + Day(i)) < 1 for i in 0:400)
    end

    @testset "climatology loader and validator" begin
        mktempdir() do dir
            path = _write_climatology(joinpath(dir, "climatology.csv2"), _CLIM)
            loaded = load_indoor_activity_climatology(path)
            # `repr` round-trips Float64 exactly, so a caller-owned file reproduces the curves
            # bit-for-bit.
            @test loaded == _CLIM
            @test all(length(knots) == N_SEASON_KNOTS for knots in values(loaded))

            # The package has no data of its own: a path is required.
            @test_throws MethodError load_indoor_activity_climatology()
            @test_throws ErrorException load_indoor_activity_climatology(
                joinpath(dir, "absent.csv2")
            )

            bad_header = _write_climatology(
                joinpath(dir, "bad_header.csv2"), _CLIM; header = "loc,knot,value"
            )
            @test_throws ErrorException load_indoor_activity_climatology(bad_header)

            # A missing knot is left NaN by the reader and rejected.
            lines = readlines(path)
            short = joinpath(dir, "short.csv2")
            write(short, join(lines[1:(end - 1)], "\n") * "\n")
            @test_throws ErrorException load_indoor_activity_climatology(short)
        end

        @test validate_indoor_activity_climatology(_CLIM) == _CLIM
        @test_throws ErrorException validate_indoor_activity_climatology(
            Dict{String, Vector{Float64}}()
        )
        @test_throws ErrorException validate_indoor_activity_climatology(
            Dict("NY" => _CLIM["ny"])
        )
        @test_throws ErrorException validate_indoor_activity_climatology(
            Dict("ny" => _CLIM["ny"][1:51])
        )
        @test_throws ErrorException validate_indoor_activity_climatology(
            Dict("ny" => 1.1 .* _CLIM["ny"])       # annual mean 1.1: a level shift
        )
        @test_throws ErrorException validate_indoor_activity_climatology(
            Dict("ny" => vcat(-1.0, _CLIM["ny"][2:end]))
        )
    end

    @testset "build_seasonal_forcing" begin
        @testset "mode selection" begin
            @test SeasonalityConfig().mode == "indoor_activity"
            @test build_seasonal_forcing(
                SeasonalityConfig(), "ny", Date(2023, 1, 1); climatology = _CLIM
            ) isa IndoorActivityForcing
            # No data ships with the package, so the default mode without a climatology is an
            # error that says so — never a silent flat curve.
            err = try
                build_seasonal_forcing(SeasonalityConfig(), "ny", Date(2023, 1, 1))
            catch e
                sprint(showerror, e)
            end
            @test occursin("ships no data", err)
            # The data-free modes ignore the keyword entirely.
            @test build_seasonal_forcing(
                SeasonalityConfig(mode = "cosine"), "ny", Date(2023, 1, 1); climatology = _CLIM
            ) isa CosineForcing
            @test build_seasonal_forcing(
                SeasonalityConfig(mode = "none"), "ny", Date(2023, 1, 1)
            ) isa UnitForcing
            @test build_seasonal_forcing(
                SeasonalityConfig(mode = "cosine"), "ny", Date(2023, 1, 1)
            ) isa CosineForcing
            @test build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 1, 1); climatology = _CLIM,
            ) isa IndoorActivityForcing
        end

        @testset "cosine anchors on the calendar" begin
            f = build_seasonal_forcing(
                SeasonalityConfig(mode = "cosine"), "ny", Date(2023, 3, 15)
            )
            @test f.day0 == Float64(dayofyear(Date(2023, 3, 15)))
        end

        @testset "empirical forcing properties" begin
            f = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 1, 1); climatology = _CLIM,
            )

            @test all(isapprox(f(_hp(), t), f(_hp(), t + 365.25); atol = 1.0e-12) for t in 0.0:5.0:360.0)
            @test all(isfinite(f(_hp(), t)) && f(_hp(), t) > 0 for t in 0.0:1.0:1500.0)
            @test @inferred(f(_hp(), 100.0)) isa Float64

            @testset "unit annual mean for every kappa" begin
                for kappa in (0.0, 0.25, 0.5, 1.0, 2.0)
                    hp = _hp(kappa = kappa)
                    @test mean(f(hp, t) for t in _year_grid()) ≈ 1.0 atol = 1.0e-6
                end
            end

            @testset "kappa scales deviation from 1" begin
                sigma(t) = f(_hp(kappa = 1.0), t)
                @test all(f(_hp(kappa = 0.0), t) == 1.0 for t in 0.0:7.0:700.0)
                for t in 0.0:11.0:700.0
                    @test f(_hp(kappa = 0.5), t) ≈ 1 + 0.5 * (sigma(t) - 1)
                    @test f(_hp(kappa = 0.25), t) ≈ 1 + 0.25 * (sigma(t) - 1)
                end
            end

            @testset "calendar anchoring" begin
                # Starting mid-year and stepping forward lands on the same seasonal position as
                # starting at New Year and stepping to the same calendar day.
                #
                # Agreement is close but not exact, by design: the anchor `u0` is leap-aware
                # (`dayofyear/daysinyear`, so 1 March sits at the same seasonal position in any
                # year) while the model clock advances at 1/365.25 per day. The mismatch is
                # bounded by 0.25 day/year — two orders of magnitude below the 7-day knot
                # spacing — so it is tolerated rather than removed. Removing it would mean
                # calendar arithmetic inside the ODE right-hand side.
                jan = build_seasonal_forcing(
                    SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 1, 1); climatology = _CLIM,
                )
                jul = build_seasonal_forcing(
                    SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 7, 1); climatology = _CLIM,
                )
                offset = Float64(Dates.value(Date(2023, 7, 1) - Date(2023, 1, 1)))

                # The anchors describe the same calendar day to well under a day.
                @test abs(jul.u0 - offset / 365.25) * 365.25 < 0.25
                # And the forcings therefore track each other closely.
                @test all(
                    isapprox(jul(_hp(), t), jan(_hp(), t + offset); atol = 0.01)
                        for t in 0.0:10.0:300.0
                )
            end

            @testset "leap year stays in range" begin
                leap = build_seasonal_forcing(
                    SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2024, 1, 1); climatology = _CLIM,
                )
                knots = _CLIM["ny"]
                lo, hi = extrema(knots ./ mean(knots))
                # A cubic spline may overshoot slightly between knots; allow a small margin.
                @test all(lo - 0.05 <= leap(_hp(), t) <= hi + 0.05 for t in 0.0:1.0:400.0)
            end
        end

        @testset "different locations give different curves" begin
            ny = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 1, 1); climatology = _CLIM,
            )
            az = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "az", Date(2023, 1, 1); climatology = _CLIM,
            )
            @test any(abs(ny(_hp(), t) - az(_hp(), t)) > 0.05 for t in 0.0:7.0:360.0)
        end

        @testset "location is case-insensitive" begin
            lower = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 1, 1); climatology = _CLIM,
            )
            upper = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "NY", Date(2023, 1, 1); climatology = _CLIM,
            )
            @test all(lower(_hp(), t) == upper(_hp(), t) for t in 0.0:7.0:360.0)
        end

        @testset "fallback for uncovered locations" begin
            # A location the climatology does not cover.
            us = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "us", Date(2023, 1, 1); climatology = _CLIM,
            )
            fell_back = @test_logs (:warn,) build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity", fallback = "us"), "pr",
                Date(2023, 1, 1); climatology = _CLIM,
            )
            @test all(fell_back(_hp(), t) == us(_hp(), t) for t in 0.0:7.0:360.0)

            flat = @test_logs (:warn,) build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity", fallback = "none"), "pr",
                Date(2023, 1, 1); climatology = _CLIM,
            )
            @test all(isapprox(flat(_hp(), t), 1.0; atol = 1.0e-12) for t in 0.0:7.0:360.0)

            @test_throws ErrorException build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity", fallback = "error"), "pr",
                Date(2023, 1, 1); climatology = _CLIM,
            )
        end
    end

    @testset "config validation" begin
        @test validate_seasonality(SeasonalityConfig()) isa SeasonalityConfig
        @test_throws ErrorException validate_seasonality(SeasonalityConfig(mode = "cosinus"))
        @test_throws ErrorException validate_seasonality(SeasonalityConfig(fallback = "canada"))
        @test_throws ErrorException validate_seasonality(SeasonalityConfig(kappa = -0.1))
        @test_throws ErrorException validate_seasonality(SeasonalityConfig(kappa = NaN))
        # The error should name the valid modes, not just reject.
        err = try
            validate_seasonality(SeasonalityConfig(mode = "cosinus"))
        catch e
            sprint(showerror, e)
        end
        @test all(occursin(m, err) for m in ("cosine", "indoor_activity", "none"))
    end

    @testset "mode-aware stability bound" begin
        specs = Dict(
            "seasonal_amp" => PriorSpec(mean = 0.8, std = 0.1),
            "seasonal_kappa" => PriorSpec(
                mean = 0.8, std = 0.15, constraint = "unit_interval"
            ),
        )
        @test seasonal_forcing_upper_bound(
            SeasonalityConfig(mode = "none"), 4.0, specs, Symbol[]
        ) == 1.0

        cosine = SeasonalityConfig(mode = "cosine")
        learned_cosine = seasonal_forcing_upper_bound(cosine, 0.2, specs, Symbol[])
        @test learned_cosine == 1 + prior_upper(specs["seasonal_amp"]; probability = 0.95)
        @test learned_cosine > 1.35
        @test seasonal_forcing_upper_bound(
            cosine, 0.6, specs, [:R0_baseline]
        ) == 1.6

        indoor = SeasonalityConfig(mode = "indoor_activity", kappa = 0.5)
        half_bound = seasonal_forcing_upper_bound(
            indoor, 9.0, specs, [:R0_baseline]; climatology = _CLIM
        )
        full_bound = seasonal_forcing_upper_bound(
            indoor, 9.0, specs, [:seasonal_kappa]; climatology = _CLIM
        )
        @test_throws ErrorException seasonal_forcing_upper_bound(
            indoor, 9.0, specs, [:seasonal_kappa]
        )
        # A single-location run bounds only its own curve, so its bound can only be tighter.
        one_location = seasonal_forcing_upper_bound(
            indoor, 9.0, specs, [:seasonal_kappa]; climatology = Dict("us" => _CLIM["us"])
        )
        @test 1.0 < one_location <= full_bound
        @test half_bound ≈ 1 + 0.5 * (full_bound - 1)
        dense_max = maximum(
            maximum(
                    build_periodic_curve(knots)(u) for u in range(0, 1; length = 5001)
                ) for knots in values(_CLIM)
        )
        @test full_bound >= dense_max
        @test full_bound > maximum(maximum, values(_CLIM))
    end

    @testset "learned-parameter policy" begin
        # The cosine's amplitude and phase are both learned (matching basic_seir's default set);
        # the empirical curve supplies both from data, so it learns neither and its strength
        # `kappa` stays a config value.
        @test default_seasonal_learned(SeasonalityConfig(mode = "cosine")) ==
            [:seasonal_amp, :seasonal_phase]
        @test isempty(default_seasonal_learned(SeasonalityConfig(mode = "indoor_activity")))
        @test isempty(default_seasonal_learned(SeasonalityConfig(mode = "none")))

        @testset "rejects parameters the active mode ignores" begin
            cos_cfg = SeasonalityConfig(mode = "cosine")
            emp_cfg = SeasonalityConfig(mode = "indoor_activity")
            none_cfg = SeasonalityConfig(mode = "none")

            @test assert_seasonal_learnable(cos_cfg, [:seasonal_phase, :seasonal_amp]) === nothing
            @test_throws ErrorException assert_seasonal_learnable(cos_cfg, [:seasonal_kappa])

            @test_throws ErrorException assert_seasonal_learnable(
                emp_cfg, [:seasonal_kappa, :Rt_sigma_stat]
            )
            bounded = Dict(
                "seasonal_kappa" => PriorSpec(
                    mean = 0.8, std = 0.15, constraint = "unit_interval"
                ),
            )
            unbounded = Dict(
                "seasonal_kappa" => PriorSpec(mean = 0.8, std = 0.15),
            )
            @test assert_seasonal_learnable(emp_cfg, [:seasonal_kappa], bounded) === nothing
            @test_throws ErrorException assert_seasonal_learnable(
                emp_cfg, [:seasonal_kappa], unbounded
            )
            @test_throws ErrorException assert_seasonal_learnable(emp_cfg, [:seasonal_phase])
            @test_throws ErrorException assert_seasonal_learnable(emp_cfg, [:seasonal_amp])

            @test_throws ErrorException assert_seasonal_learnable(none_cfg, [:seasonal_kappa])
            @test assert_seasonal_learnable(none_cfg, [:Rt_sigma_stat, :phi]) === nothing
        end
    end

    @testset "allocation-free in the hot loop" begin
        # Every forcing is evaluated inside the ODE right-hand side — at each RK4 substep, for
        # every particle — so steady-state allocation must be zero. Measured over many calls in
        # a concretely-typed loop: a single `@allocated` on a value-returning callable measures
        # the boxed return instead (see the budget note in test_closure_boxing.jl).
        function _sum_forcing(f, hp, n)
            s = 0.0
            for i in 1:n
                s += f(hp, Float64(i))
            end
            return s
        end

        # Measured inside a function so the call specialises on the concrete forcing type — the
        # same function-barrier discipline the submodels use. Looping over a heterogeneous tuple
        # of forcings instead would measure the dynamic dispatch, not the forcing.
        function _hot_loop_allocs(f, hp)
            _sum_forcing(f, hp, 100)   # warm up
            return @allocated _sum_forcing(f, hp, 100_000)
        end

        @test _hot_loop_allocs(UnitForcing(), _hp()) == 0
        @test _hot_loop_allocs(CosineForcing(247.0), _hp()) == 0
        @test _hot_loop_allocs(
            build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 9, 3); climatology = _CLIM,
            ),
            _hp(),
        ) == 0
    end

    @testset "ForwardDiff" begin
        # The UKF objective differentiates through the hyperparameters, so each mode must
        # propagate a Dual through the parameter it actually reads.
        @testset "cosine: d/d(phase) and d/d(amp)" begin
            f = CosineForcing(10.0)
            amp, phase, t = 0.1, 15.0, 100.0
            dphase = ForwardDiff.derivative(p -> f(_hp(amp = amp, phase = p), t), phase)
            @test dphase ≈ amp * (2π / 365.25) * sinpi(2 * (t + 10.0 - phase) / 365.25)
            damp = ForwardDiff.derivative(a -> f(_hp(amp = a, phase = phase), t), amp)
            @test damp ≈ cospi(2 * (t + 10.0 - phase) / 365.25)
        end

        @testset "indoor_activity: d/d(kappa) is sigma - 1" begin
            f = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "az", Date(2023, 9, 3); climatology = _CLIM,
            )
            for t in (0.0, 40.0, 200.0, 400.0)
                dkappa = ForwardDiff.derivative(k -> f(_hp(kappa = k), t), 1.0)
                @test dkappa ≈ f(_hp(kappa = 1.0), t) - 1.0
            end
        end

        @testset "none: derivative is zero" begin
            f = UnitForcing()
            @test ForwardDiff.derivative(k -> f(_hp(kappa = k), 50.0), 1.0) == 0.0
        end
    end
end
