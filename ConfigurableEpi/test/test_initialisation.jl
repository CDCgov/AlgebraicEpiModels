# Initialisation: the strategy options, the integration stability guard and the seeds each strategy builds.

using Test
using ConfigurableEpi
using ConfigurableEpi: Rk4, Rk3, Heun, ForwardEuler

@option struct InitialisationRun
    initialisation::InitialisationConfig = PeakFromHistoryInitialisation()
end

@testset "Initialisation strategies as options" begin
    @test from_dict(InitialisationRun, Dict{String, Any}()).initialisation == PeakFromHistoryInitialisation()
    slow = from_dict(InitialisationRun, Dict("initialisation" => Dict("slow_mode" => Dict("boundary_low" => 0.2))))
    @test slow.initialisation == SlowModeInitialisation(boundary_low = 0.2)
    @test slow.initialisation isa InitialisationStrategy
    # An option table built in Julia, with symbol keys and an empty block, parses once normalised.
    built = normalise_config_dict(Dict(:initialisation => Dict(:peak_at_start => Dict())))
    @test built == Dict("initialisation" => Dict("peak_at_start" => Dict{String, Any}()))
    @test from_dict(InitialisationRun, built).initialisation == PeakAtStartInitialisation()
    @test option_alias(SlowModeInitialisation()) == "slow_mode"

    @test validate_initialisation(SlowModeInitialisation()) == SlowModeInitialisation()
    @test validate_initialisation(PeakAtStartInitialisation()) == PeakAtStartInitialisation()
    @test_throws ArgumentError validate_initialisation(SlowModeInitialisation(boundary_low = 0.8))
    @test_throws ArgumentError validate_initialisation(SlowModeInitialisation(boundary_high = 1.2))
    @test_throws ArgumentError validate_initialisation(SlowModeInitialisation(historical_multiplier_logsd = -1.0))
    @test_throws ArgumentError validate_initialisation(SlowModeInitialisation(historical_multiplier_logsd = Inf))
end

@testset "integration stability guard" begin
    durations = Durations(latent = 1.0, infectious = 1.0)
    bounds = IntegrationBounds(durations; R_eff_max = 0.0)
    limit = stability_limit(Rk4)
    err = try
        assert_integration_stable(bounds, limit, 1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("at least 2 (currently 1)", sprint(showerror, err))
    @test assert_integration_stable(bounds, limit, 2) < limit

    # A rate outside the shared durations raises the bound, and so the substeps needed.
    base = IntegrationBounds(Durations(); n_E = 2, n_I = 2, R_eff_max = 3.0)
    fast = IntegrationBounds(Durations(); n_E = 2, n_I = 2, R_eff_max = 3.0, extra_rates = (20.0,))
    @test max_transition_rate(fast) == 20.0 > max_transition_rate(base)
    @test required_supersample(fast, 7.0) == ceil(Int, 20.0 * 7.0 * STABILITY_HEADROOM / limit) > required_supersample(base, 7.0)
    @test_logs (:warn, r"within 1.4x") assert_integration_stable(fast, 7.0, ceil(Int, 20.0 * 7.0 / 2.5))
    @test_throws ArgumentError IntegrationBounds(Durations(); extra_rates = (-1.0,))
    @test_throws ArgumentError IntegrationBounds(Durations(); n_E = 0)
    @test_throws ArgumentError IntegrationBounds(Durations(); R_eff_max = -1.0)
    @test_throws ArgumentError assert_integration_stable(base, 7.0, 0)

    # Each integrator is held to its own limit; one without a known limit needs it given.
    @test stability_limit(ForwardEuler) == stability_limit(Heun) == 2.0 < stability_limit(Rk3) < limit
    @test (@test_logs (:warn, r"within 1.4x") assert_integration_stable(fast, 7.0, 56)) == 2.5
    @test_throws ArgumentError assert_integration_stable(fast, 7.0, 56; integrator = Heun)
    @test required_supersample(fast, 7.0; integrator = Heun) == ceil(Int, 20.0 * 7.0 * STABILITY_HEADROOM / 2.0)
    # The implicit trapezoidal rule is A-stable, so it has no limit to know.
    trapezoidal = ConfigurableEpi.SeeToDee.Trapezoidal
    @test stability_limit(trapezoidal) === nothing
    @test_throws ArgumentError assert_integration_stable(fast, 7.0, 56; integrator = trapezoidal)
    @test (@test_logs assert_integration_stable(fast, 7.0, 56; integrator = trapezoidal, limit = Inf)) == 2.5
    @test_throws TypeError assert_integration_stable(fast, 7.0, 56; integrator = (f, h) -> f)
    @test_throws ArgumentError assert_integration_stable(fast, 7.0, 56; limit = 0.0)
end

@testset "Erlang chain fill" begin
    @test chain_stage_seed(3.0, 0.0, 0.5, 4) == fill(6.0, 4)
    @test chain_stage_seed(3.0, 0.1, 1.0, 1) == [3.0 / 1.1]
    @test chain_stage_seed(3.0, 0.1, 1.0, 3)[3] ≈ 3.0 / 1.1 * (1 / 1.1)^2
    @test issorted(chain_stage_seed(3.0, 0.1, 1.0, 5); rev = true)
    @test isempty(chain_stage_seed(3.0, 0.1, 1.0, 0))
    @test isempty(chain_stage_seed(3.0, -1.0, 1.0, 0))   # an empty chain constrains no decline
    @test_throws ArgumentError chain_stage_seed(3.0, -1.0, 1.0, 2)
    @test_throws ArgumentError chain_stage_seed(3.0, 0.0, 0.0, 2)
    @test_throws ArgumentError chain_stage_seed(3.0, 0.0, 1.0, -1)
end

@testset "Slow-mode seed" begin
    dur = Durations(immunity = 180.0)
    cfg = SlowModeInitialisation()
    context(history; population, bin_days, n_E = 1, n_I = 1, n_obs = 2) =
        SeedContext(; durations = dur, population, bin_days, ascertainment = 0.2, history, n_E, n_I, n_obs)

    seed = initial_seed(cfg, context((times = [-20.0, -10.0], counts = [10.0, 10.0]); population = 100.0, bin_days = 10.0))
    rate = 10 / (10 * 100)
    kernel = rate * dur.immunity * (1 - exp(-30 / dur.immunity))
    q = exp(-30 / dur.immunity)
    @test seed.kernel ≈ kernel
    @test seed.boundary_survival ≈ q
    @test seed.susceptible_fraction ≈ 1 - 0.5q - kernel / 0.2 * exp(cfg.historical_multiplier_logsd^2 / 2)
    @test seed.susceptible_variance > 0
    @test seed.growth_rate == 0
    @test seed.first_observation_accumulator ≈ 10 / 0.2
    @test only(seed.exposed) ≈ (10 / (10 * 0.2)) * dur.latent
    @test only(seed.infectious) ≈ (10 / (10 * 0.2)) * dur.infectious
    @test all(seed.exposed .> 0) && all(seed.infectious .> 0) && all(seed.observation .> 0)

    # A gap between bins holds the earlier rate through it.
    gapped = initial_seed(cfg, context((times = [-30.0, -10.0], counts = [10.0, 10.0]); population = 100.0, bin_days = 10.0))
    @test gapped.kernel ≈ rate * dur.immunity * (1 - exp(-40 / dur.immunity))

    rising = (times = [-21.0, -14.0, -7.0], counts = [20.0, 40.0, 80.0])
    growing = initial_seed(cfg, context(rising; population = 10_000.0, bin_days = 7.0))
    flat = initial_seed(cfg, context((times = rising.times, counts = fill(80.0, 3)); population = 10_000.0, bin_days = 7.0))
    @test growing.growth_rate ≈ log(2) / 7
    @test sum(growing.exposed) > sum(flat.exposed)
    staged = initial_seed(cfg, context(rising; population = 10_000.0, bin_days = 7.0, n_E = 2, n_I = 2, n_obs = 3))
    g = staged.growth_rate
    rho = inv(dur.obs_progression)
    report_rate = (80 / 7) * (g * 7 / (-expm1(-g * 7))) * exp(7g)
    infection_rate = report_rate / 0.2 * (1 + g / rho)^2
    @test staged.infection_rate ≈ infection_rate
    @test staged.exposed[1] ≈ infection_rate / (2 / dur.latent + g)
    @test staged.observation[2] ≈ infection_rate / (rho + g) * rho / (rho + g)
    @test staged.observation == chain_stage_seed(staged.infection_rate, g, rho, 2)
    @test staged.exposed == chain_stage_seed(staged.infection_rate, g, 2 / dur.latent, 2)

    # A decline faster than a slow reporting chain allows is fine when there is no chain to fill.
    falling = (times = [-21.0, -14.0, -7.0], counts = 80 .* exp.(-0.08 .* [0.0, 7.0, 14.0]))
    slow_chain(n_obs) = SeedContext(;
        durations = Durations(obs_progression = 20.0), population = 10_000.0, bin_days = 7.0, ascertainment = 0.2,
        history = falling, n_obs,
    )
    declining = initial_seed(cfg, slow_chain(1))
    @test declining.growth_rate ≈ -0.08
    @test isempty(declining.observation)
    @test_throws ArgumentError initial_seed(cfg, slow_chain(2))

    @test_throws ArgumentError initial_seed(cfg, context((times = [0.0], counts = [10.0]); population = 100.0, bin_days = 7.0))
    @test_throws ArgumentError initial_seed(cfg, context((times = [-14.0, -10.0], counts = [1.0, 1.0]); population = 100.0, bin_days = 7.0))
    @test_throws ArgumentError initial_seed(cfg, context((times = [-10.0, -20.0], counts = [1.0, 1.0]); population = 100.0, bin_days = 7.0))
    @test_throws DimensionMismatch initial_seed(cfg, context((times = [-10.0], counts = [1.0, 1.0]); population = 100.0, bin_days = 7.0))
    @test_throws ArgumentError initial_seed(cfg, context(nothing; population = 100.0, bin_days = 7.0))
    over = SeedContext(; durations = dur, population = 100.0, bin_days = 7.0, ascertainment = 2.0, history = rising)
    @test_throws ArgumentError initial_seed(cfg, over)
end

@testset "Peak seeds invert the first count into a flat stream" begin
    dur = Durations(latent = 2.0, infectious = 1.5, immunity = 180.0, obs_progression = 2.2)
    history = (times = Float64.(0:7:98), counts = [1.0, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2, 1, 1, 1])
    defaults = (; durations = dur, population = 1.0e6, bin_days = 7.0, ascertainment = 0.005, first_count = 14.0, R0 = 2.0, n_obs = 4)
    context(; kwargs...) = SeedContext(; merge(defaults, kwargs)...)
    recon = initial_infection_state(14.0, dur, 0.005, 7.0)

    at_start = initial_seed(PeakAtStartInitialisation(), context(chi_at = t -> 1.25))
    @test at_start.susceptible_fraction ≈ 1 / (2.0 * 1.25)
    @test only(at_start.exposed) ≈ recon.exposed
    @test only(at_start.infectious) ≈ recon.infectious
    @test at_start.observation ≈ fill(recon.obs_stage, 3)
    @test at_start.infection_rate ≈ recon.daily_incidence
    @test at_start.growth_rate == 0
    @test at_start.first_observation_accumulator ≈ 14.0 / 0.005

    from_history = initial_seed(PeakFromHistoryInitialisation(), context(; history))
    anchor = peak_anchored_susceptible_fraction(history, dur, 4, 0.005, 1.0e6, 7.0, Returns(1.0), 2.0)
    @test from_history.anchor == anchor
    @test from_history.susceptible_fraction == anchor.susceptible_fraction
    @test from_history.exposed == at_start.exposed
    # No history, or an inexact anchor, falls back to 1 / R0.
    @test initial_seed(PeakFromHistoryInitialisation(), context()).susceptible_fraction == 0.5
    inexact = initial_seed(PeakFromHistoryInitialisation(), context(; history, n_E = 2, n_I = 2))
    @test inexact.anchor === nothing && inexact.susceptible_fraction == 0.5
    @test length(inexact.exposed) == 2 && sum(inexact.exposed) ≈ recon.exposed

    @test_throws "first_count" initial_seed(PeakAtStartInitialisation(), context(first_count = nothing))
    @test_throws "R0" initial_seed(PeakFromHistoryInitialisation(), context(R0 = nothing))
    @test_throws ArgumentError context(population = 0.0)
    @test_throws ArgumentError context(n_obs = 0)
end

@testset "Peak-anchored susceptible fraction" begin
    dur = Durations(latent = 2.0, infectious = 1.5, immunity = 180.0, obs_progression = 2.2)

    @testset "Erlang-aware prevalence lag" begin
        total = dur.latent + dur.infectious
        expected = (total^2 + dur.latent^2 / 4 + dur.infectious^2 / 3) / (2 * total)
        @test prevalence_peak_lag_days(dur, 4, 3) ≈ expected
        @test prevalence_peak_lag_days(dur) == prevalence_peak_lag_days(dur, 1, 1)
        @test prevalence_peak_lag_days(dur, 4, 3) < prevalence_peak_lag_days(dur)
        @test_throws ArgumentError prevalence_peak_lag_days(dur, 0, 1)
    end

    @testset "the susceptible carry lands exactly on fractional targets" begin
        times = [0.0, 1.0]
        incidence = [0.1, 0.1]
        @test carry_susceptible(0.8, 0.0, 0.0, times, incidence, 0.0) == 0.8
        @test carry_susceptible(0.8, 0.0, 0.25, times, incidence, 0.0) ≈ 0.775
        @test carry_susceptible(0.8, 0.25, 0.0, times, incidence, 0.0) ≈ 0.825
    end

    @testset "peak timing uses strict maxima and bin midpoints" begin
        @test find_observed_peak([1.0, 5.0, 5.0, 1.0]; window = 1) === nothing
        @test find_observed_peak([1.0, 5.0, 4.0, 1.0]; window = 1) == 2

        history = (times = Float64.(0:7:98), counts = [1.0, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2, 1, 1, 1])
        ascertainment, pop, weekly_dt = 0.005, 1.0e6, 7.0
        result = peak_anchored_susceptible_fraction(history, dur, 4, ascertainment, pop, weekly_dt, _ -> 1.0, 2.0)
        peak = find_observed_peak(history.counts)
        delay = reporting_delay_days(dur, 4)
        lag = prevalence_peak_lag_days(dur)
        @test result.anchor_time ≈ history.times[peak] - weekly_dt / 2 - delay + lag
        incidence_times = history.times .- weekly_dt / 2 .- delay
        incidence = history.counts ./ (ascertainment * weekly_dt * pop)
        @test result.susceptible_fraction ≈
            carry_susceptible(result.anchor_fraction, result.anchor_time, 0.0, incidence_times, incidence, 1 / dur.immunity)
        @test peak_anchored_susceptible_fraction(history, dur, 4, 0.005, 1.0e6, 7.0, _ -> 1.0, 2.0; n_E = 2, n_I = 2) === nothing
    end
end
