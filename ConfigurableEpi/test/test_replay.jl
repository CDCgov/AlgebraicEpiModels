using Test
using ConfigurableEpi
using AlgebraicEpiMech
using AlgebraicPetri
using AlgebraicPetri.TypedPetri: typed_product
using Catlab: dom
using ConfigurableEpi: Rk4

# S and R with waning only; infection is the replay's to supply.
function _waning_vf!(du, u, p, t)
    hyper, _ = p
    du[:S] = hyper.omega * u[:R]
    du[:R] = -hyper.omega * u[:R]
    return nothing
end

# Two strains with full immune history, waning and importation, one infection accumulator per strain.
function _two_strain_net()
    typing = UninfectedInfectedTyping()
    disease = create_model(typing, SEIRS(importation = true); include_reflexives = (:waning,))
    history = create_model(
        typing, ImmuneHistory([:A, :B]; waning = true); include_reflexives = (:disease, :waning, :importation),
    )
    by_strain(label) = (label[1], Symbol(first(split(string(label[2]), "_from_"))))
    return attach_observation(
        dom(typed_product(disease, history)), AtEvent((:transmission, :importation); group = by_strain); n_stages = 1,
    )
end
_infection(name) = match(r"(?:infect|import)_([AB])_(.+?)(?:_by_.+)?$", string(name))
_strain(name) = Symbol(_infection(name)[1])
_cross(name) = _infection(name)[2] == "naive" ? 1.0 : 0.6

function _two_strain_rates(pn, population)
    beta = (A = 0.5, B = 0.42)
    prefixes = ("E_to_I" => 0.5, "I_to_R" => 1 / 3, "R_to_S" => 1 / 20, "waning_S_wane_A" => 1 / 200, "waning_S_wane_B" => 1 / 400)
    rate(s) = startswith(s, "transmission") ? beta[_strain(s)] * _cross(s) / population :
        startswith(s, "importation") ? 1.0e-6 : last(prefixes[findfirst(p -> startswith(s, first(p)), prefixes)])
    return NamedTuple(name => rate(string(name)) for name in flatten_symbols.(tnames(pn)))
end

@testset "Replay initialisation" begin
    @testset "BinnedIncidence" begin
        inc = BinnedIncidence([-14.0, -7.0, 0.0], [3.0 5.0])
        rate(t) = ConfigurableEpi._incidence_at(inc, 1, t)
        @test rate(-14.0) == 3.0 && rate(-7.5) == 3.0
        @test rate(-7.0) == 5.0 && rate(-0.1) == 5.0
        @test rate(-15.0) == 0.0 && rate(0.0) == 0.0
        @test_throws ArgumentError BinnedIncidence([0.0], zeros(1, 0))
        @test_throws ArgumentError BinnedIncidence([0.0, 0.0], zeros(1, 1))
        @test_throws DimensionMismatch BinnedIncidence([0.0, 1.0], zeros(1, 2))
        @test_throws ArgumentError BinnedIncidence([0.0, 1.0], fill(-1.0, 1, 1))
    end

    @testset "a single waning class reproduces the slow-mode seed" begin
        dur = Durations(immunity = 150.0)
        population, alpha0, boundary = 1.0e6, 0.02, 0.6
        history = (; times = collect(-70.0:7.0:-7.0), counts = [40, 90, 150, 260, 310, 280, 200, 170, 120, 100])
        layout = StateLayout((:S, :R), (), (); signal_names = ())
        model = ReplayModel(_waning_vf!, _waning_vf!, layout, InfectionRoutes([:all], [1], [[2]], [1], [1.0]))
        rates = history.counts ./ (7 * alpha0)
        incidence = BinnedIncidence([-77.0; history.times; 0.0], reshape([rates; last(rates)], 1, :))
        x_start = [boundary, 1 - boundary] .* population
        hyper = (; omega = 1 / dur.immunity)
        replay_at(scale) = replay_state(model, incidence, x_start, hyper, _ -> (;), -77.0, 0.0; scale)

        ctx = SeedContext(; durations = dur, population, bin_days = 7.0, ascertainment = alpha0, history, n_obs = 2)
        slow(logsd) = initial_seed(
            SlowModeInitialisation(boundary_low = boundary, boundary_high = boundary, historical_multiplier_logsd = logsd), ctx,
        )
        x = replay_at(1.0)
        @test sum(x) ≈ population
        by_euler = replay_state(model, incidence, x_start, hyper, _ -> (;), -77.0, 0.0; integrator = ConfigurableEpi.ForwardEuler)
        @test sum(by_euler) ≈ population
        @test by_euler[1] ≈ x[1] rtol = 1.0e-3
        @test x[1] / population ≈ slow(0.0).susceptible_fraction rtol = 1.0e-10

        logsd = 0.3
        moments = replay_scale_moments(replay_at, logsd)
        @test moments.mean[1] / population ≈ slow(logsd).susceptible_fraction rtol = 1.0e-10
        @test moments.covariance[1, 1] / population^2 ≈ slow(logsd).susceptible_variance rtol = 1.0e-6
        # Whoever is not susceptible is recovered.
        @test moments.covariance[1, 2] ≈ -moments.covariance[1, 1]
        @test replay_scale_moments(replay_at, 0.0).covariance == zeros(2, 2)
        @test_throws ArgumentError replay_scale_moments(replay_at, -0.1)
        @test_throws ArgumentError replay_at(-1.0)
    end

    pn = _two_strain_net()
    layout = StateLayout(pn, ())
    names = collect(ode_names(layout))
    at(name) = findfirst(==(name), names)
    cells = [Symbol(last(split(string(names[i]), "_"))) for i in layout.accumulator_indices]

    @testset "infection routes are read off the net" begin
        @test sort(cells) == [:A, :B]
        routes = infection_routes(pn, layout; cells, cell = _strain, weight = _cross)
        # Transmission and importation into the same class are one route.
        @test length(routes.source) == 4
        @test sort(names[routes.source]) == [:S_U_A, :S_U_B, :S_U_naive, :S_U_naive]
        for k in eachindex(routes.source)
            strain = routes.cells[routes.cell[k]]
            from = chopprefix(string(names[routes.source[k]]), "S_U_")
            @test Set(names[routes.targets[k]]) == Set([Symbol(:E_, strain, :_from_, from), Symbol(:O_transmission_1_, strain)])
            @test routes.weight[k] == (from == "naive" ? 1.0 : 0.6)
        end

        @test_throws ArgumentError infection_routes(pn, layout; cells = [:A, :B, :C], cell = _strain)
        @test_throws ArgumentError infection_routes(pn, layout; cells = [:A], cell = _strain)
        @test_throws ArgumentError infection_routes(pn, layout; cells = [:A, :A], cell = _strain)
        @test_throws ArgumentError infection_routes(pn, layout; cells, cell = _strain, weight = _ -> -1.0)
        @test_throws ArgumentError infection_routes(pn, layout; cells, cell = _strain, is_infection = _ -> false)
        # A cell or weight keyed on the infector splits one route in two.
        @test_throws ArgumentError infection_routes(
            pn, layout; cells, cell = _strain, weight = n -> endswith(string(n), "by_naive") ? 1.0 : 0.5,
        )
    end

    @testset "replaying a simulation's own incidence recovers its state" begin
        population, dt, n_steps = 1.0e5, 0.25, 1600
        defaults = _two_strain_rates(pn, population)
        no_rates(latent, hyper, t) = (;)
        model = ReplayModel(pn, layout, no_rates; defaults, cells, cell = _strain, weight = _cross)
        x0 = zeros(length(names))
        x0[at(:S_U_naive)] = population - 30
        x0[at(:E_A_from_naive)] = 20
        x0[at(:E_B_from_naive)] = 10

        full = build_unified_vf(model.vf!, layout)
        step = Rk4((x, u, p, t) -> full(x, u, (p, (;)), t), dt; supersample = 2)
        accumulators = collect(layout.accumulator_indices)
        x = copy(x0)
        infections = zeros(2, n_steps)
        for k in 1:n_steps
            x[accumulators] .= 0
            x = Vector(step(x, nothing, nothing, (k - 1) * dt))
            infections[:, k] = x[accumulators]
        end
        # Both strains circulated.
        @test all(sum(infections; dims = 2) .> 0.1 * population)

        span = n_steps * dt
        replay(incidence; kwargs...) = replay_state(model, incidence, x0, nothing, _ -> (;), 0.0, span; dt, kwargs...)
        exact = replay(BinnedIncidence(collect(0:n_steps) .* dt, infections ./ dt); keep_accumulators = true)
        @test sum(exact[layout.core_range]) ≈ population
        @test maximum(abs, exact .- x) / population < 5.0e-5
        @test exact[accumulators] ≈ infections[:, end] rtol = 1.0e-10

        # Weekly totals still place the immune classes, the slow part of the state.
        per_week = round(Int, 7 / dt)
        n_weeks, tail = divrem(n_steps, per_week)
        weekly = hcat((sum(infections[:, ((w - 1) * per_week + 1):(w * per_week)]; dims = 2) ./ 7 for w in 1:n_weeks)...)
        last_days = sum(infections[:, (n_weeks * per_week + 1):end]; dims = 2) ./ (tail * dt)
        coarse = replay(BinnedIncidence([collect(0:n_weeks) .* 7.0; span], [weekly last_days]))
        susceptible = [at(n) for n in names if startswith(string(n), "S_U_")]
        @test maximum(abs, coarse[susceptible] .- x[susceptible]) / population < 5.0e-3

        @test_throws ArgumentError replay(BinnedIncidence([0.0, span], fill(population, 2, 1)))
        @test_throws ArgumentError replay_state(model, BinnedIncidence([0.0, 1.0], zeros(2, 1)), x0, nothing, _ -> (;), 0.0, 0.3)
        @test_throws DimensionMismatch replay(BinnedIncidence([0.0, span], zeros(3, 1)))
        @test_throws DimensionMismatch replay_state(model, BinnedIncidence([0.0, 1.0], zeros(2, 1)), x0[2:end], nothing, _ -> (;), 0.0, 1.0)
    end

    @testset "an infection rate key shared with another transition cannot be switched off" begin
        function shared(name)
            s = string(flatten_symbols(name))
            return (startswith(s, "transmission") || startswith(s, "E_to_I") ? :mixed : Symbol(s), 1.0)
        end
        @test_throws ArgumentError build_replay_vf(pn, (latent, hyper, t) -> (;); rate_class = shared)
    end
end
