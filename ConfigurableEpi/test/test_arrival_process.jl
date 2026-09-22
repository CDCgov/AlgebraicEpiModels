using Test
using ConfigurableEpi
import Random

const CE = ConfigurableEpi

# A minimal SI-with-observation model: transmission scales with R0_baseline, O_I_1 accumulates
# incidence. The arrival seeds I from S.
function arrival_mock_petri_vf!(du, u, p, t)
    hyper, latent = p
    beta = hyper.R0_baseline * latent.Rt * 0.3 / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

function arrival_test_setup()
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.02))
    process = ArrivalProcess(
        :invader;
        rate = (_x, _latent, p, _t) -> p.arrival_rate,   # ungated ⇒ recurring
        mark = (_p, _rng) -> (import_size = 20.0,),      # deterministic mark for exact assertions
        transition! = seed_transition(ode_names(layout), :S, :I; size_key = :import_size),
    )
    driver_specs = (Rt_spec, process)
    stochastic = build_stochastic_update(layout, driver_specs)
    return (; layout, Rt_spec, process, driver_specs, stochastic)
end

@testset "Arrival process" begin
    (; layout, Rt_spec, process, driver_specs, stochastic) = arrival_test_setup()

    @testset "driver traits: state-carrying vs stateless" begin
        # The layout partitions on `carries_state` (the Markov question), NOT on how the process
        # evolves. An ArrivalProcess is stateless — its mark is absorbed into the compartments.
        @test carries_state(Rt_spec)
        @test !carries_state(process)
        # …so declaring the arrival adds NO slots: the layout is unchanged by its presence.
        @test layout.total_dim == 4                       # [S, I, O_I_1, Rt] — nothing for the arrival
        @test layout.latent_names == (:Rt,)
        @test build_stochastic_update(layout, (Rt_spec,)) isa CE.StochasticUpdate
        @test stochastic isa CE.StochasticUpdate          # same layout serves both driver lists
    end

    @testset "UKF-compatibility guard (separate from carries_state)" begin
        @test supports_gaussian_filter(Rt_spec)
        @test !supports_gaussian_filter(process)
        @test assert_gaussian_filter_compatible((Rt_spec,)) === nothing
        @test_throws ArgumentError assert_gaussian_filter_compatible((Rt_spec, process))
        @test_throws ArgumentError assert_gaussian_filter_compatible(process)   # single spec
    end

    @testset "step_arrival_probability" begin
        @test step_arrival_probability(0.0, 1.0) == 0.0
        @test step_arrival_probability(log(2), 1.0) ≈ 0.5
        @test step_arrival_probability(1.0, 7.0) ≈ -expm1(-7.0)
        @test step_arrival_probability(1.0e6, 1.0) ≈ 1.0        # saturates
        @test_throws ArgumentError step_arrival_probability(-1.0, 1.0)
        @test_throws ArgumentError step_arrival_probability(Inf, 1.0)
        @test_throws ArgumentError step_arrival_probability(1.0, 0.0)
    end

    @testset "beta_mark sampler" begin
        sampler = beta_mark(mean_key = :m, concentration = 2.0, out_key = :v)
        draw = sampler((m = 0.4,), Random.MersenneTwister(11))
        @test keys(draw) == (:v,)
        @test 0 < draw.v < 1
        @test sampler((m = 0.4,), Random.MersenneTwister(11)).v == draw.v   # deterministic given rng
        @test_throws ArgumentError sampler((m = 1.5,), Random.MersenneTwister(1))
        @test_throws ArgumentError beta_mark(concentration = 0.0)
    end

    @testset "seed_transition is mass-conserving and capped" begin
        trans! = seed_transition((:S, :I, :O), :S, :I; size_key = :sz)
        x = Float64[100, 10, 0]
        trans!(x, (sz = 30.0,), nothing)
        @test x == Float64[70, 40, 0]                          # 30 moved S→I, mass conserved

        x2 = Float64[100, 10, 0]
        trans!(x2, (sz = 200.0,), nothing)                     # request exceeds available
        @test x2 == Float64[0, 110, 0]                         # capped at 100

        x3 = Float64[100, 10, 0]
        trans!(x3, (sz = -5.0,), nothing)
        @test x3 == Float64[100, 10, 0]                        # negative request moves nothing

        @test_throws ArgumentError seed_transition((:S, :I), :X, :I; size_key = :sz)
    end

    @testset "advance_arrival! — the stateless jump step" begin
        latent = (Rt = 1.0,)
        # Fires (huge rate) ⇒ seeds exactly, mass-conserving, absorbed into the compartments.
        xm = [100.0, 10.0, 0.0]
        CE.advance_arrival!(
            process, view(xm, 1:3), latent, (arrival_rate = 1.0e6,), 1.0, 0.0,
            Random.MersenneTwister(1),
        )
        @test xm == [80.0, 30.0, 0.0]

        # Zero intensity ⇒ no draw, no seed.
        xm0 = [100.0, 10.0, 0.0]
        CE.advance_arrival!(
            process, view(xm0, 1:3), latent, (arrival_rate = 0.0,), 1.0, 0.0,
            Random.MersenneTwister(1),
        )
        @test xm0 == [100.0, 10.0, 0.0]

        # SELF-EXCITATION with no stored history: the intensity reads the very state its jump seeds.
        # Here: don't fire once I has already been seeded past a threshold.
        gated = ArrivalProcess(
            :invader;
            rate = (x, _l, p, _t) -> x[2] > 25.0 ? 0.0 : p.arrival_rate,
            mark = (_p, _r) -> (import_size = 20.0,),
            transition! = seed_transition((:S, :I, :O), :S, :I; size_key = :import_size),
        )
        xg = [100.0, 10.0, 0.0]                       # I = 10 ⇒ below the gate, fires once → I = 30
        CE.advance_arrival!(
            gated, view(xg, 1:3), latent, (arrival_rate = 1.0e6,), 1.0, 0.0, Random.MersenneTwister(2)
        )
        @test xg == [80.0, 30.0, 0.0]
        CE.advance_arrival!(                          # now I = 30 > 25 ⇒ intensity 0, self-extinguished
            gated, view(xg, 1:3), latent, (arrival_rate = 1.0e6,), 1.0, 0.0, Random.MersenneTwister(2)
        )
        @test xg == [80.0, 30.0, 0.0]                 # unchanged — no second seed
    end

    @testset "build_pf_dynamics with an arrival driver" begin
        p = (R0_baseline = 2.0, obs_scale = 0.2, arrival_rate = 1.0e6)   # huge rate ⇒ fires ~surely
        Rt_unc = stochastic.to_unconstrained((Rt = 1.0,))[1]
        x = [900.0, 50.0, 0.0, Rt_unc]                # [S, I, O_I_1, Rt] — no arrival block
        dynamics = build_full_dynamics(
            arrival_mock_petri_vf!, stochastic, layout; supersample = 2
        )

        @testset "noise = false never fires" begin
            dyn = build_pf_dynamics(
                dynamics, layout; rng = Random.MersenneTwister(1),
            )
            out = dyn(x, nothing, p, 0.0, false)
            @test length(out) == layout.total_dim     # the stateless arrival appends nothing
        end

        @testset "noise = true fires: seeds pre-flow, so the seed grows within the step" begin
            # Identically-seeded rng ⇒ identical base propagation; the only difference between a
            # firing run and a rate-0 run is the jump.
            dyn_fire = build_pf_dynamics(
                dynamics, layout; rng = Random.MersenneTwister(7),
            )
            dyn_ref = build_pf_dynamics(
                dynamics, layout; rng = Random.MersenneTwister(7),
            )
            p0 = (R0_baseline = 2.0, obs_scale = 0.2, arrival_rate = 0.0)
            res_fire = dyn_fire(x, nothing, p, 0.0, true)
            res_ref = dyn_ref(x, nothing, p0, 0.0, true)

            @test length(res_fire) == layout.total_dim
            @test res_fire[2] > res_ref[2]            # seeded I grew during the step
            @test res_fire[1] < res_ref[1]            # S depleted by the seed
        end

        @testset "integrates with a learned Liu-West tail" begin
            learned = build_learned_hyperparams(positive_gaussian(:arrival_rate, 0.05, 0.02), layout)
            dyn = build_pf_dynamics(
                dynamics, layout;
                rng = Random.MersenneTwister(5), learned = learned,
            )
            rate_unc = learned.to_unconstrained((arrival_rate = 1.0e6,))[1]
            xl = [900.0, 50.0, 0.0, Rt_unc, rate_unc]         # [model(4); arrival_rate_unc]
            pl = (R0_baseline = 2.0, obs_scale = 0.2)          # arrival_rate comes from the learned tail
            out = dyn(xl, nothing, pl, 0.0, true)
            @test length(out) == layout.total_dim + 1          # model + learned tail
            @test out[5] == rate_unc                           # learned tail carried unchanged
            @test out[2] > 50.0                                # rate read from the tail ⇒ fired, I seeded+grew
        end
    end
end
