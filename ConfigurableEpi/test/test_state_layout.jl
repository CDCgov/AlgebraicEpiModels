using Test
using ConfigurableEpi
using AlgebraicEpiMech
using Catlab: dom

@testset "StateLayout" begin
    @testset "Basic construction" begin
        core = (:S, :I, :R)
        obs = (:O_I_1, :O_I_2, :O_I_cum)
        latent = (:Rt_unc,)

        layout = StateLayout(core, obs, latent)

        @test layout.core_names == core
        @test layout.obs_names == obs
        @test layout.latent_names == latent
        @test layout.signal_names == (:y,)
        @test n_signals(layout) == 1
    end

    @testset "Index ranges" begin
        core = (:S, :I, :R)
        obs = (:O_I_1, :O_I_2)
        latent = (:Rt_unc, :beta_unc)

        layout = StateLayout(core, obs, latent)

        @test layout.core_range == 1:3
        @test layout.obs_range == 4:5
        @test layout.latent_range == 6:7
        @test layout.total_dim == 7
    end

    @testset "Helper functions" begin
        core = (:S, :E, :I, :R)
        obs = (:O_I_1,)
        latent = (:Rt_unc,)

        layout = StateLayout(core, obs, latent)

        @test all_names(layout) == (:S, :E, :I, :R, :O_I_1, :Rt_unc)
        @test ode_names(layout) == (:S, :E, :I, :R, :O_I_1)
        @test n_ode_states(layout) == 5
        @test n_latent(layout) == 1
    end

    @testset "Accumulator indices" begin
        core = (:S, :I, :R)
        obs = (:O_I_1, :O_I_2, :O_I_cum)
        latent = (:Rt_unc,)

        layout = StateLayout(core, obs, latent)

        @test layout.accumulator_indices == (6,)
        @test layout.accumulator_indices[1] == 6
    end

    @testset "No latent states" begin
        core = (:S, :I, :R)
        obs = (:O_I_1,)
        latent = ()

        layout = StateLayout(core, obs, latent)

        @test n_latent(layout) == 0
        @test layout.latent_range == 5:4
        @test layout.total_dim == 4
    end

    @testset "No observation signals" begin
        layout = StateLayout((:S, :I, :R), (), (:Rt,); signal_names = ())

        @test layout.obs_names == ()
        @test layout.signal_names == ()
        @test layout.accumulator_indices == ()
        @test layout.total_dim == 4
    end

    @testset "Observation signals require observation states" begin
        @test_throws ArgumentError StateLayout((:S, :I, :R), (), (:Rt,))
        @test_throws ArgumentError StateLayout(
            (:S, :I, :R), (), (:Rt,); signal_names = (:hosp,)
        )
    end

    @testset "extract_latent" begin
        core = (:S, :I, :R)
        obs = (:O_I_1,)
        latent = (:Rt_unc, :rho_unc)

        layout = StateLayout(core, obs, latent)
        state = [100.0, 10.0, 5.0, 2.0, 0.7, -0.5]

        extracted = extract_latent(state, layout)

        @test extracted isa NamedTuple
        @test keys(extracted) == (:Rt_unc, :rho_unc)
        @test extracted.Rt_unc ≈ 0.7
        @test extracted.rho_unc ≈ -0.5
    end

    @testset "Petri net constructor infers observation layout" begin
        typing = OnePopulationTyping()
        # Observation attached to the built net by pushout. `AtCompartment` samples the
        # compartment, which yields the same `O_I_1`/`O_I_2` names the operadic chain did.
        pn = attach_observation(
            dom(create_model(typing, SEIR())), AtCompartment(:I); n_stages = 2
        )
        latent_specs = (
            RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.1)),
        )

        layout = StateLayout(pn, latent_specs; signal_names = (:hosp,))

        @test Set(layout.core_names) == Set((:S, :E, :I, :R))
        @test layout.obs_names == (:O_I_1, :O_I_2)
        @test layout.signal_names == (:hosp,)
        @test layout.accumulator_indices == (6,)
    end

    @testset "Petri net constructor supports no observation chains" begin
        typing = OnePopulationTyping()
        pn = dom(create_model(typing, SEIR()))
        latent_specs = (
            RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = HyperParam(positive_gaussian(:sigma_Rt, 0.1, 0.05))),
        )

        layout = StateLayout(pn, latent_specs; signal_names = ())

        @test Set(layout.core_names) == Set((:S, :E, :I, :R))
        @test layout.obs_names == ()
        @test layout.signal_names == ()
        @test layout.accumulator_indices == ()
    end
end
