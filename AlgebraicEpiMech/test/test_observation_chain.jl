# `observation_layout` — the READ side of observation chains.
#
# Given a net that already carries chains, recover which species belong to which chain and in what
# order. The construction side it used to accompany (`WithObservation`, `ObservationChainSpec`)
# has been removed: observation is attached to a COMPOSED net by pushout, so these nets are built
# with `attach_observation`.
#
# The tests that used to live here guarding the operadic path — "obs_inflow preserved through
# typed_product with NoCrossImmunity / CompleteCrossImmunity / ImmuneHistory", and the two
# behavioural accumulator-fills checks — are gone rather than ported. The property they asserted
# (that a factor's reflexive `:observation` box survives the pullback) no longer exists, and the
# behaviour they checked is covered directly by `test_observation_rewriting.jl`.

@testsnippet ObservationLayoutSetup begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri: typed_product
    using Catlab
    using AlgebraicEpiMech

    one_pop_typing = OnePopulationTyping(population_type = :Individual)
end

@testitem "observation_layout returns ordered chain metadata" setup = [ObservationLayoutSetup] begin
    pn = attach_observation(
        dom(create_model(one_pop_typing, SEIR())),
        AtCompartment(:I); n_stages = 2,
    )

    layout = observation_layout(pn)

    @test layout.obs_names == (:O_I_1, :O_I_2)
    @test length(layout.chains) == 1
    @test layout.chains[1].source_name == :I
    @test layout.chains[1].obs_names == (:O_I_1, :O_I_2)
    @test layout.chains[1].cumulative_name == :O_I_2
    @test layout.cumulative_names == (:O_I_2,)
end

@testitem "observation_layout separates chains on different sources" setup = [ObservationLayoutSetup] begin
    # Two chains, attached one after another. Each is named for the source it samples, which is
    # what keeps them apart — a chain naming scheme that dropped the source would collapse both
    # onto one label.
    pn = attach_observation(
        attach_observation(
            dom(create_model(one_pop_typing, SEIRS())),
            AtCompartment(:I); n_stages = 2,
        ),
        AtCompartment(:R); n_stages = 1,
    )

    layout = observation_layout(pn)

    @test Set(layout.obs_names) == Set((:O_I_1, :O_I_2, :O_R_1))
    @test Set(chain.source_name for chain in layout.chains) == Set((:I, :R))
    @test Set(layout.cumulative_names) == Set((:O_I_2, :O_R_1))

    chains = Dict(chain.source_name => chain for chain in layout.chains)
    @test chains[:I].obs_names == (:O_I_1, :O_I_2)
    @test chains[:R].obs_names == (:O_R_1,)
end

@testitem "observation_layout resolves stratified chains per stratum" setup = [ObservationLayoutSetup] begin
    # Composition FIRST, observation after — so the stratification factor carries no observation
    # structure at all and each location's chain is created by the rewrite.
    composed = dom(
        typed_product(
            create_model(one_pop_typing, SEIR()),
            create_model(one_pop_typing, GeographicStratification([:ak, :ca])),
        ),
    )
    pn = attach_observation(composed, AtCompartment(:I); n_stages = 2)

    layout = observation_layout(pn)

    @test length(layout.chains) == 2
    chains = Dict(chain.source_name => chain for chain in layout.chains)
    @test Set(keys(chains)) == Set((:I_ak, :I_ca))
    @test Set(layout.cumulative_names) == Set((:O_I_2_ak, :O_I_2_ca))
    for loc in (:ak, :ca)
        @test chains[Symbol(:I_, loc)].obs_names ==
            (Symbol(:O_I_1_, loc), Symbol(:O_I_2_, loc))
    end
end
