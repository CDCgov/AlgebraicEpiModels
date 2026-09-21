# Unit tests for create_model with ContactStratification and its convenience constructors
# This generalizes create_age_stratification.jl to work with all stratification types

@testsnippet ContactStratificationModelSetup begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri
    using Catlab
    using AlgebraicEpiMech

    # Create schema instances for testing
    one_pop_schema = OnePopulationSchema(population_type = :Individual)
    ui_schema = UninfectedInfectedSchema(
        uninfected_type = :Susceptible,
        infected_type = :Infectious
    )
end

# Test that create_model returns typed Petri net for ContactStratification
@testitem "create_model(ContactStratification) returns ACSetTransformation" setup = [ContactStratificationModelSetup] begin
    @testset "AgeStratification" begin
        age_strat = AgeStratification([:child, :adult])
        typed_model = create_model(one_pop_schema, age_strat)
        @test typed_model isa ACSetTransformation
    end

    @testset "GeographicStratification" begin
        geo_strat = GeographicStratification([:urban, :rural])
        typed_model = create_model(one_pop_schema, geo_strat)
        @test typed_model isa ACSetTransformation
    end

    @testset "Direct ContactStratification" begin
        risk_strat = ContactStratification([:low, :high], :risk)
        typed_model = create_model(one_pop_schema, risk_strat)
        @test typed_model isa ACSetTransformation
    end
end

@testitem "create_model(ContactStratification) typed model has dom as LabelledPetriNet" setup = [ContactStratificationModelSetup] begin
    strats = [
        AgeStratification([:child, :adult]),
        GeographicStratification([:urban, :rural]),
        ContactStratification([:low, :high], :risk),
    ]

    for strat in strats
        typed_model = create_model(one_pop_schema, strat)
        pn = dom(typed_model)
        @test pn isa LabelledPetriNet
    end
end

@testitem "create_model(ContactStratification) schema accessible via codom" setup = [ContactStratificationModelSetup] begin
    strats = [
        AgeStratification([:child, :adult]),
        GeographicStratification([:urban, :rural]),
        ContactStratification([:low, :high], :risk),
    ]

    for strat in strats
        typed_model = create_model(one_pop_schema, strat)
        schema_pn = codom(typed_model)
        @test schema_pn isa LabelledPetriNet
    end
end

# Test with OnePopulationSchema - generalized for any ContactStratification
@testitem "create_model with 2 strata OnePopulationSchema" setup = [ContactStratificationModelSetup] begin
    @testset "AgeStratification([:child, :adult])" begin
        strat = AgeStratification([:child, :adult])
        typed_model = create_model(one_pop_schema, strat)
        pn = dom(typed_model)

        # Should have 2 species (one per stratum)
        @test length(AlgebraicPetri.snames(pn)) == 2

        # Should have n² + 3n = 10 transitions (4 transmission + 6 reflexive)
        @test length(AlgebraicPetri.tnames(pn)) == 10

        # Check transmission names represent contact patterns
        tnames = AlgebraicPetri.tnames(pn)
        @test :child_child in tnames
        @test :child_adult in tnames
        @test :adult_child in tnames
        @test :adult_adult in tnames
    end

    @testset "GeographicStratification([:urban, :rural])" begin
        strat = GeographicStratification([:urban, :rural])
        typed_model = create_model(one_pop_schema, strat)
        pn = dom(typed_model)

        @test length(AlgebraicPetri.snames(pn)) == 2
        @test length(AlgebraicPetri.tnames(pn)) == 10

        tnames = AlgebraicPetri.tnames(pn)
        @test :urban_urban in tnames
        @test :urban_rural in tnames
        @test :rural_urban in tnames
        @test :rural_rural in tnames
    end

    @testset "ContactStratification([:low, :high], :risk)" begin
        strat = ContactStratification([:low, :high], :risk)
        typed_model = create_model(one_pop_schema, strat)
        pn = dom(typed_model)

        @test length(AlgebraicPetri.snames(pn)) == 2
        @test length(AlgebraicPetri.tnames(pn)) == 10

        tnames = AlgebraicPetri.tnames(pn)
        @test :low_low in tnames
        @test :low_high in tnames
        @test :high_low in tnames
        @test :high_high in tnames
    end
end

@testitem "create_model with 3 strata OnePopulationSchema" setup = [ContactStratificationModelSetup] begin
    strats_and_names = [
        (AgeStratification([:child, :adult, :elderly]), [:child, :adult, :elderly]),
        (
            GeographicStratification([:urban, :suburban, :rural]),
            [:urban, :suburban, :rural],
        ),
        (ContactStratification([:low, :medium, :high], :risk), [:low, :medium, :high]),
    ]

    for (strat, expected_names) in strats_and_names
        typed_model = create_model(one_pop_schema, strat)
        pn = dom(typed_model)

        # Should have 3 species
        @test length(AlgebraicPetri.snames(pn)) == 3

        # Should have n² + 3n = 18 transitions
        @test length(AlgebraicPetri.tnames(pn)) == 18

        # Check some expected transition names
        tnames = AlgebraicPetri.tnames(pn)
        for name in expected_names
            # Within-stratum transmission
            within = Symbol(string(name), "_", string(name))
            @test within in tnames
        end
    end
end

# Test with UninfectedInfectedSchema
@testitem "create_model(ContactStratification) with UninfectedInfectedSchema" setup = [ContactStratificationModelSetup] begin
    @testset "AgeStratification" begin
        age_strat = AgeStratification([:child, :adult])
        typed_model = create_model(ui_schema, age_strat)
        pn = dom(typed_model)

        # Should have 4 species (uninfected + infected for each age group)
        @test length(AlgebraicPetri.snames(pn)) == 4

        # Should have n² + 3n = 10 transitions
        @test length(AlgebraicPetri.tnames(pn)) == 10

        # Check species names include _U and _I suffixes
        snames = AlgebraicPetri.snames(pn)
        @test :child_U in snames
        @test :child_I in snames
        @test :adult_U in snames
        @test :adult_I in snames
    end

    @testset "GeographicStratification" begin
        geo_strat = GeographicStratification([:urban, :rural])
        typed_model = create_model(ui_schema, geo_strat)
        pn = dom(typed_model)

        @test length(AlgebraicPetri.snames(pn)) == 4
        @test length(AlgebraicPetri.tnames(pn)) == 10

        snames = AlgebraicPetri.snames(pn)
        @test :urban_U in snames
        @test :urban_I in snames
        @test :rural_U in snames
        @test :rural_I in snames
    end
end

# Test generate_transition_names
@testitem "generate_transition_names creates correct names" setup = [ContactStratificationModelSetup] begin
    @testset "AgeStratification" begin
        strat = AgeStratification([:young, :old])
        uwd = create_model_uwd(one_pop_schema, strat)
        names = generate_transition_names(uwd, strat)

        @test length(names) == 10  # 4 transmission + 8 reflexive
        # Transmission names
        @test :young_young in names
        @test :young_old in names
        @test :old_young in names
        @test :old_old in names
        # Reflexive boxes use stratum name
        @test :young in names
        @test :old in names
    end

    @testset "GeographicStratification" begin
        strat = GeographicStratification([:north, :south])
        uwd = create_model_uwd(one_pop_schema, strat)
        names = generate_transition_names(uwd, strat)

        @test length(names) == 10
        @test :north_north in names
        @test :north_south in names
        @test :south_north in names
        @test :south_south in names
        @test :north in names
        @test :south in names
    end
end

# Test single stratum edge case
@testitem "create_model with single stratum" setup = [ContactStratificationModelSetup] begin
    strats = [
        AgeStratification([:population]),
        GeographicStratification([:region]),
        ContactStratification([:group], :custom),
    ]

    for strat in strats
        typed_model = create_model(one_pop_schema, strat)
        pn = dom(typed_model)

        # Should have 1 species
        @test length(AlgebraicPetri.snames(pn)) == 1

        # Should have n² + 3n = 4 transitions
        @test length(AlgebraicPetri.tnames(pn)) == 4
    end
end

# Test composition with compartmental models
@testitem "typed_product of SIR and ContactStratification" setup = [ContactStratificationModelSetup] begin
    sir = create_model(one_pop_schema, SIR())

    strats = [
        AgeStratification([:child, :adult]),
        GeographicStratification([:urban, :rural]),
        ContactStratification([:low, :high], :risk),
    ]

    for strat in strats
        strat_model = create_model(one_pop_schema, strat)
        composed = typed_product(sir, strat_model)

        @test composed isa ACSetTransformation
        pn = dom(composed)

        # Should have 6 species (S, I, R for each of 2 strata)
        @test length(AlgebraicPetri.snames(pn)) == 6
    end
end
