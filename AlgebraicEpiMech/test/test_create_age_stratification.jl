# Unit tests for create_age_stratification.jl
# Tests the create_model function for age stratification typed Petri nets

@testsnippet AgeStratificationModelSetup begin
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

# Test that create_model returns typed Petri net
@testitem "create_model(AgeStratification) returns ACSetTransformation" setup = [AgeStratificationModelSetup] begin
    age_strat = AgeStratification([:child, :adult])
    typed_model = create_model(one_pop_schema, age_strat)

    # Should return an ACSetTransformation (typed Petri net)
    @test typed_model isa ACSetTransformation
end

@testitem "create_model(AgeStratification) typed model has dom as LabelledPetriNet" setup = [AgeStratificationModelSetup] begin
    age_strat = AgeStratification([:child, :adult])
    typed_model = create_model(one_pop_schema, age_strat)

    # The domain should be a LabelledPetriNet (usable for ODEs)
    pn = dom(typed_model)
    @test pn isa LabelledPetriNet
end

@testitem "create_model(AgeStratification) schema accessible via codom" setup = [AgeStratificationModelSetup] begin
    age_strat = AgeStratification([:child, :adult])
    typed_model = create_model(one_pop_schema, age_strat)

    # The schema should be accessible via codom
    schema_pn = codom(typed_model)
    @test schema_pn isa LabelledPetriNet
end

# Test with OnePopulationSchema
@testitem "create_model(AgeStratification) with 2 age groups OnePopulationSchema" setup = [AgeStratificationModelSetup] begin
    age_strat = AgeStratification([:child, :adult])
    typed_model = create_model(one_pop_schema, age_strat)

    pn = dom(typed_model)

    # Should have 2 species (one per age group)
    @test length(AlgebraicPetri.snames(pn)) == 2

    # Should have n² + 3n = 10 transitions (4 transmission + 6 reflexive)
    @test length(AlgebraicPetri.tnames(pn)) == 10

    # Check transmission names represent contact patterns
    tnames = AlgebraicPetri.tnames(pn)
    @test :child_child in tnames  # within-group child
    @test :child_adult in tnames  # child infected by adult
    @test :adult_child in tnames  # adult infected by child
    @test :adult_adult in tnames  # within-group adult
end

@testitem "create_model(AgeStratification) with 3 age groups OnePopulationSchema" setup = [AgeStratificationModelSetup] begin
    age_strat = AgeStratification([:child, :adult, :elderly])
    typed_model = create_model(one_pop_schema, age_strat)

    pn = dom(typed_model)

    # Should have 3 species
    @test length(AlgebraicPetri.snames(pn)) == 3

    # Should have n² + 3n = 18 transitions
    @test length(AlgebraicPetri.tnames(pn)) == 18
end

# Test with UninfectedInfectedSchema
@testitem "create_model(AgeStratification) with UninfectedInfectedSchema" setup = [AgeStratificationModelSetup] begin
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

# Test generate_transition_names
@testitem "generate_transition_names creates correct names for age groups" setup = [AgeStratificationModelSetup] begin
    age_strat = AgeStratification([:young, :old])
    uwd = create_model_uwd(one_pop_schema, age_strat)

    names = generate_transition_names(uwd, age_strat)

    # Should have n² + 3n = 10 names
    @test length(names) == 10

    # Check transmission patterns exist
    @test :young_young in names
    @test :young_old in names
    @test :old_young in names
    @test :old_old in names
    # Check reflexive stratum names exist
    @test :young in names
    @test :old in names
end

# Test single age group edge case
@testitem "create_model(AgeStratification) with single age group" setup = [AgeStratificationModelSetup] begin
    age_strat = AgeStratification([:population])
    typed_model = create_model(one_pop_schema, age_strat)

    pn = dom(typed_model)

    # Should have 1 species
    @test length(AlgebraicPetri.snames(pn)) == 1

    # Should have n² + 3n = 4 transitions
    @test length(AlgebraicPetri.tnames(pn)) == 4

    @test :population_population in AlgebraicPetri.tnames(pn)
end

# Test composition with compartmental models
@testitem "typed_product of SIR and AgeStratification creates age-structured model" setup = [AgeStratificationModelSetup] begin
    # Create compartmental model
    sir = create_model(one_pop_schema, SIR())

    # Create age stratification
    age_strat = AgeStratification([:child, :adult])
    age_model = create_model(one_pop_schema, age_strat)

    # Compose them
    age_sir = typed_product(sir, age_model)

    # Should be a typed model
    @test age_sir isa ACSetTransformation

    pn = dom(age_sir)

    # Should have 6 species (S, I, R for each of 2 age groups)
    @test length(AlgebraicPetri.snames(pn)) == 6

    # Species names will be tuples from composition - just verify we have the right count
    # The actual naming format depends on typed_product implementation
    snames = AlgebraicPetri.snames(pn)
    @test length(snames) == 6
end
