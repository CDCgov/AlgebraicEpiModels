# Unit tests for create_multistrain_model.jl
# Tests the create_model function that produces typed Petri nets for multistrain models

@testsnippet MultistrainSetup begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri
    using Catlab
    using AlgebraicEpiMech

    # Create typing instances for testing
    one_pop_typing = OnePopulationTyping(population_type = :Individual)
    uninfected_infected_typing = UninfectedInfectedTyping(
        uninfected_type = :Susceptible, infected_type = :Infectious
    )
end

@testitem "create_model - basic NoCrossImmunity" setup = [MultistrainSetup] begin
    model = NoCrossImmunity([:h1n1, :h3n2])
    typed_model = create_model(one_pop_typing, model)

    # Should return an ACSetTransformation
    @test typed_model isa ACSetTransformation

    # Extract the underlying Petri net
    pn = dom(typed_model)
    @test pn isa LabelledPetriNet

    # Check transition names are the strain names
    transition_names = tnames(pn)
    @test :h1n1 in transition_names
    @test :h3n2 in transition_names
end

@testitem "multistrain constructors reject duplicate strain names" begin
    using AlgebraicEpiMech

    @test_throws ArgumentError NoCrossImmunity([:a, :a])
    @test_throws ArgumentError CompleteCrossImmunity([:a, :a])
end

@testitem "multistrain models require their documented typing" setup = [MultistrainSetup] begin
    no_cross = NoCrossImmunity([:a, :b])
    complete = CompleteCrossImmunity([:a, :b])

    @test applicable(create_model_uwd, one_pop_typing, no_cross)
    @test !applicable(create_model_uwd, uninfected_infected_typing, no_cross)
    @test applicable(create_model_uwd, uninfected_infected_typing, complete)
    @test !applicable(create_model_uwd, one_pop_typing, complete)

    @test_throws MethodError create_model(uninfected_infected_typing, no_cross)
    @test_throws MethodError create_model(one_pop_typing, complete)
end

@testitem "create_model - verify state structure" setup = [MultistrainSetup] begin
    model = NoCrossImmunity([:wild, :variant])
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)

    # Should only have strain states (no Uninfected in pure strain model)
    state_names = snames(pn)
    @test :wild in state_names
    @test :variant in state_names
    @test length(state_names) == 2
end

@testitem "create_model - auto-generated strain names" setup = [MultistrainSetup] begin
    model = NoCrossImmunity(3)
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)

    # Should have auto-generated strain names
    state_names = snames(pn)
    @test :strain_1 in state_names
    @test :strain_2 in state_names
    @test :strain_3 in state_names
    @test length(state_names) == 3
end

@testitem "generate_transition_names - returns strain names" setup = [MultistrainSetup] begin
    model = NoCrossImmunity([:flu_a, :flu_b])
    uwd = create_model_uwd(one_pop_typing, model)

    # Generate transition names
    names = generate_transition_names(uwd, model)

    # Should return strain names (repeated for each box)
    @test all(name in [:flu_a, :flu_b] for name in names)
    # Should have 2 strains * 3 boxes each = 6 transition names
    @test length(names) == 6
end

@testitem "create_model - composition with compartmental model" setup = [MultistrainSetup] begin
    # Create both a compartmental and multistrain model
    sir_model = create_model(one_pop_typing, SIR())
    strain_model = create_model(
        one_pop_typing, NoCrossImmunity(
            [
                :alpha, :beta,
            ]
        )
    )

    # Compose them
    combined = typed_product(sir_model, strain_model)
    combined_pn = dom(combined)

    # Should have states for each SIR compartment × strain combination
    state_names = snames(combined_pn)

    # All compartments get combined with each strain
    @test (:S, :alpha) in state_names
    @test (:S, :beta) in state_names
    @test (:I, :alpha) in state_names
    @test (:I, :beta) in state_names
    @test (:R, :alpha) in state_names
    @test (:R, :beta) in state_names

    # Total: 3 compartments × 2 strains = 6 states
    @test length(state_names) == 6
end

# Tests for CompleteCrossImmunity multistrain models

@testitem "create_model - basic CompleteCrossImmunity" setup = [MultistrainSetup] begin
    model = CompleteCrossImmunity([:wild_type, :variant])
    typed_model = create_model(uninfected_infected_typing, model)

    # Should return an ACSetTransformation
    @test typed_model isa ACSetTransformation

    # Extract the underlying Petri net
    pn = dom(typed_model)
    @test pn isa LabelledPetriNet

    # Check transition names are the strain names
    transition_names = tnames(pn)
    @test :wild_type in transition_names
    @test :variant in transition_names
end

@testitem "create_model - CompleteCrossImmunity state structure" setup = [MultistrainSetup] begin
    model = CompleteCrossImmunity([:strain_a, :strain_b])
    typed_model = create_model(uninfected_infected_typing, model)

    pn = dom(typed_model)

    # Should have 1 shared susceptible + 2 strain-specific infected states
    state_names = snames(pn)
    @test :susceptible in state_names
    @test :strain_a in state_names
    @test :strain_b in state_names
    @test length(state_names) == 3
end

@testitem "create_model - CompleteCrossImmunity auto-generated names" setup = [MultistrainSetup] begin
    model = CompleteCrossImmunity(3)
    typed_model = create_model(uninfected_infected_typing, model)

    pn = dom(typed_model)

    # Should have auto-generated strain names
    state_names = snames(pn)
    @test :susceptible in state_names
    @test :strain_1 in state_names
    @test :strain_2 in state_names
    @test :strain_3 in state_names
    @test length(state_names) == 4  # 1 susceptible + 3 strains
end

@testitem "generate_transition_names - CompleteCrossImmunity" setup = [MultistrainSetup] begin
    model = CompleteCrossImmunity([:h1n1, :h3n2])
    uwd = create_model_uwd(uninfected_infected_typing, model)

    # Generate transition names
    names = generate_transition_names(uwd, model)

    # Should return strain names (repeated for each box), excluding :susceptible
    @test all(name in [:h1n1, :h3n2] for name in names)
    # Should have 2 strains * 3 boxes each = 6 transition names
    @test length(names) == 6
end

@testitem "create_model - CompleteCrossImmunity composition with SIR" setup = [MultistrainSetup] begin
    # Create both a compartmental and multistrain model
    sir_model = create_model(uninfected_infected_typing, SIR())
    strain_model = create_model(
        uninfected_infected_typing,
        CompleteCrossImmunity([:alpha, :beta])
    )

    # Compose them
    combined = typed_product(sir_model, strain_model)
    combined_pn = dom(combined)

    # For CompleteCrossImmunity with SIR:
    # - S is shared (just one S)
    # - I and R are strain-specific (I_alpha, I_beta, R_alpha, R_beta)
    state_names = snames(combined_pn)

    # Shared susceptible
    @test (:S, :susceptible) in state_names

    # Strain-specific infected and recovered
    @test (:I, :alpha) in state_names
    @test (:I, :beta) in state_names
    @test (:R, :alpha) in state_names
    @test (:R, :beta) in state_names

    # Total: 1 shared S + 2*2 strain-specific (I,R) = 5 states
    @test length(state_names) == 5
end
