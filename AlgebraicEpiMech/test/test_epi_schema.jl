@testitem "epidemiological typing constructors" begin
    one_population = OnePopulationTyping()
    @test one_population isa EpidemiologicalTyping
    @test one_population.population_type == :Population

    custom_population = OnePopulationTyping(population_type = :Individual)
    @test custom_population.population_type == :Individual

    uninfected_infected = UninfectedInfectedTyping()
    @test uninfected_infected isa EpidemiologicalTyping
    @test uninfected_infected.uninfected_type == :Uninfected
    @test uninfected_infected.infected_type == :Infected

    custom_uninfected_infected = UninfectedInfectedTyping(
        uninfected_type = :Susceptible,
        infected_type = :Infectious,
    )
    @test custom_uninfected_infected.uninfected_type == :Susceptible
    @test custom_uninfected_infected.infected_type == :Infectious
end

@testitem "type_system materializes OnePopulationTyping" setup = [ModelSetup] begin
    typing = OnePopulationTyping(population_type = :Individual)
    codomain = type_system(typing)

    @test codomain isa LabelledPetriNet
    @test snames(codomain) == [:Individual]
    @test all(in(tnames(codomain)), [:transmission, :disease, :reversion, :waning, :birth, :death])

    system_with_vaccination = type_system(typing, :vaccination => (:Individual => :Individual))
    @test :vaccination in tnames(system_with_vaccination)
end

@testitem "type_system materializes UninfectedInfectedTyping" setup = [ModelSetup] begin
    typing = UninfectedInfectedTyping(
        uninfected_type = :Susceptible,
        infected_type = :Infectious,
    )
    codomain = type_system(typing)

    @test codomain isa LabelledPetriNet
    @test snames(codomain) == [:Susceptible, :Infectious]
    @test all(in(tnames(codomain)), [:transmission, :disease, :reversion, :waning])

    system_with_recovery = type_system(
        typing,
        :recovery => (:Infectious => :Susceptible),
    )
    @test :recovery in tnames(system_with_recovery)
end

@testitem "type_system rejects unimplemented typings" begin
    struct MockTyping <: EpidemiologicalTyping end

    error = try
        type_system(MockTyping())
        nothing
    catch exception
        exception
    end

    @test error isa ErrorException
    @test occursin("MockTyping", string(error))
    @test occursin("not implemented", string(error))
end

@testitem "type_system builds the standard transition kinds from a typing's populations" setup = [ModelSetup] begin
    one = OnePopulationTyping(population_type = :Person)
    @test tnames(type_system(one)) == [:transmission, :disease, :reversion, :waning, :strata, :birth, :death, :importation]
    typing = UninfectedInfectedTyping(uninfected_type = :U, infected_type = :J)
    split = type_system(typing)
    @test tnames(split) == [
        :transmission, :disease, :reversion, :waning, :strata_U, :strata_I, :birth, :death_U, :death_I,
        :importation,
    ]
    arcs(t) = (snames(split)[AlgebraicPetri.inputs(split, t)], snames(split)[AlgebraicPetri.outputs(split, t)])
    @test arcs.(1:10) == [
        ([:U, :J], [:J, :J]), ([:J], [:J]), ([:J], [:U]), ([:U], [:U]), ([:U], [:U]), ([:J], [:J]),
        (Symbol[], [:U]), ([:U], Symbol[]), ([:J], Symbol[]), ([:U], [:J]),
    ]

    @test AlgebraicEpiMech.population_box(AlgebraicEpiMech.Movement(), one, :Person) == :strata
    @test AlgebraicEpiMech.population_box(AlgebraicEpiMech.Movement(), typing, :U) == :strata_U
    @test AlgebraicEpiMech.population_box(AlgebraicEpiMech.Movement(), typing, :J) == :strata_I
    @test_throws ArgumentError AlgebraicEpiMech.population_box(AlgebraicEpiMech.Movement(), typing, :Other)
    @test AlgebraicEpiMech.population_box(AlgebraicEpiMech.Death(), one, :Person) == :death
    @test AlgebraicEpiMech.population_box(AlgebraicEpiMech.Death(), typing, :U) == :death_U
    @test AlgebraicEpiMech.population_box(AlgebraicEpiMech.Death(), typing, :J) == :death_I

    struct TwoClassTyping <: EpidemiologicalTyping end
    AlgebraicEpiMech.get_uninfected_type(::TwoClassTyping) = :Healthy
    AlgebraicEpiMech.get_infected_type(::TwoClassTyping) = :Sick
    @test snames(type_system(TwoClassTyping())) == [:Healthy, :Sick]
    @test codom(create_model(one_pop_typing, SIR())) ==
        codom(create_model(one_pop_typing, AgeStratification(:a, :b)))
end
