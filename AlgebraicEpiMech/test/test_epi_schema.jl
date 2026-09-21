@testitem "epidemiological typing constructors" begin
    using AlgebraicEpiMech

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

@testitem "type_system materializes OnePopulationTyping" begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri
    using AlgebraicEpiMech

    typing = OnePopulationTyping(population_type = :Individual)
    codomain = type_system(typing)

    @test codomain isa LabelledPetriNet
    @test snames(codomain) == [:Individual]
    @test all(in(tnames(codomain)), [:transmission, :disease, :reversion, :waning])

    system_with_birth = type_system(
        typing,
        :birth => (:Individual => (:Individual, :Individual)),
    )
    @test :birth in tnames(system_with_birth)
end

@testitem "type_system materializes UninfectedInfectedTyping" begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri
    using AlgebraicEpiMech

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
    using AlgebraicEpiMech

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
