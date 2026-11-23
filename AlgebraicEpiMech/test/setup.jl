@testsnippet ModelSetup begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri
    using Catlab
    using LabelledArrays
    using SeeToDee: Rk4

    one_pop_typing = OnePopulationTyping(population_type = :Individual)
    ui_typing = UninfectedInfectedTyping(uninfected_type = :Susceptible, infected_type = :Infectious)
end
