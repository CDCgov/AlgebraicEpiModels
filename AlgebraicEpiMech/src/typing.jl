"""
Abstract base type for epidemiological typing strategies.

An `EpidemiologicalTyping` describes how populations and transitions are typed when
constructing epidemiological models. [`type_system`](@ref) materializes the strategy as
the `LabelledPetriNet` used as the codomain of a typed Petri net.

## Reference

[Libkind et al. (2023) _An algebraic framework for structured epidemic modelling_](https://royalsocietypublishing.org/rsta/article/380/2233/20210309/112239)
"""
abstract type EpidemiologicalTyping end

"""
    OnePopulationTyping(; population_type = :Population)

Typing strategy in which every compartment belongs to one population type and therefore
has the same available stratifications.

# Examples
```julia
typing = OnePopulationTyping()
typing = OnePopulationTyping(population_type = :CityPopulation)
```
"""
@kwdef struct OnePopulationTyping <: EpidemiologicalTyping
    "The symbolic name for the population type."
    population_type::Symbol = :Population
end

"""
    UninfectedInfectedTyping(; uninfected_type = :Uninfected, infected_type = :Infected)

Typing strategy that distinguishes uninfected and infected populations. The two
populations may have different stratifications.

# Examples
```julia
typing = UninfectedInfectedTyping()
typing = UninfectedInfectedTyping(
    uninfected_type = :Susceptible,
    infected_type = :Infectious,
)
```

See also: [`type_system`](@ref), [`OnePopulationTyping`](@ref)
"""
@kwdef struct UninfectedInfectedTyping <: EpidemiologicalTyping
    "The symbolic name for the uninfected population."
    uninfected_type::Symbol = :Uninfected
    "The symbolic name for the infected population."
    infected_type::Symbol = :Infected
end

"""
    type_system(typing::EpidemiologicalTyping, population_transitions...)

Materialize an epidemiological typing strategy as the `LabelledPetriNet` used as the
codomain of a typed Petri net. Concrete `EpidemiologicalTyping` subtypes must implement
this interface.
"""
function type_system(typing::EpidemiologicalTyping, population_transitions...)
    return error(
        "`type_system` not implemented for typing $(typeof(typing)). " *
            "Implement a method for this typing or use " *
            "OnePopulationTyping/UninfectedInfectedTyping."
    )
end

"""
    type_system(typing::OnePopulationTyping, population_transitions...)

Create the single-population Petri-net type system described by `typing`. Additional
transition specifications are appended to the standard `:transmission`, `:disease`,
`:reversion`, and `:waning` transitions.

# Examples
```julia
typing = OnePopulationTyping(population_type = :Individual)
codomain = type_system(
    typing,
    :birth => (:Individual => (:Individual, :Individual)),
)
```
"""
function type_system(typing::OnePopulationTyping, population_transitions...)
    population_type = typing.population_type
    return LabelledPetriNet(
        [population_type],
        :transmission => (
            (population_type, population_type) => (population_type, population_type)
        ),
        :disease => (population_type => population_type),
        :reversion => (population_type => population_type),
        :waning => (population_type => population_type),
        population_transitions...
    )
end

"""
    type_system(typing::UninfectedInfectedTyping, population_transitions...)

Create the uninfected/infected Petri-net type system described by `typing`. Additional
transition specifications are appended to the standard `:transmission`, `:disease`,
`:reversion`, and `:waning` transitions.

# Examples
```julia
typing = UninfectedInfectedTyping(
    uninfected_type = :Susceptible,
    infected_type = :Infectious,
)
codomain = type_system(
    typing,
    :vaccination => (:Susceptible => :Susceptible),
)
```
"""
function type_system(typing::UninfectedInfectedTyping, population_transitions...)
    uninfected_type = typing.uninfected_type
    infected_type = typing.infected_type
    return LabelledPetriNet(
        [uninfected_type, infected_type],
        :transmission => (
            (uninfected_type, infected_type) => (infected_type, infected_type)
        ),
        :disease => (infected_type => infected_type),
        :reversion => (infected_type => uninfected_type),
        :waning => (uninfected_type => uninfected_type),
        population_transitions...
    )
end

"""
Return the type label used for infected compartments (E, I, and R).

This interface lets model construction remain agnostic to the selected epidemiological
typing strategy.
"""
get_infected_type(typing::OnePopulationTyping) = typing.population_type
get_infected_type(typing::UninfectedInfectedTyping) = typing.infected_type

"""
Return the type label used for uninfected compartments such as S.
"""
get_uninfected_type(typing::OnePopulationTyping) = typing.population_type
get_uninfected_type(typing::UninfectedInfectedTyping) = typing.uninfected_type
