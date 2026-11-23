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
    get_uninfected_type(typing)
    get_infected_type(typing)

The population types of uninfected compartments (such as S) and of infected compartments (E, I
and R). They coincide under [`OnePopulationTyping`](@ref), and are all [`type_system`](@ref)
needs from a typing.
"""
get_uninfected_type(typing::OnePopulationTyping) = typing.population_type
get_uninfected_type(typing::UninfectedInfectedTyping) = typing.uninfected_type
get_uninfected_type(typing::EpidemiologicalTyping) = _not_implemented(typing)
get_infected_type(typing::OnePopulationTyping) = typing.population_type
get_infected_type(typing::UninfectedInfectedTyping) = typing.infected_type
get_infected_type(typing::EpidemiologicalTyping) = _not_implemented(typing)

function _not_implemented(typing)
    return error(
        "typing interface not implemented for $(typeof(typing)): define `get_uninfected_type` " *
            "and `get_infected_type`, or use OnePopulationTyping/UninfectedInfectedTyping."
    )
end

populations(typing) = unique([get_uninfected_type(typing), get_infected_type(typing)])

# A transition kind owns the box names and Petri signatures it contributes to a type system.
abstract type TransitionKind end
struct Transmission <: TransitionKind end
struct Disease <: TransitionKind end
struct Reversion <: TransitionKind end
struct Waning <: TransitionKind end
struct Movement <: TransitionKind end

const TRANSITION_KINDS = (
    transmission = Transmission(), disease = Disease(), reversion = Reversion(), waning = Waning(),
    strata = Movement(),
)

# Movement keeps the population type, so a typing with two populations has one box per population.
const MOVEMENT_BOXES = (shared = :strata, uninfected = :strata_U, infected = :strata_I)

function signatures(::Transmission, typing)
    U, I = get_uninfected_type(typing), get_infected_type(typing)
    return (:transmission => ((U, I) => (I, I)),)
end
signatures(::Disease, typing) = (:disease => (get_infected_type(typing) => get_infected_type(typing)),)
signatures(::Reversion, typing) = (:reversion => (get_infected_type(typing) => get_uninfected_type(typing)),)
signatures(::Waning, typing) = (:waning => (get_uninfected_type(typing) => get_uninfected_type(typing)),)
function signatures(::Movement, typing)
    U, I = get_uninfected_type(typing), get_infected_type(typing)
    U == I && return (MOVEMENT_BOXES.shared => (U => U),)
    return (MOVEMENT_BOXES.uninfected => (U => U), MOVEMENT_BOXES.infected => (I => I))
end

box_names(kind::TransitionKind, typing) = first.(signatures(kind, typing))

function movement_box(typing, population::Symbol)
    for (name, (input, _)) in signatures(Movement(), typing)
        input == population && return name
    end
    throw(ArgumentError("population $population is not part of the typing"))
end

"""
    type_system(typing::EpidemiologicalTyping, extras...)

The `LabelledPetriNet` that a typed Petri net built under `typing` maps into: one species per
population type and these transition types.

| Transition type | `OnePopulationTyping` | `UninfectedInfectedTyping` |
|---|---|---|
| `:transmission` | `(P, P) → (P, P)` | `(U, I) → (I, I)` |
| `:disease` | `P → P` | `I → I` |
| `:reversion` | `P → P` | `I → U` |
| `:waning` | `P → P` | `U → U` |
| `:strata` (movement) | `P → P` | `:strata_U`: `U → U`, `:strata_I`: `I → I` |

`extras` are further `name => (inputs => outputs)` transition types, appended for Petri-level
experiments; [`create_model`](@ref) uses only the standard ones.

# Examples
```julia
type_system(OnePopulationTyping(population_type = :Individual))
type_system(UninfectedInfectedTyping(), :vaccination => (:Uninfected => :Uninfected))
```
"""
function type_system(typing::EpidemiologicalTyping, extras...)
    kinds = (s for kind in TRANSITION_KINDS for s in signatures(kind, typing))
    return LabelledPetriNet(populations(typing), kinds..., extras...)
end
