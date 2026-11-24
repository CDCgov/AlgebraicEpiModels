"""
Abstract base type for all epidemiological-mechanical models in the AlgebraicEpiMech framework.

All concrete epidemiological-mechanical model types should be subtypes of `EpiMechModel`.
This abstract type serves as the root of the type hierarchy for models that combine
epidemiological dynamics with mechanical or algebraic structures.

# Extended help

Subtypes of `EpiMechModel` should implement the necessary interface methods for their
specific model formulation.

# See also
- Related concrete model types (define as needed)
"""
abstract type EpiMechModel end

"""
Abstract base type for compartmental epidemiological models.

Compartmental models define the specific disease dynamics and transitions
between epidemiological states (e.g., S→I→R) as undirected wiring diagrams.

Every concrete model also takes the [`Demography`](@ref) keywords, naming compartments, for
example `SIR(births = [:S], deaths = [:I], importation = true)`.
"""
abstract type CompartmentalModel <: EpiMechModel end

"""
    Demography(; births = (), deaths = (), importation = false)

The open-population part of a model specification: the names (compartments, or strata of a
[`ContactStratification`](@ref)) that births enter and that deaths leave. Each birth becomes
one `:birth` box and each death one death box per junction of that name; a stratum's births
enter its uninfected junction. `importation = true` adds an `:importation` box from `S` into
a compartmental model's first infected stage.

In a `typed_product` a transition type survives only if every factor has it. A factor that
declares no deaths can still be passive for them through `include_reflexives = :death` on
[`create_model`](@ref), and a stratification factor is passive for importation through
`:importation`; births cannot be passive, because a newborn must be placed in every factor.
"""
struct Demography
    births::Tuple{Vararg{Symbol}}
    deaths::Tuple{Vararg{Symbol}}
    importation::Bool

    # Tuples, so that specifications compare by value (`SIR() == SIR()`).
    function Demography(;
            births::Union{Tuple{Vararg{Symbol}}, AbstractVector{Symbol}} = (),
            deaths::Union{Tuple{Vararg{Symbol}}, AbstractVector{Symbol}} = (),
            importation::Bool = false,
        )
        allunique(births) || throw(ArgumentError("duplicate `births` entry in $births"))
        allunique(deaths) || throw(ArgumentError("duplicate `deaths` entry in $deaths"))
        return new(Tuple(births), Tuple(deaths), importation)
    end
end

function _validate_stage_count(name::Symbol, count::Int)
    count >= 1 || throw(ArgumentError("$name must be at least 1"))
    return count
end

# Base compartmental models - concrete structs for clean dispatch

"""
Susceptible-Infected (SI) compartmental model.

The SI model includes:
- Susceptible (S): Individuals who can become infected
- Infected (I): Individuals who are infectious

Transitions:
- S + I → I + I (infection/transmission)

The SI model represents endemic diseases with no recovery.
"""
struct SI <: CompartmentalModel
    number_I_stages::Int
    number_of_states::Int
    demography::Demography

    function SI(; number_I_stages::Int = 1, demography...)
        _validate_stage_count(:number_I_stages, number_I_stages)
        return new(number_I_stages, 1 + number_I_stages, Demography(; demography...))
    end
end

"""
Susceptible-Exposed-Infected (SEI) compartmental model.

The SEI model extends SI with an exposed compartment:
- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious

Transitions:
- S + I → E + I (exposure/transmission)
- E → I (progression to infectious)

The SEI model represents diseases with an incubation period but no recovery.
"""
struct SEI <: CompartmentalModel
    number_E_stages::Int
    number_I_stages::Int
    number_of_states::Int
    demography::Demography

    function SEI(; number_E_stages::Int = 1, number_I_stages::Int = 1, demography...)
        _validate_stage_count(:number_E_stages, number_E_stages)
        _validate_stage_count(:number_I_stages, number_I_stages)
        return new(number_E_stages, number_I_stages, 1 + number_E_stages + number_I_stages, Demography(; demography...))
    end
end

# Extended compartmental models - built by extending base models

"""
Susceptible-Infected-Recovered (SIR) compartmental model.

The SIR model extends SI with recovery:
- Susceptible (S): Individuals who can become infected
- Infected (I): Individuals who are infectious
- Recovered (R): Individuals who have recovered and gained immunity

Transitions:
- S + I → I + I (infection/transmission) [from SI]
- I → R (recovery) [extension]

Built by extending SI model with recovery transition.
"""
struct SIR <: CompartmentalModel
    number_I_stages::Int
    number_of_states::Int
    demography::Demography

    function SIR(; number_I_stages::Int = 1, demography...)
        _validate_stage_count(:number_I_stages, number_I_stages)
        return new(number_I_stages, 2 + number_I_stages, Demography(; demography...))
    end
end

"""
Susceptible-Exposed-Infected-Recovered (SEIR) compartmental model.

The SEIR model extends SEI with recovery:
- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious
- Recovered (R): Individuals who have recovered and gained immunity

Transitions:
- S + I → E + I (exposure/transmission) [from SEI]
- E → I (progression to infectious) [from SEI]
- I → R (recovery) [extension]

Built by extending SEI model with recovery transition.
"""
struct SEIR <: CompartmentalModel
    number_E_stages::Int
    number_I_stages::Int
    number_of_states::Int
    demography::Demography

    function SEIR(; number_E_stages::Int = 1, number_I_stages::Int = 1, demography...)
        _validate_stage_count(:number_E_stages, number_E_stages)
        _validate_stage_count(:number_I_stages, number_I_stages)
        return new(number_E_stages, number_I_stages, 2 + number_E_stages + number_I_stages, Demography(; demography...))
    end
end

"""
Susceptible-Infected-Susceptible (SIS) compartmental model.

The SIS model extends SI with waning immunity (reversion):
- Susceptible (S): Individuals who can become infected
- Infected (I): Individuals who are infectious

Transitions:
- S + I → I + I (infection/transmission) [from SI]
- I → S (waning immunity/reversion) [extension]

The SIS model represents diseases with no lasting immunity.
"""
struct SIS <: CompartmentalModel
    number_I_stages::Int
    number_of_states::Int
    demography::Demography

    function SIS(; number_I_stages::Int = 1, demography...)
        _validate_stage_count(:number_I_stages, number_I_stages)
        return new(number_I_stages, 1 + number_I_stages, Demography(; demography...))
    end
end

"""
Susceptible-Exposed-Infected-Susceptible (SEIS) compartmental model.

The SEIS model extends SEI with waning immunity (reversion):
- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious

Transitions:
- S + I → E + I (exposure/transmission) [from SEI]
- E → I (progression to infectious) [from SEI]
- I → S (waning immunity/reversion) [extension]

The SEIS model represents diseases with incubation period but no lasting immunity.
"""
struct SEIS <: CompartmentalModel
    number_E_stages::Int
    number_I_stages::Int
    number_of_states::Int
    demography::Demography

    function SEIS(; number_E_stages::Int = 1, number_I_stages::Int = 1, demography...)
        _validate_stage_count(:number_E_stages, number_E_stages)
        _validate_stage_count(:number_I_stages, number_I_stages)
        return new(number_E_stages, number_I_stages, 1 + number_E_stages + number_I_stages, Demography(; demography...))
    end
end

"""
Susceptible-Exposed-Infected-Recovered-Susceptible (SEIRS) compartmental model.

The SEIRS model extends SEIR with waning immunity:
- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious
- Recovered (R): Individuals who have recovered but may lose immunity

Transitions:
- S + I → E + I (exposure/transmission) [from SEIR]
- E → I (progression to infectious) [from SEIR]
- I → R (recovery) [from SEIR]
- R → S (waning immunity) [extension]

Built by extending SEIR model with waning transition.
"""
struct SEIRS <: CompartmentalModel
    number_E_stages::Int
    number_I_stages::Int
    number_of_states::Int
    demography::Demography

    function SEIRS(; number_E_stages::Int = 1, number_I_stages::Int = 1, demography...)
        _validate_stage_count(:number_E_stages, number_E_stages)
        _validate_stage_count(:number_I_stages, number_I_stages)
        return new(number_E_stages, number_I_stages, 2 + number_E_stages + number_I_stages, Demography(; demography...))
    end
end
