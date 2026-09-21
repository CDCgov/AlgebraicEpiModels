module AlgebraicEpiMech

using AlgebraicPetri, AlgebraicPetri.TypedPetri
using Catlab
using DocStringExtensions
import Base: *

# Epidemiological typing strategies and their Petri-net type systems
export EpidemiologicalTyping, OnePopulationTyping, UninfectedInfectedTyping, type_system

# Epi mechanism functions
export add_infection!, add_disease_progression!,
    add_uninfected_density_progression!, add_reversion_progression!

# Observation chain
# Observation by pushout (attached after composition)
export ObservationTarget, AtEvent, AtCompartment, attach_observation

export ObservationChainLayout, ObservationLayout, observation_layout

# Compartmental models types
export CompartmentalModel, SI, SEI, SIS, SIR, SEIR, SEIRS, SEIS

# Multistrain model types
export MultiStrainModel, NoCrossImmunity, CompleteCrossImmunity

# Immune-history model types
export ImmuneHistory, ImmuneHistoryMode, FullHistory, LatestInfection

# Stratification types
export Stratification, ContactStratification, AgeStratification, GeographicStratification,
    compose_stratifications

# Model functions
export EpiMechModel, create_model_uwd, create_model, generate_transition_names

# Vectorfield functions
export flatten_symbols, vectorfield_flat

# Documentation strings
include("docstrings.jl")

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

include("typing.jl")
include("mechanisms.jl")
include("CompartmentalModel.jl")
include("MultiStrainModel.jl")
include("ImmuneHistory.jl")
include("Stratification.jl")
include("ObservationLayout.jl")
include("construction_helpers.jl")
include("create_model_uwd.jl")
include("generate_transition_names.jl")
include("create_model.jl")
include("vectorfield_flat.jl")
# Observation attachment. Runs AFTER composition, so it comes last.
include("observation_rewriting.jl")

end
