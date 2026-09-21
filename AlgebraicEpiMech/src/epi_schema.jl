"""
Abstract base type for epidemiological model schemas.

Epidemiological schemas define the structure and organization of populations
and transitions in epidemiological models using typed Petri nets. They serve as a
template for "typing" each species and transition in the Petri net.

## Reference

[Libkind et al. (2023) _An algebraic framework for structured epidemic modelling_](https://royalsocietypublishing.org/rsta/article/380/2233/20210309/112239)
"""
abstract type EpidemiologicalSchema end

"""
Schema for a single-population epidemiological model.

This schema represents models where all individuals belong to a single population
type, with the same stratifications.

# Example
```julia
# Default schema with :Population type
schema = OnePopulationSchema()

# Custom population type
schema = OnePopulationSchema(population_type = :CityPopulation)
```
"""
@kwdef struct OnePopulationSchema <: EpidemiologicalSchema
    "The symbolic name for the population type. Defaults to `:Population`."
    population_type::Symbol = :Population
end

"""
Schema for epidemiological models with uninfected and infected populations, and these
populations may have different stratifications.

This schema represents models where individuals are stratified into uninfected and infected
populations, with transitions for disease transmission between populations and
density-dependent effects within each population.

# Fields
- `uninfected_type::Symbol = :Uninfected`: The symbolic name for the uninfected population
- `infected_type::Symbol = :Infected`: The symbolic name for the infected population

# Examples
```julia
# Default schema with :Uninfected and :Infected types
schema = UninfectedInfectedSchema()

# Custom population types
schema = UninfectedInfectedSchema(
    uninfected_type = :Susceptible,
    infected_type = :Infectious
)
```

See also: [`create_schema`](@ref), [`OnePopulationSchema`](@ref)
"""
@kwdef struct UninfectedInfectedSchema <: EpidemiologicalSchema
    "The symbolic name for the uninfected population. Defaults to `:Uninfected`."
    uninfected_type::Symbol = :Uninfected
    "The symbolic name for the infected population. Defaults to `:Infected`."
    infected_type::Symbol = :Infected
end

function create_schema(schema::EpidemiologicalSchema)
    error("`create_schema` not implemented for schema type $(typeof(schema)). Implement a method for this schema type or use OnePopulationSchema/UninfectedInfectedSchema.")
end

"""
    create_schema(schema::OnePopulationSchema, population_transitions...)

Create a labelled Petri net to be a single-population group schema.

Delegates to [`create_one_population_schema`](@ref) with the population type
specified in the schema.

# Arguments
- `schema::OnePopulationSchema`: Schema defining the single population type
- `population_transitions...`: Additional transition specifications

# Returns
- `LabelledPetriNet`: A single-population Petri net model

# Examples
```julia
schema = OnePopulationSchema(population_type = :Individual)
net = create_schema(schema, :birth => (:Individual => (:Individual, :Individual)))
```
"""
function create_schema(schema::OnePopulationSchema, population_transitions...)
    return create_one_population_schema(
        population_transitions...; population_type = schema.population_type
    )
end

"""
    create_schema(schema::UninfectedInfectedSchema, population_transitions...)

Create a labelled Petri net to be the schema for an uninfected/infected population split.

Delegates to [`create_uninfected_infected_schema`](@ref) with the population types
specified in the schema.

# Arguments
- `schema::UninfectedInfectedSchema`: Schema defining the uninfected and infected population types
- `population_transitions...`: Additional transition specifications

# Returns
- `LabelledPetriNet`: A two-population Petri net model

# Examples
```julia
schema = UninfectedInfectedSchema(
    uninfected_type = :Susceptible,
    infected_type = :Infectious
)
net = create_schema(schema, :reversion => (:Infectious => :Susceptible))
```
"""
function create_schema(schema::UninfectedInfectedSchema, population_transitions...)
    return create_uninfected_infected_schema(
        population_transitions...;
        uninfected_type = schema.uninfected_type,
        infected_type = schema.infected_type
    )
end

"""
Create a labelled Petri net schema for a single population epidemiological model, where all the population
have the same stratification.

This function constructs a `LabelledPetriNet` that defines the structure for modelling
epidemiological dynamics within a single population. The default schema is for one population which
has transitions for disease transmission (these have two inputs and two outputs) and
density-dependent effects (these have one input and one output).

# Arguments
- `population_transitions...`: Variable number of transition specifications that define
new transitions within the population (e.g., aging, different locations)
- `population_type::Symbol = :Population`: The symbolic name for the population type
  in the model (defaults to `:Population`)

# Returns
- `LabelledPetriNet`: A Petri net schema with the specified population type and transitions
"""
function create_one_population_schema(
        population_transitions...; population_type::Symbol = :Population
    )
    schema = LabelledPetriNet(
        [population_type],
        :transmission => (
            (
                population_type, population_type,
            ) => (population_type, population_type)
        ),
        :disease => (population_type => population_type),
        :reversion => (population_type => population_type),
        :waning => (population_type => population_type),
        population_transitions...
    )
    return schema
end

"""
Create a labelled Petri net schema for a two-population epidemiological model with
uninfected and infected population stratification.

This function constructs a `LabelledPetriNet` that defines the structure for modelling
epidemiological dynamics with separate uninfected and infected populations. The schema
includes transitions for disease transmission (requiring interaction between both
populations) and separate density-dependent effects for each population.

# Arguments
- `population_transitions...`: Variable number of transition specifications that define
  additional transitions between or within the populations (e.g., aging, vaccination,
recovery)
- `uninfected_type::Symbol = :Uninfected`: The symbolic name for the uninfected population
type
- `infected_type::Symbol = :Infected`: The symbolic name for the infected population type

# Returns
- `LabelledPetriNet`: A Petri net schema with the specified population types and
transitions.

# Examples
```julia
# Basic uninfected/infected schema
schema = create_uninfected_infected_schema()

# Custom population type names
schema = create_uninfected_infected_schema(
    uninfected_type = :Susceptible,
    infected_type = :Infectious
)
```
"""
function create_uninfected_infected_schema(
        population_transitions...; uninfected_type::Symbol = :Uninfected,
        infected_type::Symbol = :Infected
    )
    schema = LabelledPetriNet(
        [uninfected_type, infected_type],
        :transmission => (
            (
                uninfected_type, infected_type,
            ) => (infected_type, infected_type)
        ),
        :disease => (infected_type => infected_type),
        :reversion => (infected_type => uninfected_type),
        :waning => (uninfected_type => uninfected_type),
        population_transitions...
    )
    return schema
end

"""
Extract the type label used for infected compartments (E, I, R) from the schema.

# Compositional Role
This abstracts away schema differences when adding infected-class compartments. Whether
using `OnePopulationSchema` (all compartments same type) or `UninfectedInfectedSchema` (S
separate from E/I/R), this function returns the correct type for E, I, and R junctions. This
will be useful for composition with other typed Petri nets after implementing such models.
Enables schema-agnostic model construction in `setup_basic!` and extension methods.
"""
function get_infected_type(schema::OnePopulationSchema)
    return schema.population_type
end

function get_infected_type(schema::UninfectedInfectedSchema)
    return schema.infected_type
end

"""
Get the uninfected/susceptible type from a schema.

For UninfectedInfectedSchema: returns the uninfected_type field.
For OnePopulationSchema: returns the population_type (same as infected_type).

Enables schema-agnostic model construction for multistrain models.
"""
function get_uninfected_type(schema::UninfectedInfectedSchema)
    return schema.uninfected_type
end

function get_uninfected_type(schema::OnePopulationSchema)
    return schema.population_type
end
