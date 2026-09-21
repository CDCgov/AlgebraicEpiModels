# Type system for multistrain epidemiological models

"""
Abstract base type for multistrain epidemiological models.

Multistrain models define how multiple pathogen strains interact through
different assumptions about cross-immunity. These compose with compartmental
models via typed_product to create strain-structured disease dynamics.
"""
abstract type MultiStrainModel <: EpiMechModel end

"""
No cross-immunity multistrain model.

In the no cross-immunity model:
- Each strain operates independently with no interaction
- Strains act as independent strata (similar to age groups)
- Typing: OnePopulationTyping (all compartments have the same type)
- Composition: Creates parallel disease dynamics per strain via typed_product

# Fields
- `number_of_strains::Int`: Number of independent strains in the model
- `strain_names::Vector{Symbol}`: Names for each strain (e.g., [:h1n1, :h3n2, :b])

# Examples
```julia
# Create a 3-strain model with custom names
multistrain = NoCrossImmunity([:h1n1, :h3n2, :b])

# Create a 2-strain model with auto-generated names
multistrain = NoCrossImmunity(2)  # Creates [:strain_1, :strain_2]

# Compose to create strain-structured SIR
typing = OnePopulationTyping()
strain_typed = create_multistrain_model(typing, multistrain)
sir_typed = create_compartmental_model(typing, SIR())
combined = typed_product(sir_typed, strain_typed)
```
"""
struct NoCrossImmunity <: MultiStrainModel
    number_of_strains::Int
    strain_names::Vector{Symbol}

    function NoCrossImmunity(strain_names::Vector{Symbol})
        if length(strain_names) < 1
            throw(ArgumentError("strain_names must contain at least 1 strain"))
        end
        return new(length(strain_names), strain_names)
    end

    function NoCrossImmunity(number_of_strains::Int)
        if number_of_strains < 1
            throw(ArgumentError("number_of_strains must be at least 1"))
        end
        strain_names = [Symbol("strain_$(i)") for i in 1:number_of_strains]
        return new(number_of_strains, strain_names)
    end
end

"""
Complete cross-immunity multistrain model.

In the complete cross-immunity model:
- All strains share a common susceptible pool
- Infection by any strain confers immunity to all strains
- Typing: UninfectedInfectedTyping (typed S vs I/R compartments)
- Composition: Strains compete for susceptibles via shared S depletion

# Fields
- `number_of_strains::Int`: Number of competing strains in the model
- `strain_names::Vector{Symbol}`: Names for each strain (e.g., [:h1n1, :h3n2])

# Examples
```julia
# Create a 2-strain model with custom names
multistrain = CompleteCrossImmunity([:wild_type, :variant])

# Create a 3-strain model with auto-generated names
multistrain = CompleteCrossImmunity(3)  # Creates [:strain_1, :strain_2, :strain_3]

# Compose to create competing strain SIR
typing = UninfectedInfectedTyping()
strain_typed = create_multistrain_model(typing, multistrain)
sir_typed = create_compartmental_model(typing, SIR())
combined = typed_product(sir_typed, strain_typed)
```
"""
struct CompleteCrossImmunity <: MultiStrainModel
    number_of_strains::Int
    strain_names::Vector{Symbol}

    function CompleteCrossImmunity(strain_names::Vector{Symbol})
        if length(strain_names) < 1
            throw(ArgumentError("strain_names must contain at least 1 strain"))
        end
        return new(length(strain_names), strain_names)
    end

    function CompleteCrossImmunity(number_of_strains::Int)
        if number_of_strains < 1
            throw(ArgumentError("number_of_strains must be at least 1"))
        end
        strain_names = [Symbol("strain_$(i)") for i in 1:number_of_strains]
        return new(number_of_strains, strain_names)
    end
end
