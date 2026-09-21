# Unified create_model function for all model types
# Multiple dispatch on model type parameter handles different model types
# (CompartmentalModel, AgeStratification, MultiStrainModel)

"""
Create a typed Petri net model from an epidemiological schema and model specification.

This is the unified entry point for creating all model types in AlgebraicEpiMech. Through
multiple dispatch on the `model` parameter, it supports any subtype of `EpiMechModel`.

# Functionality Overview
The `create_model` function follows a consistent four-step pattern to build the typed
Petri net representation of the epidemiological-mechanical model:

1. **Create Schema**: Generates a `LabelledPetriNet` defining the type system
2. **Create UWD**: Builds an undirected wiring diagram with model structure and box names
3. **Generate Transition Names**: Creates unique names for ODE parameters via dispatch
4. **Apply Typing**: Uses `oapply_typed` to create a typed Petri net

The resulting typed Petri net can be used directly for ODE simulation or composed with other
typed Petri nets via `typed_product` to build complex hierarchical models.

# Arguments
- `schema::EpidemiologicalSchema`: The population schema defining possible compartment and transition types.
- `model::EpiMechModel`: The model specification (uses dispatch for different types)

# Keyword Arguments
Keyword arguments are forwarded to `create_model_uwd`. Model-specific keywords include:
- `include_reflexives::Bool=true` (for `ContactStratification`): Controls whether per-stratum
  reflexive boxes (disease, reversion, waning, observation) are added. Required for `typed_product`
  composition; disable for standalone use to avoid zero-effect transitions.

# Returns
- `ACSetTransformation`: A typed Petri net with two components accessible via:
  - `dom(typed_model)`: Extract the underlying `LabelledPetriNet` for ODE solving
  - `codom(typed_model)`: Access the original schema

# Model-Specific Transition Naming
Transition names are generated automatically based on model type:
- **Compartmental**: `transmission_S_I`, `E_to_I`, `I_to_R` (describes flow between compartments)
- **Age Stratification**: `child_child`, `child_adult`, `adult_child` (infectee_infector pattern)
- **Multistrain**: Strain identifiers like `h1n1`, `h3n2` (enables clean composition)

These names become ODE parameters and differ from box names (`:transmission`, `:density`) used for composition.

# Examples
```julia
# Basic compartmental model
schema = OnePopulationSchema()
seir = create_model(schema, SEIR())
seir_pn = dom(seir)  # Extract for ODE solving
# Transitions: :transmission_S_I, :E_to_I, :I_to_R

# Multi-stage compartments (gamma-distributed delays)
si_multi = create_model(schema, SI(number_I_stages=3))
# Transitions: :transmission_S_I1, :transmission_S_I2, :transmission_S_I3, :I1_to_I2, :I2_to_I3

# Age-structured model
age_strat = AgeStratification([:child, :adult, :elderly])
age_model = create_model(schema, age_strat)
# Transitions: :child_child, :child_adult, :child_elderly, :adult_child, ...

# Multistrain model (no cross-immunity)
strains = NoCrossImmunity([:h1n1, :h3n2, :seasonal_b])
strain_model = create_model(schema, strains)

# Compose models via typed_product
sir = create_model(schema, SIR())
age = create_model(schema, AgeStratification([:child, :adult]))
age_sir = typed_product(sir, age)
age_sir_pn = dom(age_sir)
# Result: S_child, I_child, R_child, S_adult, I_adult, R_adult compartments
# Transition names become tuples: (:transmission_S_I, :child_adult)
```

# Compositional Design
Models expressed as typed petri nets compose via `typed_product` using their common
schema/codomain when box names match. The `vectorfield_flat` function automatically handles tuple transition names
in composed models for ODE parameter assignment.

See also: [`create_model_uwd`](@ref), [`generate_transition_names`](@ref), [`create_schema`](@ref),
          `typed_product` (from `AlgebraicPetri.TypedPetri`), [`vectorfield_flat`](@ref)
"""
function create_model(
        schema::EpidemiologicalSchema,
        model::EpiMechModel;
        include_reflexives::Bool = true
    )
    # Create the schema as a LabelledPetriNet
    schema_pn = create_schema(schema)

    # Create the UWD for the model (dispatch on include_reflexives via positional arg)
    uwd = _create_model_uwd(schema, model, include_reflexives)

    # Generate transition names (dispatch on model type)
    transition_names = generate_transition_names(uwd, model)

    # Apply oapply_typed to create the typed Petri net
    typed_model = oapply_typed(schema_pn, uwd, transition_names)

    return typed_model
end

# Intermediate layer: converts include_reflexives kwarg to positional for dispatch
_create_model_uwd(schema, model, ::Bool) = create_model_uwd(schema, model)
function _create_model_uwd(schema, model::ContactStratification, include_reflexives::Bool)
    return create_model_uwd(schema, model; include_reflexives)
end
