# Unified create_model function for all model types
# Multiple dispatch on model type parameter handles different model types
# (CompartmentalModel, AgeStratification, MultiStrainModel)

"""
Create a typed Petri net model from an epidemiological typing strategy and model specification.

This is the unified entry point for creating all model types in AlgebraicEpiMech. Through
multiple dispatch on the `model` parameter, it supports any subtype of `EpiMechModel`.
The undirected wiring diagram (UWD) provides an intermediate structural representation, recording
how compartments and mechanisms connect independently of the final typed Petri net constructed by
`oapply_typed`.

# Functionality Overview
The `create_model` function builds the typed
Petri net representation of the epidemiological-mechanical model in four steps:

1. **Build Type System**: Materializes the typing strategy as a `LabelledPetriNet`
2. **Create UWD**: Builds an undirected wiring diagram with model structure and box names
3. **Generate Transition Names**: Creates unique names for ODE parameters via dispatch
4. **Apply Typing**: Uses `oapply_typed` to create a typed Petri net

The resulting typed Petri net can be used directly for ODE simulation or composed with other
typed Petri nets via `typed_product` to build complex hierarchical models.

# Arguments
- `typing::EpidemiologicalTyping`: The strategy defining possible compartment and transition types.
- `model::EpiMechModel`: The model specification (uses dispatch for different types)

# Keyword Arguments
Keyword arguments are forwarded to `create_model_uwd`. Model-specific keywords include:
- `include_reflexives::Bool=true` (for `ContactStratification`): Controls whether per-stratum
  reflexive boxes (disease, reversion, waning, observation) are added. Required for `typed_product`
  composition; disable for standalone use to avoid zero-effect transitions.

# Returns
- `ACSetTransformation`: A typed Petri net with two components accessible via:
  - `dom(typed_model)`: Extract the underlying `LabelledPetriNet` for ODE solving
  - `codom(typed_model)`: Access the Petri-net type system

# Model-Specific Transition Naming
Transition names are generated automatically based on model type:
- **Compartmental**: `transmission_S_I`, `E_to_I`, `I_to_R` (describes flow between compartments)
- **Age Stratification**: `child_child`, `child_adult`, `adult_child` (infectee_infector pattern)
- **Multistrain**: Strain identifiers like `h1n1`, `h3n2` (enables clean composition)

These names become ODE parameters and differ from box names (`:transmission`, `:density`) used for composition.

# Examples
```julia
# Basic compartmental model
typing = OnePopulationTyping()
seir = create_model(typing, SEIR())
seir_pn = dom(seir)  # Extract for ODE solving
# Transitions: :transmission_S_I, :E_to_I, :I_to_R

# Multi-stage compartments (gamma-distributed delays)
si_multi = create_model(typing, SI(number_I_stages=3))
# Transitions: :transmission_S_I1, :transmission_S_I2, :transmission_S_I3, :I1_to_I2, :I2_to_I3

# Age-structured model
age_strat = AgeStratification([:child, :adult, :elderly])
age_model = create_model(typing, age_strat)
# Transitions: :child_child, :child_adult, :child_elderly, :adult_child, ...

# Multistrain model (no cross-immunity)
strains = NoCrossImmunity([:h1n1, :h3n2, :seasonal_b])
strain_model = create_model(typing, strains)

# Compose models via typed_product
sir = create_model(typing, SIR())
age = create_model(typing, AgeStratification([:child, :adult]))
age_sir = typed_product(sir, age)
age_sir_pn = dom(age_sir)
# Result: S_child, I_child, R_child, S_adult, I_adult, R_adult compartments
# Transition names become tuples: (:transmission_S_I, :child_adult)
```

# Compositional Design
Models expressed as typed Petri nets compose via `typed_product` using their common
type-system codomain when box names match. The `vectorfield_flat` function automatically handles tuple transition names
in composed models for ODE parameter assignment.

See also: [`create_model_uwd`](@ref), [`generate_transition_names`](@ref), [`type_system`](@ref),
          `typed_product` (from `AlgebraicPetri.TypedPetri`), [`vectorfield_flat`](@ref)
"""
function create_model(
        typing::EpidemiologicalTyping,
        model::EpiMechModel;
        include_reflexives::Bool = true
    )
    # Materialize the typing strategy as the Petri-net codomain for the typed petri net.
    codomain = type_system(typing)

    # Create the UWD for the model (dispatch on include_reflexives via positional arg)
    uwd = _create_model_uwd(typing, model, include_reflexives)

    # Generate transition names (dispatch on model type)
    transition_names = generate_transition_names(uwd, model)

    # Apply oapply_typed to create the typed Petri net
    typed_model = oapply_typed(codomain, uwd, transition_names)

    return typed_model
end

# Intermediate layer: converts include_reflexives kwarg to positional for dispatch
_create_model_uwd(typing, model, ::Bool) = create_model_uwd(typing, model)
function _create_model_uwd(typing, model::ContactStratification, include_reflexives::Bool)
    return create_model_uwd(typing, model; include_reflexives)
end
