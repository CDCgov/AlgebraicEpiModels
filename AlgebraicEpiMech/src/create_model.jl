"""
Create a typed Petri net model from an epidemiological typing strategy and model specification.

This is the unified entry point for creating all model types in AlgebraicEpiMech. Through
multiple dispatch on the `model` parameter, it supports any subtype of `EpiMechModel`.
The undirected wiring diagram (UWD) provides an intermediate structural representation, recording
how compartments and mechanisms connect independently of the final typed Petri net constructed by
`oapply_typed`.

The resulting typed Petri net can be used directly for ODE simulation or composed with other
typed Petri nets via `typed_product` to build complex hierarchical models.

# Arguments
- `typing::EpidemiologicalTyping`: The strategy defining possible compartment and transition types.
- `model::EpiMechModel`: The model specification (uses dispatch for different types)

# Keyword Arguments
- `include_reflexives = true`: the transition types this factor is *passive* for. A
  `typed_product` keeps a transition type only if every factor has a box of that type, so a
  factor carries a reflexive box, on each junction of the matching population type, for every
  listed type it does not implement. Accepts `true` (the defaults of
  [`default_reflexives`](@ref)), `false` (none, for standalone use), or a transition type name
  or collection of them, such as `:strata` or `(:disease, :strata)`. `:strata` stands for every
  movement type of the typing (`:strata_U` and `:strata_I` under the uninfected/infected
  split), and `:death` for every death type. Listed types the factor already implements are
  skipped. `:birth` cannot be passive: declare `births` on the model specification.

# Returns
- `ACSetTransformation`: A typed Petri net with two components accessible via:
  - `dom(typed_model)`: Extract the underlying `LabelledPetriNet` for ODE solving
  - `codom(typed_model)`: Access the Petri-net type system

# Model-Specific Transition Naming
Transition names are generated automatically based on model type:
- **Compartmental**: `transmission_S_I`, `E_to_I`, `I_to_R` (describes flow between compartments)
- **Age Stratification**: `child_child`, `child_adult`, `adult_child` (infectee_infector pattern)
- **Multistrain**: Strain identifiers like `h1n1`, `h3n2` (enables clean composition)
- **Movement** (`transitions` on a `ContactStratification`): `child_to_adult`
- **Reflexive boxes**: the junction's variable (`child`, `child_U`); on a compartmental model
  `S`, `I`, `R` for movement types and the type-prefixed `waning_S` for any other
- **Births and deaths**: `birth_S`, `death_S`; composites read `birth_S_child`, `death_I_adult`

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

# Ageing: the age factor implements movement, the disease model is passive for it
sir = create_model(typing, SIR(); include_reflexives = :strata)
aging = create_model(typing, AgeStratification([:child, :adult]; transitions = [:child => :adult]))
aging_sir = dom(typed_product(sir, aging))
# Adds S_child_to_adult, I_child_to_adult, R_child_to_adult

# Births into children, deaths out of adults
sir = create_model(typing, SIR(births = [:S]); include_reflexives = (:strata, :death))
age = AgeStratification([:child, :adult]; transitions = [:child => :adult], births = [:child], deaths = [:adult])
open_sir = dom(typed_product(sir, create_model(typing, age)))
# Adds birth_S_child, death_S_adult, death_I_adult, death_R_adult
```

# Compositional Design
Models expressed as typed Petri nets compose via `typed_product` using their common
type-system codomain when box names match. The `vectorfield_flat` function automatically handles tuple transition names
in composed models for ODE parameter assignment.

See also: [`create_model_uwd`](@ref), [`generate_transition_names`](@ref), [`type_system`](@ref),
          [`default_reflexives`](@ref), `typed_product` (from `AlgebraicPetri.TypedPetri`),
          [`vectorfield_flat`](@ref)
"""
function create_model(
        typing::EpidemiologicalTyping, model::EpiMechModel; include_reflexives = true
    )
    codomain = type_system(typing)
    reflexives = reflexive_types(typing, model, include_reflexives)
    uwd = _create_model_uwd(typing, model, reflexives)
    names = generate_transition_names(uwd, model)
    # Added after naming, so the model-specific naming rules never see a junction-to-itself box.
    _add_passive_reflexives!(uwd, names, typing, model, reflexives, codomain)
    return oapply_typed(codomain, uwd, names)
end

"""
    default_reflexives(typing, model)

The transition types a factor is passive for under `include_reflexives = true`.

| Model | Default |
|---|---|
| `CompartmentalModel`, `MultiStrainModel` | none; pass `:strata` to compose with a moving stratification, `:death` with one that has deaths |
| `ContactStratification` | `:disease`, `:reversion`, `:waning` |
| `ImmuneHistory` | `:disease`, `:waning` |

Extend a default rather than restating it:
`include_reflexives = (default_reflexives(typing, model)..., :strata)`.
"""
default_reflexives(::EpidemiologicalTyping, ::EpiMechModel) = Symbol[]
default_reflexives(::EpidemiologicalTyping, ::ContactStratification) = [:disease, :reversion, :waning]
default_reflexives(::EpidemiologicalTyping, ::ImmuneHistory) = [:disease, :waning]

# `include_reflexives` resolved to the ordered, unique box names of the typing it requests.
reflexive_types(typing, model, include::Bool) =
    include ? reflexive_types(typing, model, default_reflexives(typing, model)) : Symbol[]
reflexive_types(typing, model, type::Symbol) = reflexive_types(typing, model, (type,))
function reflexive_types(typing, model, types)
    resolved = unique(Iterators.flatten(_reflexive_boxes(typing, t) for t in types))
    available = tnames(type_system(typing))
    for name in resolved
        name in available || throw(
            ArgumentError("`$name` is not a transition type of the typing; available types are $available")
        )
    end
    return resolved
end

_reflexive_boxes(typing, type::Symbol) =
    haskey(TRANSITION_KINDS, type) ? box_names(TRANSITION_KINDS[type], typing) : (type,)
_reflexive_boxes(typing, type) = throw(
    ArgumentError("`include_reflexives` takes a Bool, a transition type name or a collection of them; got $(repr(type))")
)

# Builders that add their own reflexive boxes receive the list; the others take no keyword.
_create_model_uwd(typing, model, _) = create_model_uwd(typing, model)
function _create_model_uwd(typing, model::Union{ContactStratification, ImmuneHistory}, reflexives)
    return create_model_uwd(typing, model; include_reflexives = reflexives)
end

# One passive box per junction of the right population type, for every listed type the UWD does
# not implement: a reflexive box for a type `P → P`, a death box for `P → ∅`.
function _add_passive_reflexives!(uwd, names, typing, model, reflexives, codomain)
    implemented = Set(uwd[:name])
    for type in reflexives
        type in implemented && continue
        t = findfirst(==(type), tnames(codomain))
        ins, outs = AlgebraicPetri.inputs(codomain, t), AlgebraicPetri.outputs(codomain, t)
        isempty(ins) && throw(
            ArgumentError(
                "a factor cannot be passive for `$type`: an entry must be placed, so declare " *
                    "`births` on the model specification"
            )
        )
        (length(ins) == 1 && length(outs) <= 1 && outs ⊆ ins) || throw(
            ArgumentError(
                "cannot add reflexive `$type` boxes: it does not keep to one population type, " *
                    "so a factor must implement it"
            )
        )
        population = sname(codomain, only(ins))
        for j in junctions(uwd)
            uwd[j, :junction_type] == population || continue
            box = add_box!(uwd, fill(population, 1 + length(outs)); name = type)
            foreach(port -> set_junction!(uwd, port, j), ports(uwd, box))
            push!(names, _reflexive_name(model, type, uwd[j, :variable]))
        end
    end
    return nothing
end

# A reflexive box is named after its junction. A compartmental model prefixes any non-movement
# type (`waning_S`, `death_S`) so two passive types on one compartment cannot collide, while
# movement keeps the bare name so composites read `S_child_to_adult`.
_reflexive_name(::EpiMechModel, type, variable) = variable
_reflexive_name(::CompartmentalModel, type, variable) =
    type in MOVEMENT_BOXES ? variable : Symbol(_kind_prefix(type), "_", variable)

# Death boxes are named for the kind, not the per-population box (`death_S`, not `death_U_S`).
_kind_prefix(box) = box in DEATH_BOXES ? :death : box
