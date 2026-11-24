# The undirected wiring diagram of each model specification.

"""
    create_model_uwd(typing, model::CompartmentalModel)

The undirected wiring diagram (UWD) of a compartmental model under `typing`: one junction per
compartment, typed by the typing's population types, and one box per transition, plus the
births and deaths declared by the model's [`Demography`](@ref). The UWD declares no outer
ports; `oapply_typed` derives the boundary from its junctions.

Multi-stage compartments (`number_E_stages`, `number_I_stages`) are chained, and recovery or
reversion always leaves the last infectious stage.

# Examples
```julia
sir_uwd = create_model_uwd(OnePopulationTyping(population_type = :Individual), SIR())
typing = UninfectedInfectedTyping(uninfected_type = :Susceptible, infected_type = :Infectious)
seir_uwd = create_model_uwd(typing, SEIR(births = [:S]))  # S::Susceptible, E, I, R::Infectious
```
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::CompartmentalModel)
    uwd = _compartments_uwd(typing, model)
    compartments = Dict(uwd[j, :variable] => [j] for j in junctions(uwd))
    for name in (model.demography.births..., model.demography.deaths...)
        haskey(compartments, name) || throw(
            ArgumentError(
                "`$name` is not a compartment of $(nameof(typeof(model))); " *
                    "compartments are $(uwd[:variable])"
            )
        )
    end
    return _add_demography!(uwd, typing, model.demography, compartments)
end

function _compartments_uwd(typing, model::CompartmentalModel)
    return error("create_model_uwd not implemented for $(typeof(typing)) and $(typeof(model))")
end

function _compartments_uwd(typing, model::Union{SI, SEI})
    uwd, _, _ = setup_basic!(RelationDiagram(Symbol[]), typing, model)
    return uwd
end

function _compartments_uwd(typing, model::SIR)
    uwd, _, last_I = setup_basic!(RelationDiagram(Symbol[]), typing, SI(number_I_stages = model.number_I_stages))
    R = add_junction!(uwd, get_infected_type(typing), variable = :R)
    add_disease_progression!(uwd, last_I, R, typing)
    return uwd
end

function _compartments_uwd(typing, model::SIS)
    uwd, S, last_I = setup_basic!(RelationDiagram(Symbol[]), typing, SI(number_I_stages = model.number_I_stages))
    add_reversion_progression!(uwd, last_I, S, typing)
    return uwd
end

_latent_base(model) = SEI(number_E_stages = model.number_E_stages, number_I_stages = model.number_I_stages)

function _compartments_uwd(typing, model::SEIR)
    uwd, _, last_I = setup_basic!(RelationDiagram(Symbol[]), typing, _latent_base(model))
    R = add_junction!(uwd, get_infected_type(typing), variable = :R)
    add_disease_progression!(uwd, last_I, R, typing)
    return uwd
end

function _compartments_uwd(typing, model::SEIS)
    uwd, S, last_I = setup_basic!(RelationDiagram(Symbol[]), typing, _latent_base(model))
    add_reversion_progression!(uwd, last_I, S, typing)
    return uwd
end

function _compartments_uwd(typing, model::SEIRS)
    uwd, S, last_I = setup_basic!(RelationDiagram(Symbol[]), typing, _latent_base(model))
    R = add_junction!(uwd, get_infected_type(typing), variable = :R)
    add_disease_progression!(uwd, last_I, R, typing)
    add_reversion_progression!(uwd, R, S, typing)
    return uwd
end

# ============================================================================
# Multistrain Model UWD Construction
# ============================================================================


"""
Construct an undirected wiring diagram (UWD) for a no cross-immunity multistrain model.

Creates a strain-stratified UWD where each strain operates independently. The strains
are represented as separate junctions, and each strain has its own set of disease
transition boxes (`:transmission`, `:disease`, `:reversion`) that will
compose with corresponding boxes from a compartmental model via `typed_product`.

All box types are always included; `typed_product` will only compose boxes that exist in
both UWDs, naturally filtering out non-matching transitions. That filtering is silent — a
transition present in one factor and absent from the other is dropped without error — which is
why each factor carries a reflexive box per transition type it wants preserved.

There is deliberately no reflexive observation box. Observation is attached to the COMPOSED net
by `attach_observation`, so a stratification factor never has to know it exists.

# Arguments
- `typing::OnePopulationTyping`: The required single-population typing
- `multistrain::NoCrossImmunity`: The multistrain model configuration with strain names

# Returns
- Undirected wiring diagram with strain junctions and transition boxes for composition

# Examples
```julia
# Create a 3-strain model with custom names
typing = OnePopulationTyping()
multistrain = NoCrossImmunity([:h1n1, :h3n2, :b])
strain_uwd = create_model_uwd(typing, multistrain)

# Compose with compartmental model
sir_typed = create_model(typing, SIR())
strain_typed = create_model(typing, multistrain)
combined = typed_product(sir_typed, strain_typed)
```
"""
function create_model_uwd(
        typing::OnePopulationTyping,
        multistrain::NoCrossImmunity
    )
    strain_names = multistrain.strain_names

    # For NoCrossImmunity, use OnePopulationTyping type system
    pop_type = get_infected_type(typing)

    uwd = RelationDiagram(Symbol[])

    # Create junction for each strain
    strain_junctions = [
        add_junction!(uwd, pop_type, variable = strain_name)
            for strain_name in strain_names
    ]

    # Add all transition boxes for each strain
    # typed_product will only compose boxes that match between UWDs
    for strain_junction in strain_junctions
        # Transmission: strain + strain → strain + strain (like S + I → I + I)
        add_infection!(uwd, strain_junction, strain_junction, strain_junction, typing)

        # Disease progression: strain → strain (like E → I or I → R)
        add_disease_progression!(uwd, strain_junction, strain_junction, typing)

        # Reversion/waning: strain → strain (like I → S or R → S)
        add_reversion_progression!(uwd, strain_junction, strain_junction, typing)

    end

    return uwd
end

"""
Construct an undirected wiring diagram (UWD) for a complete cross-immunity multistrain model.

Creates a strain-structured UWD where all strains share a common susceptible pool but have
separate infected compartments. Infection by any strain depletes the shared susceptible pool
and confers immunity to all strains.

The UWD has:
- One shared uninfected/susceptible junction
- N infected strain junctions
- Transmission boxes connecting shared susceptible to each strain's infected
- Disease progression boxes for each strain's infected compartments
- Reversion boxes to return to shared susceptible pool

# Arguments
- `typing::UninfectedInfectedTyping`: The required uninfected/infected typing
- `multistrain::CompleteCrossImmunity`: The multistrain model configuration with strain names

# Returns
- Undirected wiring diagram with shared susceptible and strain-specific infected junctions

# Examples
```julia
# Create a 2-strain competing model
typing = UninfectedInfectedTyping()
multistrain = CompleteCrossImmunity([:wild_type, :variant])
strain_uwd = create_model_uwd(typing, multistrain)

# Compose with compartmental model
sir_typed = create_model(typing, SIR())
strain_typed = create_model(typing, multistrain)
combined = typed_product(sir_typed, strain_typed)
```
"""
function create_model_uwd(
        typing::UninfectedInfectedTyping,
        multistrain::CompleteCrossImmunity
    )
    strain_names = multistrain.strain_names

    # For CompleteCrossImmunity, use UninfectedInfectedTyping type system
    uninfected_type = get_uninfected_type(typing)
    infected_type = get_infected_type(typing)

    uwd = RelationDiagram(Symbol[])

    # Create shared uninfected/susceptible junction
    shared_uninfected = add_junction!(uwd, uninfected_type, variable = :susceptible)

    # Create infected junction for each strain
    infected_junctions = [
        add_junction!(uwd, infected_type, variable = strain_name)
            for strain_name in strain_names
    ]

    # Add transition boxes for each strain
    for infected_junction in infected_junctions
        # Transmission: shared_S + strain_I → strain_I + strain_I
        # This connects the shared susceptible pool to each strain's infected compartment
        add_infection!(uwd, shared_uninfected, infected_junction, infected_junction, typing)

        # Disease progression: strain_I → strain_I (e.g., I → R within strain)
        add_disease_progression!(uwd, infected_junction, infected_junction, typing)

        # Reversion/waning: strain_I → shared_S (e.g., R → S or I → S)
        # This returns individuals to the shared susceptible pool
        add_reversion_progression!(uwd, infected_junction, shared_uninfected, typing)

    end

    return uwd
end

"""
Construct an undirected wiring diagram (UWD) for contact strata.

Creates an UWD where transmission boxes represent contact patterns between different strata.
Example strata include age groups, risk levels, or demographic divisions, but only covers instantaneous
transmission dynamics, e.g. movement between geographic regions would require additional demographic modeling.

The UWD structure depends on the typing:
- `OnePopulationTyping`: Single junction per age group (combined susceptible/infected)
- `UninfectedInfectedTyping`: Two junctions per age group (uninfected and infected)

All transmission boxes are created with the `:transmission` name, enabling composition
with compartmental models that have `:transmission` boxes (SI, SIR, SEIR, etc.).

# Arguments
- `typing::EpidemiologicalTyping`: The population typing defining type system
- `model::ContactStratification`: The contact stratification configuration with stratum names

# Returns
- Undirected wiring diagram with stratum junctions and transmission boxes for composition

# Mathematical Structure
For n strata, creates n² transmission boxes representing the contact matrix:
- Diagonal boxes (i→i): within-stratum transmission
- Off-diagonal boxes (i→j, i≠j): between-stratum transmission


# Composition Behavior
When composed with a compartmental model UWD via `typed_product`, the transmission boxes
will align based on the `:transmission` name, allowing the contact structure to modulate
the disease dynamics defined in the compartmental model.

The movement graph (`model.transitions`) adds one movement box per edge and population type
through [`add_strata_transition!`](@ref). Declared births enter a stratum's uninfected junction
([`add_birth!`](@ref)) and declared deaths leave each of its junctions ([`add_death!`](@ref)).

# Keyword Arguments
- `include_reflexives = true`: the transition types this factor is passive for (see
  [`create_model`](@ref)). The per-stratum `:disease`, `:reversion` and `:waning` boxes are
  added when listed; any other listed type is added by `create_model`.
"""
function create_model_uwd(
        typing::EpidemiologicalTyping, model::ContactStratification; include_reflexives = true
    )
    reflexives = reflexive_types(typing, model, include_reflexives)
    uwd = RelationDiagram(Symbol[])

    # Create junction for each stratum
    # These are tuples (uninfected_stratum_junction, infected_stratum_junction) in case needed for more complex typings
    stratum_junction_tuples = [
        set_stratum_junction!(uwd, typing, stratum)
            for stratum in model.stratum_names
    ]

    for stratum_infectee in stratum_junction_tuples
        for stratum_infector in stratum_junction_tuples
            # Add transmission between strata (full contact matrix)
            add_infection!(uwd, stratum_infectee[1], stratum_infector[2], stratum_infectee[2], typing)
        end
    end

    for (uninfected, infected) in stratum_junction_tuples
        :disease in reflexives && add_disease_progression!(uwd, infected, infected, typing)
        :reversion in reflexives && add_reversion_progression!(uwd, infected, uninfected, typing)
        :waning in reflexives && add_uninfected_density_progression!(uwd, uninfected, uninfected, typing)
    end

    # One movement box per edge and population type; under OnePopulationTyping both junctions
    # of a stratum are the same one.
    junctions_of = Dict(zip(model.stratum_names, unique.(stratum_junction_tuples)))
    for (source, target) in model.transitions
        for (from, to) in zip(junctions_of[source], junctions_of[target])
            add_strata_transition!(uwd, from, to, typing)
        end
    end

    return _add_demography!(uwd, typing, model.demography, junctions_of)
end

# ============================================================================
# Immune-history model UWD construction
# ============================================================================

"""
Immune-history stratification requires the uninfected/infected type split.
"""
function create_model_uwd(::OnePopulationTyping, ::ImmuneHistory; kwargs...)
    return error(
        "ImmuneHistory requires UninfectedInfectedTyping — the uninfected/infected " *
            "split is what carries immune history; OnePopulationTyping has no uninfected type."
    )
end

"""
Construct the UWD for the `ImmuneHistory` stratification **factor** over a
`UninfectedInfectedTyping`.

This is a stratification factor (like `ContactStratification`), meant to be
`typed_product`-composed with a disease model — the disease model supplies
`S→E→I→R`, while this factor supplies the immune-status structure.

- Uninfected junctions `U_h`, one per immune-history class.
- Infected junctions `(h,i)` = "history `h`, currently fighting strain `i`", one per
  (class, susceptible-strain) pair.
- **`:transmission`** (escape-selective): `U_h + (h',i) → (h,i)` for every susceptible
  class `h` (`i ∉ h`) and every infector `(h',i)`. A class immune to `i` simply has no
  such box, so the pullback drops that infection — the escape.
- **`:disease`** is reflexive on each `(h,i)`, so the base model's progression runs
  inside a fixed `(h,i)`.
- **`:reversion`** is off-diagonal: `(h,i) → U_{recover}` with `recover = h∪{i}`
  (`FullHistory`) or `{i}` (`LatestInfection`). This is the one stratum-changing move
  — recovering from `i` folds it into the immune history.
- **`:waning`** reflexive on each `U_h` (inert with SEIRS; lets a waning-bearing base
  compose).

The `:disease` and `:waning` reflexives are added when listed in `include_reflexives` (see
[`create_model`](@ref)); both are by default.

# Examples
```julia
typing = UninfectedInfectedTyping()
history = create_model(typing, ImmuneHistory([:current, :invader]))
model = typed_product(create_model(typing, SEIRS()), history)
```
"""
function create_model_uwd(
        typing::UninfectedInfectedTyping, model::ImmuneHistory; include_reflexives = true
    )
    reflexives = reflexive_types(typing, model, include_reflexives)
    strains = model.strain_names
    mode = model.mode
    uninfected_type = get_uninfected_type(typing)
    infected_type = get_infected_type(typing)

    uwd = RelationDiagram(Symbol[])

    # Uninfected immune-history classes.
    U = Dict{_ImmuneClass, Int}()
    for h in _immune_classes(mode, strains)
        U[h] = add_junction!(uwd, uninfected_type, variable = _uninfected_name(h))
    end

    # Infected (history, current-strain) junctions; index the infectors of each strain.
    pairs = _infected_pairs(mode, strains)
    Iof = Dict{Tuple{_ImmuneClass, Symbol}, Int}()
    infectors = Dict{Symbol, Vector{Int}}(s => Int[] for s in strains)
    for (h, i) in pairs
        Iof[(h, i)] = add_junction!(uwd, infected_type, variable = _infected_name(h, i))
        push!(infectors[i], Iof[(h, i)])
    end

    for (h, i) in pairs
        # Escape-selective infection: U_h + (any strain-i infector) → (h,i).
        for inf_j in infectors[i]
            add_infection!(uwd, U[h], inf_j, Iof[(h, i)], typing)
        end
        # Reflexive :disease so the base's E→I→R progression runs within this (h,i).
        :disease in reflexives && add_disease_progression!(uwd, Iof[(h, i)], Iof[(h, i)], typing)
        # Off-diagonal :reversion — the history-incrementing move.
        add_reversion_progression!(uwd, Iof[(h, i)], U[_recover_to(mode, h, i)], typing)
    end

    if :waning in reflexives
        # Reflexive :waning on each uninfected class (inert with SEIRS), in class order.
        for h in _immune_classes(mode, strains)
            add_uninfected_density_progression!(uwd, U[h], U[h], typing)
        end
    end

    return uwd
end
