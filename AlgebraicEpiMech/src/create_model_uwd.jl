# Constructors for compartmental epidemiological models
# Uses functions from construction_helpers.jl and mechanisms.jl
# Dispatch methods to create compartmental models based on typing and model instances

"""
Create an undirected wiring diagram (UWD) for a compartmental model within a specific typing.

This function returns the undirected wiring diagram that defines the
compartmental structure and transitions, using the type names from the typing.
The compartments are typed according to the typing's population structure.
The returned UWD does not declare an outer boundary/ports; `oapply_typed` derives the
boundary from its junctions when materializing the typed Petri net.

# Arguments
- `typing::EpidemiologicalTyping`: The population typing defining type names
- `model::CompartmentalModel`: The compartmental model instance (SI(), SEI(), SIR(), SEIR(), etc.)

# Returns
- Undirected wiring diagram defining the compartmental model structure with typing-specific types

# Examples
```julia
# Create SIR model UWD with single population typing
typing = OnePopulationTyping(population_type = :Individual)
sir_uwd = create_model_uwd(typing, SIR())  # Uses Individual type

# Create SEIR model UWD with uninfected/infected typing
typing = UninfectedInfectedTyping(uninfected_type = :Susceptible, infected_type = :Infectious)
seir_uwd = create_model_uwd(typing, SEIR())  # S::Susceptible, E,I,R::Infectious
```
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::CompartmentalModel)
    error("create_model_uwd not implemented for typing type $(typeof(typing)) and model type $(typeof(model)). Implement a method for this combination.")
end

"""
Construct a complete undirected wiring diagram (UWD) for SI or SEI models.

Populates an initially empty UWD with compartments and mechanisms via `setup_basic!`.
These are the base models that other compartmental models extend from.

# Usage with oapply_typed
The returned UWD can be applied to the typing using `oapply_typed` to create a typed
Petri net, enabling composition with other typed Petri nets (e.g., demographic processes,
interventions).
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::Union{SI, SEI})
    uwd = RelationDiagram(Symbol[])
    (uwd, _, _) = setup_basic!(uwd, typing, model)
    return uwd
end

"""
Construct a complete SIR model UWD by extending SI with recovery.

Builds on the SI base model (S→I dynamics with multi-stage I support) and adds:
- R (recovered) compartment
- Recovery transition: I_last → R

The multi-stage I capability from SI is preserved, enabling gamma-distributed infectious
periods. Recovery always occurs from the last I stage, ensuring proper sequencing.

# Usage with oapply_typed
Apply to typing with `oapply_typed` to create a typed Petri net for composition.
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::SIR)
    uwd = RelationDiagram(Symbol[])

    # Get SI base model
    uwd, _,
        last_I_junction = setup_basic!(uwd, typing, SI(number_I_stages = model.number_I_stages))

    pop_type = get_infected_type(typing)

    # Add R compartment junction
    R_junction = add_junction!(uwd, pop_type, variable = :R)

    # Add recovery: last I stage → R
    add_disease_progression!(uwd, last_I_junction, R_junction, typing)

    return uwd
end

"""
Construct a complete SIS model UWD by extending SI with waning immunity.

Builds on the SI base model (S→I dynamics with multi-stage I support) and adds:
- Reversion transition: I_last → S (waning immunity)

Represents diseases where recovery returns individuals to susceptible state. The multi-stage
I capability enables realistic infectious period distributions. Reversion occurs from the
last I stage back to S.

# Usage with oapply_typed
Apply to typing with `oapply_typed` to create a typed Petri net for composition.
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::SIS)
    uwd = RelationDiagram(Symbol[])

    # Get SI base model
    uwd, S_junction,
        last_I_junction = setup_basic!(uwd, typing, SI(number_I_stages = model.number_I_stages))

    # Add reversion: last I stage → S (waning immunity)
    add_reversion_progression!(uwd, last_I_junction, S_junction, typing)

    return uwd
end

"""
Construct a complete SEIR model UWD by extending SEI with recovery.

Builds on the SEI base model (S→E→I dynamics with multi-stage E and I support) and adds:
- R (recovered) compartment
- Recovery transition: I_last → R

The dual multi-stage capability (independent E and I stages) enables independent control
of latent and infectious period distributions. Recovery always occurs from the last I stage.

# Usage with oapply_typed
Apply to typing with `oapply_typed` to create a typed Petri net for composition.
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::SEIR)
    uwd = RelationDiagram(Symbol[])

    # Get SEI base model
    uwd, _,
        last_I_junction = setup_basic!(
        uwd,
        typing,
        SEI(
            number_E_stages = model.number_E_stages,
            number_I_stages = model.number_I_stages
        )
    )

    pop_type = get_infected_type(typing)

    # Add R compartment junction
    R_junction = add_junction!(uwd, pop_type, variable = :R)

    # Add recovery: last I stage → R
    add_disease_progression!(uwd, last_I_junction, R_junction, typing)

    return uwd
end

"""
Construct a complete SEIS model UWD by extending SEI with waning immunity.

Builds on the SEI base model (S→E→I dynamics with multi-stage E and I support) and adds:
- Reversion transition: I_last → S (waning immunity)

Represents diseases with latent periods but no lasting immunity. The dual multi-stage
capability enables independent control of latent and infectious period distributions.
Reversion occurs from the last I stage back to S.

# Usage with oapply_typed
Apply to typing with `oapply_typed` to create a typed Petri net for composition.
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::SEIS)
    uwd = RelationDiagram(Symbol[])

    # Get SEI base model
    uwd, S_junction,
        last_I_junction = setup_basic!(
        uwd,
        typing,
        SEI(
            number_E_stages = model.number_E_stages,
            number_I_stages = model.number_I_stages
        )
    )

    # Add reversion: last I stage → S (waning immunity)
    add_reversion_progression!(uwd, last_I_junction, S_junction, typing)

    return uwd
end

"""
Construct a complete SEIRS model UWD by extending SEI with recovery and waning immunity.

Builds on the SEI base model (S→E→I dynamics with multi-stage E and I support) and adds:
- R (recovered) compartment
- Recovery transition: I_last → R
- Waning immunity transition: R → S

Represents diseases with latent periods and temporary immunity. The dual multi-stage capability
enables independent control of latent and infectious period distributions. Recovery occurs from
the last I stage, and waning immunity returns individuals from R to S.

# Usage with oapply_typed
Apply to typing with `oapply_typed` to create a typed Petri net for composition.
"""
function create_model_uwd(typing::EpidemiologicalTyping, model::SEIRS)
    uwd = RelationDiagram(Symbol[])

    # Get SEIR model (which already calls SEI and adds R)
    # SEIR returns just uwd since it's already complete
    uwd, S_junction,
        last_I_junction = setup_basic!(
        uwd,
        typing,
        SEI(
            number_E_stages = model.number_E_stages,
            number_I_stages = model.number_I_stages
        )
    )

    # Add R compartment junction
    pop_type = get_infected_type(typing)
    R_junction = add_junction!(uwd, pop_type, variable = :R)

    # Add recovery: last I stage → R
    add_disease_progression!(uwd, last_I_junction, R_junction, typing)

    # Add waning immunity: R → S
    add_reversion_progression!(uwd, R_junction, S_junction, typing)

    return uwd
end

# ============================================================================
# Multistrain Model UWD Construction
# ============================================================================


"""
Construct an undirected wiring diagram (UWD) for a no cross-immunity multistrain model.

Creates a strain-stratified UWD where each strain operates independently. The strains
are represented as separate junctions, and each strain has its own set of disease
transition boxes (`:transmission`, `:disease`, `:reversion`, `:observation`) that will
compose with corresponding boxes from a compartmental model via `typed_product`.

All box types are always included; `typed_product` will only compose boxes that exist in
both UWDs, naturally filtering out non-matching transitions. That filtering is silent — a
transition present in one factor and absent from the other is dropped without error — which is
why each factor carries a reflexive box per transition type it wants preserved.

There is deliberately no reflexive observation box. Observation is attached to the COMPOSED net
by pushout (`attach_observation`), so a stratification factor never has to know it exists; see
`docs/concepts/composition-and-observation.md`.

# Arguments
- `typing::EpidemiologicalTyping`: The population typing (typically OnePopulationTyping)
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
sir_typed = create_compartmental_model(typing, SIR())
strain_typed = create_multistrain_model(typing, multistrain)
combined = typed_product(sir_typed, strain_typed)
```
"""
function create_model_uwd(
        typing::EpidemiologicalTyping,
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
- `typing::EpidemiologicalTyping`: The population typing (must be UninfectedInfectedTyping)
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
sir_typed = create_compartmental_model(typing, SIR())
strain_typed = create_multistrain_model(typing, multistrain)
combined = typed_product(sir_typed, strain_typed)
```
"""
function create_model_uwd(
        typing::EpidemiologicalTyping,
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
"""
function create_model_uwd(
        typing::EpidemiologicalTyping, model::ContactStratification;
        include_reflexives::Bool = true
    )
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
            add_infection!(
                uwd,
                stratum_infectee[1],  # infectee junction (the stratum being infected)
                stratum_infector[2],   # infector junction (the stratum causing infection)
                stratum_infectee[2],   # first infected output junction (the stratum being infected is unchanged for ContactStratification)
                typing
            )
        end
    end

    # Add per-stratum reflexive boxes for non-transmission transitions.
    # These ensure typed_product preserves disease, reversion, waning, and
    # observation transitions when composing with compartmental models.
    # Can be disabled via include_reflexives=false for standalone use.
    if include_reflexives
        for stratum in stratum_junction_tuples
            add_disease_progression!(uwd, stratum[2], stratum[2], typing)
            add_reversion_progression!(uwd, stratum[2], stratum[1], typing)
            add_uninfected_density_progression!(uwd, stratum[1], stratum[1], typing)
        end
    end

    return uwd
end

# ============================================================================
# Immune-history model UWD construction (issue #279)
# ============================================================================

"""
Immune-history stratification requires the uninfected/infected type split.
"""
function create_model_uwd(::OnePopulationTyping, ::ImmuneHistory)
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
- **`:disease`** and **`:observation`** are reflexive on each `(h,i)`, so the base
  model's progression and observation chains run inside a fixed `(h,i)`.
- **`:reversion`** is off-diagonal: `(h,i) → U_{recover}` with `recover = h∪{i}`
  (`FullHistory`) or `{i}` (`LatestInfection`). This is the one stratum-changing move
  — recovering from `i` folds it into the immune history.
- **`:waning`** reflexive on each `U_h` (inert with SEIRS; lets a waning-bearing base
  compose).

# Examples
```julia
typing = UninfectedInfectedTyping()
history = create_model(typing, ImmuneHistory([:current, :invader]))
model = typed_product(create_model(typing, SEIRS()), history)
```
"""
function create_model_uwd(typing::UninfectedInfectedTyping, model::ImmuneHistory)
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
        # Reflexive :disease and :observation so the base's E→I→R and obs chain
        # run within this (h,i).
        add_disease_progression!(uwd, Iof[(h, i)], Iof[(h, i)], typing)
        # Off-diagonal :reversion — the history-incrementing move.
        add_reversion_progression!(uwd, Iof[(h, i)], U[_recover_to(mode, h, i)], typing)
    end

    # Reflexive :waning on each uninfected class (inert with SEIRS).
    for h in keys(U)
        add_uninfected_density_progression!(uwd, U[h], U[h], typing)
    end

    return uwd
end
