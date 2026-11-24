# Functions for generating transition names from UWD structures
# Used by create_model to generate unique transition names for different model types

"""
Generate transition names from UWD structure for compartmental models.

For boxes with 2 ports (one input, one output), generates names in the format `input_to_output`.
For boxes with 4 ports (two inputs, two outputs), generates names in the format `box_name_input1_input2`
where inputs are sorted in reverse alphabetical order for consistency (e.g., `transmission_S_I`).
Birth and death boxes have one port and are named `birth_S`, `death_I`. An importation box is
named `importation_S_to_E`.

# Arguments
- `uwd`: The undirected wiring diagram with species names in junction :variable attributes
- `model::CompartmentalModel`: The compartmental model (used for dispatch)

# Returns
- `Vector{Symbol}`: Vector of unique transition names, one per box in the UWD

# Examples
```julia
# Used internally by create_model
typing = OnePopulationTyping()
uwd = create_model_uwd(typing, SEIR())
names = generate_transition_names(uwd, SEIR())
# Returns: [:transmission_S_I, :E_to_I, :I_to_R]
```

See also: [`create_model`](@ref)
"""
function generate_transition_names(uwd, ::CompartmentalModel)
    return map(boxes(uwd)) do box
        vars = [uwd[junction(uwd, port), :variable] for port in ports(uwd, box)]
        name = uwd[box, :name]
        length(vars) == 1 && return Symbol(_kind_prefix(name), "_", only(vars))
        name == :importation && return Symbol(name, "_", vars[1], "_to_", vars[2])
        length(vars) == 2 && return Symbol(vars[1], "_to_", vars[2])
        # The first two of four ports are the inputs, sorted so `S_I` never reads `I_S`.
        length(vars) == 4 && return Symbol(name, "_", join(sort(string.(vars[1:2]); rev = true), "_"))
        return name
    end
end

"""
Generate transition names for a `ContactStratification` UWD.

- Transmission boxes are named `infectee_infector`, e.g. `child_adult` for a child infected
  by an adult.
- Movement boxes are named `source_to_target`, e.g. `child_to_adult` or `child_U_to_adult_U`.
- Reflexive, birth and death boxes take their stratum's junction variable, e.g. `child`, so a
  composite reads `I_to_R_child` or `birth_S_child`.

# Examples
```julia
age = AgeStratification([:child, :adult])
uwd = create_model_uwd(OnePopulationTyping(), age; include_reflexives = false)
generate_transition_names(uwd, age)   # [:child_child, :child_adult, :adult_child, :adult_adult]
```

See also: [`create_model`](@ref), [`ContactStratification`](@ref), [`AgeStratification`](@ref)
"""
function generate_transition_names(uwd, ::ContactStratification)
    return map(boxes(uwd)) do box
        vars = [uwd[junction(uwd, port), :variable] for port in ports(uwd, box)]
        name = uwd[box, :name]
        name == :transmission && return Symbol(vars[1], "_", vars[2])
        name in MOVEMENT_BOXES && return Symbol(vars[1], "_to_", vars[2])
        return vars[1]
    end
end


"""
Generate transition names for a multistrain model from UWD structure.

For multistrain models, transition names are just the strain names. This creates cleaner
composed transition names like `(:transmission_S_I, :h1n1)` rather than
`(:transmission_S_I, :trans_h1n1)`. The strain model UWD is never used alone, only for composition.

For NoCrossImmunity: All box ports connect to the same strain junction.
For CompleteCrossImmunity: Boxes connect to both shared susceptible and strain-specific infected junctions.

# Arguments
- `uwd`: The undirected wiring diagram with strain junctions
- `multistrain::MultiStrainModel`: The multistrain model containing strain names

# Returns
- `Vector{Symbol}`: Vector of transition names (just strain names), one per box in the UWD

# Examples
```julia
# Used internally by create_model
typing = OnePopulationTyping()
multistrain = NoCrossImmunity([:h1n1, :h3n2])
uwd = create_model_uwd(typing, multistrain)
names = generate_transition_names(uwd, multistrain)
# Returns one strain name per box; 3 boxes per strain
# (transmission, disease, reversion), so 2 strains → 6 names.
```
"""
function _strain_junction(uwd, box_id, ::NoCrossImmunity)
    box_junctions = unique([junction(uwd, port) for port in ports(uwd, box_id)])
    return only(box_junctions)
end

function _strain_junction(uwd, box_id, ::CompleteCrossImmunity)
    box_junctions = unique([junction(uwd, port) for port in ports(uwd, box_id)])
    strain_junctions = filter(box_junctions) do junction_id
        return subpart(uwd, junction_id, :variable) != :susceptible
    end
    return only(strain_junctions)
end

function generate_transition_names(uwd, model::MultiStrainModel)
    transition_names = Symbol[]

    for box_id in boxes(uwd)
        strain_junction = _strain_junction(uwd, box_id, model)
        strain_variable = subpart(uwd, strain_junction, :variable)

        # Transition name is just the strain name
        # This creates clean composed names like (:transmission_S_I, :h1n1)
        push!(transition_names, strain_variable)
    end

    return transition_names
end

"""
Generate transition names for the `ImmuneHistory` stratification factor. These become
the second component of the composed `typed_product` transition names.

- **`:transmission`** (4-port) → `infect_<strain>_<susceptible-class>_by_<infector-class>`,
  exposing the infecting strain (recovered from the infector junction) so the composed
  name stays keyable per strain.
- **`:waning` edges** (`waning = true`) → `wane_<strain>_<class>`: the strain forgotten, then
  the class it is forgotten from, e.g. `wane_b_a_b` for `U_a_b → U_a`.
- **`:importation`** (`U_h → (h,i)`) → `import_<strain>_<class>`, e.g. `import_b_naive`.
- **reflexive / reversion boxes** (2–3 port) → the first junction's variable, mirroring
  `ContactStratification` — e.g. the `:disease` reflexives and the off-diagonal
  `:reversion` on `(h,i)` are named by that `(h,i)`, and a reflexive `:waning` by its
  `U_h`. Composition with the base disambiguates them (base name differs).

Every name that changes stratum starts with its strain, so a per-strain rate is keyed by
prefix in the composed name.

# Examples
```julia
factor = create_model_uwd(UninfectedInfectedTyping(), ImmuneHistory([:current, :invader]))
names = generate_transition_names(factor, ImmuneHistory([:current, :invader]))
# transmission names begin `infect_current…` / `infect_invader…`
```
"""
function generate_transition_names(uwd, model::ImmuneHistory)
    strains = model.strain_names
    class_of = Dict(_uninfected_name(h) => h for h in _immune_classes(model.mode, strains))
    return map(boxes(uwd)) do box
        vars = [uwd[junction(uwd, port), :variable] for port in ports(uwd, box)]
        name = uwd[box, :name]
        # Infection ports: [infectee U_h, infector (h',i), first-infected (h,i), infector].
        if name == :transmission
            infector = replace(string(vars[2]), r"^.*_from_" => "")
            return Symbol("infect_", _strain_of(vars[2], strains), "_", _htag(class_of[vars[1]]), "_by_", infector)
        elseif name == :importation
            return Symbol("import_", _strain_of(vars[2], strains), "_", _htag(class_of[vars[1]]))
        elseif name == :waning && vars[1] != vars[2]
            from = class_of[vars[1]]
            return Symbol("wane_", only(setdiff(from, class_of[vars[2]])), "_", _htag(from))
        end
        return vars[1]
    end
end
