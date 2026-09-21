# Functions for generating transition names from UWD structures
# Used by create_model to generate unique transition names for different model types

"""
Generate transition names from UWD structure for compartmental models.

For boxes with 2 ports (one input, one output), generates names in the format `input_to_output`.
For boxes with 4 ports (two inputs, two outputs), generates names in the format `box_name_input1_input2`
where inputs are sorted in reverse alphabetical order for consistency (e.g., `transmission_S_I`).

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
function generate_transition_names(uwd, model::CompartmentalModel)
    transition_names = Symbol[]

    for box_id in boxes(uwd)
        box_ports = ports(uwd, box_id)
        num_ports = length(box_ports)
        box_name = subpart(uwd, box_id, :name)

        # Get junctions connected to this box's ports
        box_junctions = [junction(uwd, port) for port in box_ports]

        if num_ports == 2
            # One input, one output: name as "input_to_output"
            input_junction = box_junctions[1]
            output_junction = box_junctions[2]
            input_name = subpart(uwd, input_junction, :variable)
            output_name = subpart(uwd, output_junction, :variable)
            transition_name = Symbol(input_name, "_to_", output_name)
            push!(transition_names, transition_name)
        elseif num_ports == 4
            # Two inputs, two outputs: name as "box_name_input1_input2"
            # First two ports are inputs, last two are outputs (by convention in mechanisms.jl)
            input1_junction = box_junctions[1]
            input2_junction = box_junctions[2]
            input1_name = subpart(uwd, input1_junction, :variable)
            input2_name = subpart(uwd, input2_junction, :variable)
            # Sort input names in reverse alphabetical order for consistency (e.g., S_I not I_S)
            input_names_sorted = sort(
                [string(input1_name), string(input2_name)], rev = true
            )
            transition_name = Symbol(box_name, "_", join(input_names_sorted, "_"))
            push!(transition_names, transition_name)
        else
            # Fallback: just use the box name
            # This shouldn't happen in standard compartmental models
            transition_name = box_name
            push!(transition_names, transition_name)
        end
    end

    return transition_names
end

"""
Generate transition names from UWD structure for age stratification models.

For transmission boxes with 4 ports (two inputs, two outputs), generates names capturing
the age-to-age contact pattern in the format `infectee_infector`. This naming
convention clearly identifies which age group is being infected (infectee) by which age
group (infector), enabling interpretation of age-structured contact matrices.

For boxes with 2 ports (one input, one output), generates names in the format `input_to_output`.
This fallback handles any non-standard box types that might appear in extended models.

# Arguments
- `uwd`: The undirected wiring diagram with age group names in junction :variable attributes
- `model::AgeStratification`: The age stratification model (used for dispatch)

# Returns
- `Vector{Symbol}`: Vector of transition names, one per box in the UWD

# Examples
```julia
# Used internally by create_model
typing = OnePopulationTyping()
age_strat = AgeStratification([:child, :adult])
uwd = create_model_uwd(typing, age_strat)
names = generate_transition_names(uwd, age_strat)
# Returns: [:child_child, :child_adult, :adult_child, :adult_adult]
# Representing: child←child, child←adult, adult←child, adult←adult transmission
```

See also: [`create_model`](@ref), [`ContactStratification`](@ref), [`AgeStratification`](@ref)
"""
function generate_transition_names(uwd, model::ContactStratification)
    transition_names = Symbol[]

    for box_id in boxes(uwd)
        box_ports = ports(uwd, box_id)
        num_ports = length(box_ports)
        box_name = subpart(uwd, box_id, :name)

        # Get junctions connected to this box's ports
        box_junctions = [junction(uwd, port) for port in box_ports]

        if num_ports == 4
            # Transmission: infectee_infector (2 inputs, 2 outputs)
            input1_name = subpart(uwd, box_junctions[1], :variable)
            input2_name = subpart(uwd, box_junctions[2], :variable)
            input_names = [string(input1_name), string(input2_name)]
            push!(transition_names, Symbol(join(input_names, "_")))
        elseif num_ports == 2 || num_ports == 3
            # Per-stratum reflexive boxes (disease, reversion, and waning):
            # Use first port's junction variable as the stratum name.
            # After typed_product, these compose as e.g. (:I_to_R, :child) → I_to_R_child
            first_name = subpart(uwd, box_junctions[1], :variable)
            push!(transition_names, first_name)
        else
            push!(transition_names, box_name)
        end
    end

    return transition_names
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
- **reflexive / reversion boxes** (2–3 port) → the first junction's variable, mirroring
  `ContactStratification` — e.g. the `:disease` reflexives and the off-diagonal
  `:reversion` on `(h,i)` are named by that `(h,i)`, and `:waning` by its
  `U_h`. Composition with the base disambiguates them (base name differs).

# Examples
```julia
factor = create_model_uwd(UninfectedInfectedTyping(), ImmuneHistory([:current, :invader]))
names = generate_transition_names(factor, ImmuneHistory([:current, :invader]))
# transmission names begin `infect_current…` / `infect_invader…`
```
"""
function generate_transition_names(uwd, model::ImmuneHistory)
    strains = model.strain_names
    transition_names = Symbol[]

    for box_id in boxes(uwd)
        box_ports = ports(uwd, box_id)
        box_name = subpart(uwd, box_id, :name)
        vars = [subpart(uwd, junction(uwd, port), :variable) for port in box_ports]

        if length(box_ports) == 4 && box_name == :transmission
            # Ports: [infectee U_h, infector (h',i), first-infected (h,i), infector].
            strain = _strain_of(vars[2], strains)
            susceptible = replace(string(vars[1]), r"^U_" => "")
            infector = replace(string(vars[2]), r"^.*_from_" => "")
            push!(
                transition_names,
                Symbol("infect_", strain, "_", susceptible, "_by_", infector)
            )
        else
            # Reflexive / reversion boxes: name by the first junction (ContactStratification style).
            push!(transition_names, vars[1])
        end
    end

    return transition_names
end
