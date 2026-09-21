# Observation-chain metadata for Petri nets.
#
# Observation models are attached to the petri net representing the epidemiological dynamics by
# pushout, that is assembling the joint dynamics and observation metadata on shared components.
# Here we define the shared components.

"""
    ObservationChainLayout(source_name, obs_names)

Metadata for one observation chain.

# Fields
- `source_name`: Flattened name of the event or compartment being observed
- `obs_names`: Observation-state names in stage order
- `cumulative_name`: Terminal observation-state name, derived from `obs_names`
"""
struct ObservationChainLayout{N}
    source_name::Symbol
    obs_names::NTuple{N, Symbol}
    cumulative_name::Symbol

    function ObservationChainLayout(
            source_name::Symbol, obs_names::NTuple{N, Symbol}
        ) where {N}
        N > 0 || throw(ArgumentError("An observation chain must contain at least one stage"))
        return new{N}(source_name, obs_names, last(obs_names))
    end
end

"""
    ObservationLayout(obs_names, chains)

Metadata for all observation chains in an augmented Petri net.

# Fields
- `obs_names`: All flattened observation-state names in Petri-net order
- `chains`: Observation chains in first-encounter order
- `cumulative_names`: Terminal state of each chain, derived from `chains`
"""
struct ObservationLayout{N, C <: Tuple, M}
    obs_names::NTuple{N, Symbol}
    chains::C
    cumulative_names::NTuple{M, Symbol}

    function ObservationLayout(
            obs_names::NTuple{N, Symbol}, chains::C
        ) where {N, C <: Tuple}
        all(chain -> chain isa ObservationChainLayout, chains) || throw(
            ArgumentError("All chains must be ObservationChainLayout values")
        )
        cumulative_names = Tuple(chain.cumulative_name for chain in chains)
        return new{N, C, length(cumulative_names)}(
            obs_names, chains, cumulative_names
        )
    end
end

"""
    observation_layout(pn::LabelledPetriNet; prefix::Symbol = :O)

Return deterministic observation-chain metadata for an augmented Petri net.

The result is an [`ObservationLayout`](@ref) with fields:
- `obs_names`: all flattened observation state names in Petri-net order
- `chains`: ordered [`ObservationChainLayout`](@ref) values
- `cumulative_names`: flattened terminal observation state for each chain

This centralizes the observation naming/ordering contract, so downstream packages do not each
re-implement observation-state parsing. `prefix` must match the prefix passed to
[`attach_observation`](@ref). The function recognises the `<prefix>_<source>_<stage>` leaf that
`attach_observation` produces; a chain named otherwise is not recognised as one.
"""
function observation_layout(pn; prefix::Symbol = :O)
    obs_names = Symbol[]
    chain_order = Symbol[]
    chain_stages = Dict{Symbol, Vector{Tuple{Int, Symbol}}}()

    for raw_name in snames(pn)
        meta = _observation_species_metadata(raw_name, prefix)
        isnothing(meta) && continue

        push!(obs_names, meta.obs_name)

        if !haskey(chain_stages, meta.source_name)
            chain_stages[meta.source_name] = Tuple{Int, Symbol}[]
            push!(chain_order, meta.source_name)
        end

        push!(chain_stages[meta.source_name], (meta.stage, meta.obs_name))
    end

    chains = map(chain_order) do source_name
        stage_entries = sort(chain_stages[source_name], by = first)
        stage_numbers = first.(stage_entries)
        expected_stages = collect(1:length(stage_numbers))
        stage_numbers == expected_stages || throw(
            ArgumentError(
                "Observation chain for :$source_name has non-consecutive stages $stage_numbers"
            )
        )

        stage_names = Tuple(last.(stage_entries))
        ObservationChainLayout(source_name, stage_names)
    end

    return ObservationLayout(Tuple(obs_names), Tuple(chains))
end

function _observation_species_metadata(name, prefix)
    leaves = _flatten_symbol_leaves(name)
    parsed = map(leaf -> _parse_observation_stage(leaf, prefix), leaves)
    matches = findall(!isnothing, parsed)

    isempty(matches) && return nothing
    length(matches) == 1 || throw(
        ArgumentError("Expected at most one observation-stage leaf in $name")
    )

    match_idx = only(matches)
    stage_info = something(parsed[match_idx])

    source_leaves = copy(leaves)
    source_leaves[match_idx] = stage_info.source

    return (
        obs_name = flatten_symbols(name),
        source_name = Symbol(join(string.(source_leaves), "_")),
        stage = stage_info.stage,
    )
end

_flatten_symbol_leaves(name::Symbol) = [name]

function _flatten_symbol_leaves(name::Tuple)
    leaves = Symbol[]
    for item in name
        append!(leaves, _flatten_symbol_leaves(item))
    end
    return leaves
end

function _parse_observation_stage(name::Symbol, prefix::Symbol)
    prefix_marker = string(prefix, "_")
    raw_name = String(name)
    startswith(raw_name, prefix_marker) || return nothing

    parts = split(chopprefix(raw_name, prefix_marker), "_")
    length(parts) >= 2 || return nothing
    stage = tryparse(Int, parts[end])
    isnothing(stage) && return nothing

    source = Symbol(join(parts[1:(end - 1)], "_"))
    isempty(String(source)) && return nothing

    return (source = source, stage = stage)
end
