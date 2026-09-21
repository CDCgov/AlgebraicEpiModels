# Observation-chain metadata for Petri nets.
#
# Observation is attached to a composed net by PUSHOUT (`src/observation_rewriting.jl`), not by
# composing an observation factor into it. What survives here is the READ side: given a net that
# already carries observation chains, recover which species belong to which chain and in what
# order, so downstream packages do not each re-implement the parsing.
#
# The construction side — `WithObservation`, `ObservationChainSpec`, `add_observation_chain!` and
# the UWD plumbing they needed — was removed once every submodel had migrated to
# `attach_observation`. See `docs/concepts/composition-and-observation.md` for why observation is
# a colimit rather than an operadic composition.

"""
    observation_layout(pn::LabelledPetriNet)

Return deterministic observation-chain metadata for an augmented Petri net.

The result is a NamedTuple with fields:
- `obs_names`: all flattened observation state names in Petri-net order
- `chains`: ordered chain metadata tuples with `source_name`, `obs_names`, and `cumulative_name`
- `cumulative_names`: flattened terminal observation state for each chain

This centralizes the observation naming/ordering contract, so downstream packages do not each
re-implement observation-state parsing. It expects the `<prefix>_<source>_<stage>` leaf that
[`attach_observation`](@ref) produces; a chain named otherwise is not recognised as one.
"""
function observation_layout(pn)
    obs_names = Symbol[]
    chain_order = Symbol[]
    chain_stages = Dict{Symbol, Vector{Tuple{Int, Symbol}}}()

    for raw_name in snames(pn)
        meta = _observation_species_metadata(raw_name)
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
        (
            source_name = source_name,
            obs_names = stage_names,
            cumulative_name = last(stage_names),
        )
    end

    return (
        obs_names = Tuple(obs_names),
        chains = Tuple(chains),
        cumulative_names = Tuple(chain.cumulative_name for chain in chains),
    )
end

function _observation_species_metadata(name)
    leaves = _flatten_symbol_leaves(name)
    parsed = map(_parse_observation_stage, leaves)
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

function _parse_observation_stage(name::Symbol)
    parts = split(String(name), "_")
    length(parts) >= 3 || return nothing
    startswith(parts[1], "O") || return nothing

    stage = tryparse(Int, parts[end])
    isnothing(stage) && return nothing

    source = Symbol(join(parts[2:(end - 1)], "_"))
    isempty(String(source)) && return nothing

    return (source = source, stage = stage)
end
