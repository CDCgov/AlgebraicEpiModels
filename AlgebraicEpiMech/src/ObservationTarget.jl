# Attaching observation to a composed model, by pushout.
#
# The typed petri net models compose via their limit / categorical pullback operation, that is
# identically typed transitions are generated from the set of all tensor products of the component transitions.
#
# Observation models also introduce structure, via a different operation: a colimit / categorical pushout operation.
# These join the component models along shared structure, in this case either a species or transitions that on one hand are part of
# the petri net representing the epidemiological dynamics and on the other hand represent generating new observations
# in the future.
#
# The main types of observation are "incidence"-like and "prevalence"-like.
# Both kinds of observation are the same colimit. Only the rule "L -> R" differs:
#
#   incidence   L =  S + I -> E + I           R =  S + I -> E + I + O
#               the SAME event, additionally recorded. One transition, one rate, so "one
#               observation per infection" is structural rather than a convention.
#
#   prevalence  L =  I                        R =  I -> I + O
#               a new catalytic mechanism reading occupancy, with its own detection rate.
#               Rapid antigen testing is the motivating case.
#
#
# Because this runs after the pullback compositions, stratification factors need to know nothing whatsoever
# about observation: no per-stratum accumulator, no reflexive observation box etc.
#
# Both rules are monotone additions, nothing deleted and nothing merged beyond gluing onto a chain
# findable by label, so the colimit is materialized directly in one pass rather than by a
# pushout per matched transition.

"""
    ObservationTarget

What an observation attaches to. [`AtEvent`](@ref) records a transition (incidence);
[`AtCompartment`](@ref) samples a species (prevalence).
"""
abstract type ObservationTarget end

"""
    AtEvent(transition::Symbol; group = identity)
    AtEvent(transitions::Tuple; source = first(transitions), group = identity)

Record a TRANSITION: incidence. Every transition whose (flattened) name begins with one of
`transitions` gains an output arc into an accumulator, so it fires at that transition's own
rate and counts exactly one observation per occurrence. Chains are named after `source`.

`AtEvent(:transmission)` counts infections as they happen, which is what decouples a reporting
delay from the latent period — the delay is then whatever the chain adds and nothing else.
`AtEvent((:transmission, :importation); source = :infection)` counts local and imported
infections together.

`group` maps the label of the species an event nets to the label its chain is keyed on, so
several products can share one chain. Dropping a stratum from the label, such as the immune
class, merges those chains; this is exact because every chain is linear with the same delay
rates.
"""
struct AtEvent{G} <: ObservationTarget
    transitions::Tuple{Vararg{Symbol}}
    source::Symbol
    group::G

    function AtEvent(
            transitions::Tuple{Vararg{Symbol}}; source::Symbol = first(transitions), group::G = identity
        ) where {G}
        allunique(transitions) || throw(ArgumentError("duplicate prefix in $transitions"))
        return new{G}(transitions, source, group)
    end
end
AtEvent(transition::Symbol; group = identity) = AtEvent((transition,); group)
AtEvent(::Tuple{}; kwargs...) = throw(ArgumentError("AtEvent needs at least one transition prefix"))

"""
    AtCompartment(species)

Sample a SPECIES: prevalence. Every species whose (flattened) name begins with `species` gains a
catalytic tap `X -> X + O`, carrying its own detection rate — whoever is in the compartment is
currently detectable (although detection does not cause removal)
"""
struct AtCompartment <: ObservationTarget
    species::Symbol
end

# Labels in a composed net are nested tuples: `(:S, :a)` for one stratification, `((:S, :a), :b)`
# for two. An accumulator belongs to the same stratum as the thing it observes, so it takes that
# label with the COMPARTMENT part swapped out and the stratification parts kept.
_relabel_head(label::Symbol, head::Symbol) = head
_relabel_head(label::Tuple, head::Symbol) = (_relabel_head(label[1], head), Base.tail(label)...)

_flat(x) = flatten_symbols(x)
_startswith(label, prefix::Symbol) = startswith(string(_flat(label)), string(prefix))

# The species a transition NETS — outputs that are not also inputs. For `S + I -> E + I` that is
# `E`, so grouping infection routes by it gives one accumulator per infectee stratum, which is
# what "incidence in location x" means. Structural, so it needs no name parsing. A death nets
# nothing, so it is grouped by the species it leaves.
function _net_product(pn, t)
    ins, outs = AlgebraicPetri.inputs(pn, t), AlgebraicPetri.outputs(pn, t)
    isempty(outs) && return first(ins)
    net = setdiff(outs, ins)
    return isempty(net) ? first(outs) : first(net)
end

"""
    attach_observation(pn, target; n_stages = 1, prefix = :O) -> LabelledPetriNet

Attach an observation chain to `pn`, returning a new net. The result is the colimit described at
the top of this file, materialized in one direct pass (see the note there for why it is not
built by pushout).

`n_stages` Erlang delay stages are added per chain: the first receives the observation, each
subsequent one takes flow from the last, and the final stage is cumulative (no outflow), i.e. the
accumulator the observation model reads.

Reapplying the same target, prefix, and stage count is idempotent. Reusing a target and prefix
for an incompatible chain shape throws an `ArgumentError`; choose a different prefix when both
observation layouts are required.

Chains are grouped by stratum. For [`AtEvent`](@ref) the group is the transition's net product —
for infection routes `S_x + I_y -> E_x + I_y` that is `E_x`, so all routes infecting `x` count
into `x`'s chain. A transition with no output (a death) is grouped by the species it leaves,
and the target's `group` can merge these groups further. For [`AtCompartment`](@ref) each
matched species gets its own chain.

Apply this after all `typed_product` composition so chains are grouped over the final strata.
"""
function attach_observation(pn, target::ObservationTarget; n_stages::Int = 1, prefix::Symbol = :O)
    n_stages >= 1 || throw(ArgumentError("n_stages must be >= 1"))
    net = deepcopy(pn)
    _attach_all!(net, target, n_stages, prefix)
    return net
end

# One pass over the matches, appending directly to `net`. Matches are taken
# against the ORIGINAL parts only: everything appended carries the `prefix` head, which the name
# filters cannot match, and freezing the bound makes that explicit.
function _attach_all!(net, target::AtEvent, n_stages, prefix)
    chain_heads = Dict{Any, Int}()   # chain head label -> species index of stage 1
    for t in 1:nt(net)
        name = tname(net, t)
        any(p -> _startswith(name, p), target.transitions) || continue
        key = target.group(sname(net, _net_product(net, t)))
        chain, _ = _chain_labels(key, n_stages, prefix, target.source)
        head = get!(chain_heads, chain[1]) do
            # A chain may pre-exist from an earlier `attach_observation` call: validate and glue.
            first(_ensure_chain!(net, key, n_stages, prefix, target.source))
        end
        # An identical re-application already has this output arc.
        any(==(head), AlgebraicPetri.outputs(net, t)) ||
            add_outputs!(net, 1, [t], [head])               # the recorded event
    end
    return net
end

function _attach_all!(net, target::AtCompartment, n_stages, prefix)
    for s in 1:ns(net)
        _startswith(sname(net, s), target.species) || continue
        src = sname(net, s)
        source = _head_symbol(src)
        chain, _ = _chain_labels(src, n_stages, prefix, source)
        tap = _relabel_head(src, Symbol("obs_inflow_", _flat(src)))
        head, built = _ensure_chain!(net, src, n_stages, prefix, source)
        taps = [t for t in 1:nt(net) if tname(net, t) == tap]
        if built
            isempty(taps) || _incompatible_chain(chain[1], n_stages)
            add_transitions!(net, 1; tname = [tap])
            t = nt(net)
            add_inputs!(net, 1, [t], [s])
            add_outputs!(net, 1, [t], [s])                  # catalytic: source returned
            add_outputs!(net, 1, [t], [head])               # into the chain
        elseif length(taps) != 1 || !_is_observation_tap(net, only(taps), s, head)
            _incompatible_chain(chain[1], n_stages)
        end
    end
    return net
end

_stage_head(prefix::Symbol, source::Symbol, stage::Int) = Symbol(prefix, "_", source, "_", stage)

# Chain labels for the group keyed by label `key`: one per stage, plus the delay transitions.
# `source` is part of each label, so every stage of a multi-stage compartment keys its own chain.
function _chain_labels(key, n_stages, prefix, source)
    species = [_relabel_head(key, _stage_head(prefix, source, i)) for i in 1:n_stages]
    delays = [
        _relabel_head(
                key,
                Symbol(
                    prefix, "_", source, "_", i, "_to_", prefix, "_", source, "_", i + 1
                ),
            ) for i in 1:(n_stages - 1)
    ]
    return species, delays
end

function _ensure_chain!(net, key, n_stages, prefix, source)
    chain, delays = _chain_labels(key, n_stages, prefix, source)
    extended_chain, extended_delays = _chain_labels(key, n_stages + 1, prefix, source)

    species_matches = [
        [s for s in 1:ns(net) if sname(net, s) == label] for label in chain
    ]
    delay_matches = [
        [t for t in 1:nt(net) if tname(net, t) == label] for label in delays
    ]
    next_species = [s for s in 1:ns(net) if sname(net, s) == extended_chain[end]]
    next_delay = [t for t in 1:nt(net) if tname(net, t) == extended_delays[end]]

    has_existing_parts =
        any(matches -> !isempty(matches), species_matches) ||
        any(matches -> !isempty(matches), delay_matches) ||
        !isempty(next_species) || !isempty(next_delay)

    if !has_existing_parts
        first_new = ns(net) + 1
        _build_chain!(net, chain, delays, first_new)
        return first_new, true
    end

    valid =
        all(matches -> length(matches) == 1, species_matches) &&
        all(matches -> length(matches) == 1, delay_matches) &&
        isempty(next_species) && isempty(next_delay)

    if valid
        stages = only.(species_matches)
        for (i, matches) in enumerate(delay_matches)
            transition = only(matches)
            valid &= AlgebraicPetri.inputs(net, transition) == [stages[i]]
            valid &= AlgebraicPetri.outputs(net, transition) == [stages[i + 1]]
        end
    end

    valid || _incompatible_chain(chain[1], n_stages)
    return only(first(species_matches)), false
end

function _incompatible_chain(head, n_stages)
    throw(
        ArgumentError(
            "observation chain $head already exists with structure incompatible with " *
                "n_stages=$n_stages; reuse its original stage count or choose a different prefix",
        ),
    )
end

function _is_observation_tap(net, transition, source, head)
    inputs = AlgebraicPetri.inputs(net, transition)
    outputs = AlgebraicPetri.outputs(net, transition)
    return inputs == [source] && length(outputs) == 2 &&
        count(==(source), outputs) == 1 && count(==(head), outputs) == 1
end

# The compartment part of a (possibly nested) label: `(:I1, :a)` -> `:I1`, `:I1` -> `:I1`.
_head_symbol(label::Symbol) = label
_head_symbol(label::Tuple) = _head_symbol(label[1])

# Append a chain's species and delay transitions to `P`, the species starting at index
# `first_new`. Returns nothing; mutates `P`.
function _build_chain!(P, chain, delays, first_new)
    add_species!(P, length(chain); sname = chain)
    if !isempty(delays)
        add_transitions!(P, length(delays); tname = delays)
    end
    for (i, _) in enumerate(delays)
        # Delay transitions are numbered after whatever transitions the rule already carries.
        t = nt(P) - length(delays) + i
        add_inputs!(P, 1, [t], [first_new + i - 1])
        add_outputs!(P, 1, [t], [first_new + i])
    end
    return nothing
end
