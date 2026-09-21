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
# Optimisation
# An earlier version of this file materialized the observations literally using Catlab functions one pushout per matched transition,
# re-matching against the grown net after every step. That was O(L^4) in the number of strata —
# ~L^2 matched routes, each paying a `homomorphism` search plus a colimit on a net that is itself
# ~L^2 transitions. For example, on a 50-state geographic model it meant hours of model construction
# before any model running (1.2 s at L = 6, 109 s at L = 20, extrapolated 1-2 h at L = 50; 0.01 s
# now). Both rules are monotone additions — nothing is deleted, nothing is merged beyond gluing
# onto a chain that is findable by label — so the same colimit is materialized directly below in
# one pass, appending chains and arcs in first-encounter order.

"""
    ObservationTarget

What an observation attaches to. [`AtEvent`](@ref) records a transition (incidence);
[`AtCompartment`](@ref) samples a species (prevalence).
"""
abstract type ObservationTarget end

"""
    AtEvent(transition)

Record a TRANSITION: incidence. Every transition whose (flattened) name begins with `transition`
gains an output arc into an accumulator, so it fires at that transition's own rate and counts
exactly one observation per occurrence.

`AtEvent(:transmission)` counts infections as they happen, which is what decouples a reporting
delay from the latent period — the delay is then whatever the chain adds and nothing else.
"""
struct AtEvent <: ObservationTarget
    transition::Symbol
end

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
# what "incidence in location x" means. Structural, so it needs no name parsing.
function _net_product(pn, t)
    ins = Set(AlgebraicPetri.inputs(pn, t))
    outs = [s for s in AlgebraicPetri.outputs(pn, t) if !(s in ins)]
    return isempty(outs) ? first(AlgebraicPetri.outputs(pn, t)) : first(outs)
end

"""
    attach_observation(pn, target; n_stages = 1, prefix = :O) -> LabelledPetriNet

Attach an observation chain to `pn`, returning a new net. The result is the colimit described at
the top of this file, materialized in one direct pass (see the note there for why it is not
built by pushout).

`n_stages` Erlang delay stages are added per chain: the first receives the observation, each
subsequent one takes flow from the last, and the final stage is cumulative (no outflow), i.e. the
accumulator the observation model reads.

Chains are grouped by stratum. For [`AtEvent`](@ref) the group is the transition's net product —
for infection routes `S_x + I_y -> E_x + I_y` that is `E_x`, so all routes infecting `x` count
into `x`'s chain. For [`AtCompartment`](@ref) each matched species gets its own chain.

Apply this AFTER all composition; see `docs/concepts/composition-and-observation.md`.
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
    chain_head = Dict{Any, Int}()   # chain head label -> species index of stage 1
    for t in 1:nt(net)
        _startswith(tname(net, t), target.transition) || continue
        chain, delays = _chain_labels(
            net, _net_product(net, t), n_stages, prefix, target.transition
        )
        # Already recorded into this chain (re-application)? Then this transition is done.
        any(s -> sname(net, s) == chain[1], AlgebraicPetri.outputs(net, t)) && continue
        head = get!(chain_head, chain[1]) do
            # A chain may pre-exist from an earlier `attach_observation` call: glue, don't rebuild.
            existing = findfirst(s -> sname(net, s) == chain[1], 1:ns(net))
            if existing === nothing
                first_new = ns(net) + 1
                _build_chain!(net, chain, delays, first_new)
                first_new
            else
                existing
            end
        end
        add_outputs!(net, 1, [t], [head])                   # the recorded event
    end
    return net
end

function _attach_all!(net, target::AtCompartment, n_stages, prefix)
    for s in 1:ns(net)
        _startswith(sname(net, s), target.species) || continue
        chain, delays = _chain_labels(
            net, s, n_stages, prefix, _head_symbol(sname(net, s))
        )
        _has_species(net, chain[1]) && continue             # already tapped
        src = sname(net, s)
        tap = _relabel_head(src, Symbol("obs_inflow_", _flat(src)))
        add_transitions!(net, 1; tname = [tap])
        t = nt(net)
        add_inputs!(net, 1, [t], [s])
        add_outputs!(net, 1, [t], [s])                      # catalytic: source returned
        first_new = ns(net) + 1
        _build_chain!(net, chain, delays, first_new)
        add_outputs!(net, 1, [t], [first_new])              # into the chain
    end
    return net
end

_stage_head(prefix::Symbol, source::Symbol, stage::Int) = Symbol(prefix, "_", source, "_", stage)

# Chain labels for the group keyed by species `key_idx`: one per stage, plus the delay transitions.
#
# `source` is part of the name, not decoration. Dropping it breaks two contracts at once:
# every `I` stage of a multi-stage compartment would key to the SAME chain label and all but the
# first would be skipped as already-tapped; and `_parse_observation_stage` (observation_chain.jl)
# requires a `<prefix>_<source>_<stage>` leaf, so `observation_layout` — and hence
# `ConfigurableEpi`'s `StateLayout` — would not recognise the accumulators as observation states
# at all and would treat them as core model state.
function _chain_labels(net, key_idx, n_stages, prefix, source)
    key = sname(net, key_idx)
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

_has_species(net, label) = any(s -> sname(net, s) == label, 1:ns(net))

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
