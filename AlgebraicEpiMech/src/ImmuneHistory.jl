# Immune-history stratification (issue #279)
#
# A stratification *factor* over `UninfectedInfectedTyping` that composes with a
# generic disease model (SEIRS, multi-stage, +observation) via `typed_product`.
# Sits on the cross-immunity spectrum next to `NoCrossImmunity` (independent) and
# `CompleteCrossImmunity` (one shared immune pool) as the history-resolved option.
#
# The uninfected pool is split into immune-history classes `U_h`; the infected side
# carries `(h, i)` — the prior history `h` and the strain `i` currently being fought
# — so recovery knows what to increment. Unlike `ContactStratification`, whose strata
# are inert (reflexive on every transition type), immune history changes the stratum
# on ONE transition: `:reversion` runs OFF-diagonal, `(h,i) → h∪{i}`. Infection is
# escape-selective (a class immune to `h` only carries `:transmission` for `i ∉ h`),
# and `:disease`/`:observation` stay reflexive so the base model's progression and
# observation chains run inside each `(h,i)`.
#
# Index sets (per the S-vs-E/I/R rule): after `typed_product` the base's `S` is
# indexed by `{h}` (→ `S_h`) and `E`/`I`/`R` by `{(h,i)}` (→ `E_{h,i}`, …).

"""
Selects how much immune history the uninfected group retains. A singleton stored on
`ImmuneHistory`; construction dispatches on it, so a new mode is additive.
"""
abstract type ImmuneHistoryMode end

"""
Full immune history: uninfected individuals are indexed by the **set** of strains
they are immune to (`2^n` classes). Immunity accumulates — reversion routes to
`h ∪ {i}`.
"""
struct FullHistory <: ImmuneHistoryMode end

"""
Latest-infection status: uninfected individuals are indexed by their **most recent**
infection (`n + 1` classes). Reversion overwrites immune status — routes to `{i}`
(Gog–Grenfell "status-based").
"""
struct LatestInfection <: ImmuneHistoryMode end

"""
    ImmuneHistory{M<:ImmuneHistoryMode} <: MultiStrainModel

Immune-history stratification factor over `UninfectedInfectedTyping`. Compose it with
a disease model via `typed_product` — the disease model provides `S→E→I→R`, this
factor adds the immune-status structure (escape-selective infection + the
history-incrementing reversion).

# Fields
- `strain_names::Vector{Symbol}`: strain names (e.g. `[:current, :invader]`)
- `mode::M`: `FullHistory()` or `LatestInfection()`

# Examples
```julia
typing = UninfectedInfectedTyping()

seirs   = create_model(typing, SEIRS())
history = create_model(typing, ImmuneHistory([:current, :invader]))

model = typed_product(seirs, history)   # immune-history-resolved SEIRS
# States: S_naive, S_current, …, E_invader_from_current, I_…, R_…
```
"""
struct ImmuneHistory{M <: ImmuneHistoryMode} <: MultiStrainModel
    strain_names::Vector{Symbol}
    mode::M

    function ImmuneHistory(
            strain_names::Vector{Symbol}; mode::M = FullHistory()
        ) where {M <: ImmuneHistoryMode}
        isempty(strain_names) &&
            throw(ArgumentError("ImmuneHistory requires at least one strain"))
        length(strain_names) == length(unique(strain_names)) ||
            throw(ArgumentError("strain names must be unique"))
        return new{M}(strain_names, mode)
    end
end

"""
    ImmuneHistory(n::Int; kwargs...)

Auto-name `n` strains as `strain_1 … strain_n`.
"""
function ImmuneHistory(n::Int; kwargs...)
    n >= 1 || throw(ArgumentError("number of strains must be >= 1"))
    return ImmuneHistory([Symbol("strain_$(i)") for i in 1:n]; kwargs...)
end

# ============================================================================
# Immune-class + wiring helpers (internal; not exported)
#
# An immune class is a sorted `Tuple` of strains immune to; `()` is naive.
# ============================================================================

# Immune class: a sorted tuple of strain symbols (`()` is naive).
const _ImmuneClass = Tuple{Vararg{Symbol}}

function _powerset(v::AbstractVector{T}) where {T}
    subsets = Vector{Vector{T}}([T[]])
    for x in v
        # `append!` grows in place; the comprehension reads the current `subsets`
        # fully before appending, so we avoid recopying the accumulated powerset.
        append!(subsets, [vcat(s, x) for s in subsets])
    end
    return subsets
end

_immune_classes(::FullHistory, strains::Vector{Symbol}) =
    [Tuple(sort(s; by = string)) for s in _powerset(strains)]

function _immune_classes(::LatestInfection, strains::Vector{Symbol})
    classes = Vector{Tuple{Vararg{Symbol}}}()
    push!(classes, ())
    for s in strains
        push!(classes, (s,))
    end
    return classes
end

# Strains a class immune to `class` is susceptible to.
_susceptible_strains(class, strains::Vector{Symbol}) =
    [i for i in strains if !(i in class)]

# Immune class reached after recovering from strain `i` (the :reversion target).
_recover_to(::FullHistory, class, i::Symbol) = Tuple(sort(unique((class..., i)); by = string))
_recover_to(::LatestInfection, ::Any, i::Symbol) = (i,)

_htag(class) = isempty(class) ? "naive" : join(string.(class), "_")

_uninfected_name(class) = Symbol("U_", _htag(class))

# Infected junction for "history h, currently infected with strain i": `<i>_from_<h>`.
_infected_name(class, i::Symbol) = Symbol(string(i), "_from_", _htag(class))

# The infected `(h, i)` junctions to create: every (class, susceptible strain) pair.
function _infected_pairs(mode::ImmuneHistoryMode, strains::Vector{Symbol})
    pairs = Tuple{_ImmuneClass, Symbol}[]
    for h in _immune_classes(mode, strains), i in _susceptible_strains(h, strains)
        push!(pairs, (h, i))
    end
    return pairs
end

# Identify the strain fought by an infected junction `<i>_from_<h>` (match longest
# strain name first so prefix names are safe).
function _strain_of(var::Symbol, strain_names)
    s = string(var)
    for cand in sort(strain_names; by = x -> -length(string(x)))
        cs = string(cand)
        startswith(s, cs * "_from_") && return cand
    end
    return error("could not identify strain in $var among $strain_names")
end
