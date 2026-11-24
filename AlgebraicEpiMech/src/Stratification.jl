# Contact strata, with an optional movement graph between them.

"""
Represents a stratification structure for epidemiological models.
"""
abstract type Stratification <: EpiMechModel end

"""
    ContactStratification(stratum_names, label; transitions = Pair{Symbol, Symbol}[], births = (), deaths = ())

Strata with a full contact matrix between every pair (age groups, geographic regions, risk
groups), and an optional directed graph of movement between them (ageing, migration).
Movement keeps the disease state and is independent of contact-based infection.

Multiple contact stratifications can be stacked via `compose_stratifications` to
create product strata (e.g., age × geography).

# Fields
- `stratum_names::Vector{Symbol}`: Names of strata (e.g., [:child, :adult] or [:urban, :rural])
- `label::Symbol`: Semantic label for the stratification (e.g., :age, :geography, :risk)
- `transitions::Vector{Pair{Symbol, Symbol}}`: Directed movement edges between distinct
  declared strata. Each edge becomes one movement box per population type, named
  `source_to_target`. With no edges the stratification is static and a product with it has
  no movement.
- `demography::Demography`: the strata that births enter and deaths leave, from the `births`
  and `deaths` keywords (see [`Demography`](@ref)). A factor declaring no births drops them
  from a `typed_product`.

Reflexive boxes for the types a factor does not implement are controlled via the
`include_reflexives` keyword on [`create_model`](@ref), not on the struct itself.

# Examples
```julia
age = ContactStratification([:child, :adult], :age)
aging = ContactStratification([:child, :adult], :age; transitions = [:child => :adult])
geo = ContactStratification([:urban, :rural], :geography)

# Compose to create age × geography strata
typing = OnePopulationTyping()
sir = create_model(typing, SIR())
age_sir = typed_product(sir, create_model(typing, age))
age_geo_sir = typed_product(age_sir, create_model(typing, geo))
# Result: 4 strata (childxurban, childxrural, adultxurban, adultxrural)

# Ageing: the disease model must carry reflexive movement boxes for the pullback to keep it
sir = create_model(typing, SIR(); include_reflexives = :strata)
aging_sir = dom(typed_product(sir, create_model(typing, aging)))
# Transitions include S_child_to_adult, I_child_to_adult, R_child_to_adult

# Births into the youngest group, deaths out of the oldest
demography = ContactStratification(
    [:child, :adult], :age; transitions = [:child => :adult], births = [:child], deaths = [:adult],
)
```

See also: [`AgeStratification`](@ref), [`GeographicStratification`](@ref)
"""
struct ContactStratification <: Stratification
    stratum_names::Vector{Symbol}
    label::Symbol
    transitions::Vector{Pair{Symbol, Symbol}}
    demography::Demography

    function ContactStratification(
            stratum_names::Vector{Symbol}, label::Symbol;
            transitions::AbstractVector{<:Pair{Symbol, Symbol}} = Pair{Symbol, Symbol}[],
            demography...
        )
        if isempty(stratum_names)
            error("ContactStratification requires at least one stratum")
        end
        if length(stratum_names) != length(unique(stratum_names))
            error("Stratum names must be unique")
        end
        for (source, target) in transitions
            source in stratum_names && target in stratum_names ||
                throw(ArgumentError("transition $source => $target must connect declared strata"))
            source == target && throw(ArgumentError("transition $source => $target is a self-loop"))
        end
        allunique(transitions) || throw(ArgumentError("duplicate stratum transitions in $transitions"))
        d = Demography(; demography...)
        d.importation && throw(
            ArgumentError(
                "a contact stratification is passive for importation: list `:importation` in " *
                    "`include_reflexives` instead"
            )
        )
        for stratum in (d.births..., d.deaths...)
            stratum in stratum_names ||
                throw(ArgumentError("demography entry $stratum is not a declared stratum"))
        end
        return new(copy(stratum_names), label, collect(transitions), d)
    end
end

"""
    compose_stratifications(a::ContactStratification, b::ContactStratification)

The Cartesian product of two contact stratifications, with combined stratum names and the
label `a_x_b`. Movement graphs multiply: every edge of one factor pairs with every edge of the
other, and a static factor contributes the identity on its strata, so ageing × static
geography ages within each location. Two static factors give a static product. Births and
deaths multiply the same way, a factor that declares none contributing all of its strata.

# Examples
```julia
age = ContactStratification([:child, :adult], :age; transitions = [:child => :adult])
geo = ContactStratification([:urban, :rural], :geography)
age_geo = compose_stratifications(age, geo)
# Resulting stratum names: [:childxurban, :childxrural, :adultxurban, :adultxrural]
# Resulting label: :age_x_geography
# Resulting transitions: [:childxurban => :adultxurban, :childxrural => :adultxrural]
```
"""
function compose_stratifications(a::ContactStratification, b::ContactStratification)
    combined(sa, sb) = Symbol(sa, "x", sb)
    names = [combined(sa, sb) for sa in a.stratum_names for sb in b.stratum_names]
    label = Symbol(a.label, "_x_", b.label)
    identity_edges(s) = [n => n for n in s.stratum_names]
    transitions = _pair_entries(a, b, s -> s.transitions, identity_edges) do ea, eb
        combined(first(ea), first(eb)) => combined(last(ea), last(eb))
    end
    births = _pair_entries(combined, a, b, s -> s.demography.births, s -> s.stratum_names)
    deaths = _pair_entries(combined, a, b, s -> s.demography.deaths, s -> s.stratum_names)
    return ContactStratification(names, label; transitions, births, deaths)
end

# Every entry of one factor paired with every entry of the other, where a factor with none
# contributes its default; two factors with none give none.
function _pair_entries(combine, a, b, entries, default)
    ea, eb = entries(a), entries(b)
    isempty(ea) && isempty(eb) && return empty(ea)
    return [combine(x, y) for x in (isempty(ea) ? default(a) : ea) for y in (isempty(eb) ? default(b) : eb)]
end

*(a::ContactStratification, b::ContactStratification) = compose_stratifications(a, b)

#-------------------------------------
# Convenience constructors
#-------------------------------------

"""
    AgeStratification(age_group_names::Vector{Symbol}; kwargs...)
    AgeStratification(names...; kwargs...)

A `ContactStratification` with label `:age`; `transitions` declares ageing between age groups,
and `births` and `deaths` the groups newborns enter and deaths leave.

# Examples
```julia
age = AgeStratification([:child, :adult, :elderly])
age = AgeStratification(:child, :adult, :elderly)  # splatting syntax
aging = AgeStratification([:child, :adult]; transitions = [:child => :adult])
```
"""
AgeStratification(names::Vector{Symbol}; kwargs...) = ContactStratification(names, :age; kwargs...)
AgeStratification(names...; kwargs...) = AgeStratification(collect(Symbol.(names)); kwargs...)

"""
    GeographicStratification(location_names::Vector{Symbol}; kwargs...)
    GeographicStratification(names...; kwargs...)

A `ContactStratification` with label `:geography`; `transitions` declares movement between
regions, and `births` and `deaths` the regions with demography.

# Examples
```julia
geo = GeographicStratification([:urban, :suburban, :rural])
geo = GeographicStratification(:urban, :suburban, :rural)  # splatting syntax
```
"""
GeographicStratification(names::Vector{Symbol}; kwargs...) =
    ContactStratification(names, :geography; kwargs...)
GeographicStratification(names...; kwargs...) =
    GeographicStratification(collect(Symbol.(names)); kwargs...)
