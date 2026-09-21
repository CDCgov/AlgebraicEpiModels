# Typing for stratification, in the context meaning a set of static labels for stratum
# carried by population individuals

"""
Represents a stratification structure for epidemiological models.
"""
abstract type Stratification <: EpiMechModel end

"""
Contact-based stratification for epidemiological models.

Represents stratifications where populations are divided into strata with
contact-based interactions (full contact matrix between all pairs). This
generalizes age groups, geographic regions modelled as cross-region contact (rather than explicit movement),
risk groups, etc.

Multiple contact stratifications can be stacked via `compose_stratifications` to
create product strata (e.g., age × geography).

# Fields
- `stratum_names::Vector{Symbol}`: Names of strata (e.g., [:child, :adult] or [:urban, :rural])
- `label::Symbol`: Semantic label for the stratification (e.g., :age, :geography, :risk)

Reflexive boxes (disease, reversion, and waning) are controlled via the
`include_reflexives` keyword on `create_model`, not on the struct itself.

# Examples
```julia
# Age stratification
age = ContactStratification([:child, :adult], :age)

# Geographic stratification
geo = ContactStratification([:urban, :rural], :geography)

# Compose to create age × geography strata
typing = OnePopulationTyping()
sir = create_model(typing, SIR())
age_sir = typed_product(sir, create_model(typing, age))
age_geo_sir = typed_product(age_sir, create_model(typing, geo))
# Result: 4 strata (childxurban, childxrural, adultxurban, adultxrural)
```

See also: [`AgeStratification`](@ref), [`GeographicStratification`](@ref)
"""
struct ContactStratification <: Stratification
    stratum_names::Vector{Symbol}
    label::Symbol

    function ContactStratification(
            stratum_names::Vector{Symbol}, label::Symbol
        )
        if isempty(stratum_names)
            error("ContactStratification requires at least one stratum")
        end
        if length(stratum_names) != length(unique(stratum_names))
            error("Stratum names must be unique")
        end
        return new(stratum_names, label)
    end
end

"""
Compose two `ContactStratification`s via Cartesian product.
# Arguments
- `a::ContactStratification`: First contact stratification
- `b::ContactStratification`: Second contact stratification
# Returns
- `ContactStratification`: New contact stratification representing the product of `a` and
    `b`, with combined stratum names and a composite label.
# Examples
```julia
age = ContactStratification([:child, :adult], :age)
geo = ContactStratification([:urban, :rural], :geography)
age_geo = compose_stratifications(age, geo)
# Resulting stratum names: [:childxurban, :childxrural, :adultxurban, :adultxrural]
# Resulting label: :age_x_geography
```
"""
function compose_stratifications(a::ContactStratification, b::ContactStratification)
    combined_names = [
        Symbol(string(sa), "x", string(sb)) for sa in a.stratum_names
            for sb in b.stratum_names
    ]
    combined_label = Symbol(string(a.label), "_x_", string(b.label))
    return ContactStratification(combined_names, combined_label)
end

*(a::ContactStratification, b::ContactStratification) = compose_stratifications(a, b)

#-------------------------------------
# Convenience constructors
#-------------------------------------

"""
    AgeStratification(age_group_names::Vector{Symbol})
    AgeStratification(names...)

Convenience constructor for age-based contact stratification.

Creates a `ContactStratification` with label `:age`. Provided for backward
compatibility and semantic clarity.

# Examples
```julia
age = AgeStratification([:child, :adult, :elderly])
age = AgeStratification(:child, :adult, :elderly)  # splatting syntax
```
"""
AgeStratification(names::Vector{Symbol}) = ContactStratification(names, :age)
AgeStratification(names...) = ContactStratification(collect(Symbol.(names)), :age)

"""
    GeographicStratification(location_names::Vector{Symbol})
    GeographicStratification(names...)

Convenience constructor for geography-based contact stratification.

Creates a `ContactStratification` with label `:geography`. Represents
geographic regions with cross-region contact (not movement-based models).

# Examples
```julia
geo = GeographicStratification([:urban, :suburban, :rural])
geo = GeographicStratification(:urban, :suburban, :rural)  # splatting syntax
```
"""
GeographicStratification(names::Vector{Symbol}) = ContactStratification(names, :geography)
function GeographicStratification(names...)
    return ContactStratification(collect(Symbol.(names)), :geography)
end
