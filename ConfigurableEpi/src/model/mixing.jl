# Contact between groups from caller-owned tables: between locations from radiation flows, and
# between age bands from contact matrices. The package ships no data.

# The rows after the header of a comma-separated table, each split into stripped fields.
function _read_table(path::AbstractString, header::AbstractString, what::AbstractString)
    isfile(path) || error("$what not found at `$path`")
    lines = filter(!isempty ∘ strip, readlines(path))
    length(lines) > 1 || error("$what `$path` is empty")
    strip(first(lines)) == header || error("$what `$path` has header `$(first(lines))`, expected `$header`")
    n_fields = count(==(','), header) + 1
    return map(@view lines[2:end]) do line
        fields = strip.(split(line, ','))
        length(fields) == n_fields || error("$what `$path`: expected $n_fields fields in `$line`")
        fields
    end
end

"""
    load_radiation_matrix(locations; path) -> Matrix{Float64}

Row-normalised contact matrix for `locations` (in order) from an `origin,destination,flow` table:
`M[x, y]` is the share of location `x`'s out-of-location contact that is with `y`, re-normalised
over the selected locations. Errors when a location is absent, has a self-flow, or retains no
flow to the others.
"""
function load_radiation_matrix(locations; path::AbstractString)
    wanted = [lowercase(String(l)) for l in locations]
    isempty(wanted) && throw(ArgumentError("need at least one location"))
    allunique(wanted) || throw(ArgumentError("locations must be unique, got $wanted"))
    index = Dict(k => i for (i, k) in enumerate(wanted))
    flows = zeros(length(wanted), length(wanted))
    seen = Set{Int}()
    for (origin, destination, flow) in _read_table(path, "origin,destination,flow", "radiation flows")
        x = get(index, lowercase(origin), 0)
        x > 0 || continue
        push!(seen, x)
        y = get(index, lowercase(destination), 0)
        y > 0 || continue
        value = parse(Float64, flow)
        isfinite(value) && value >= 0 ||
            error("radiation flows `$path`: the flow from $origin to $destination must be finite and non-negative, got $value")
        flows[x, y] = value
    end
    absent = [wanted[x] for x in eachindex(wanted) if x ∉ seen]
    isempty(absent) || throw(ArgumentError("locations absent from $(basename(path)): $(sort!(absent))"))
    for x in eachindex(wanted)
        iszero(flows[x, x]) ||
            throw(ArgumentError("$(basename(path)) has a non-zero self-flow for $(wanted[x]); expected off-diagonal only"))
        total = sum(view(flows, x, :))
        total > 0 || throw(
            ArgumentError("location $(wanted[x]) has no retained flow to any of $wanted and would be isolated")
        )
        flows[x, :] ./= total
    end
    return flows
end

"""
    contact_matrix(M, a_com) -> Matrix{Float64}

`(1 - a_com) I + a_com M`: within-location contact at `a_com = 0`, the radiation matrix at 1.
"""
contact_matrix(M::AbstractMatrix, a_com::Real) = (1 - a_com) .* Matrix(1.0I, size(M)...) .+ a_com .* M

"""
    AgeBand(name, lower_months, upper_months)

One age class, `[lower_months, upper_months)` in months of age, so the first year of life can be
banded finely. The oldest band of a model is open-ended (`upper_months = Inf`).
"""
struct AgeBand
    name::Symbol
    lower_months::Float64
    upper_months::Float64
    function AgeBand(name::Symbol, lower_months::Real, upper_months::Real)
        0 <= lower_months < upper_months ||
            throw(ArgumentError("age band $name needs 0 <= lower < upper, got [$lower_months, $upper_months)"))
        isfinite(lower_months) || throw(ArgumentError("age band $name has a non-finite lower bound"))
        return new(name, Float64(lower_months), Float64(upper_months))
    end
end
AgeBand(name::AbstractString, lower_months::Real, upper_months::Real) = AgeBand(Symbol(name), lower_months, upper_months)

"""
    validate_age_bands(bands) -> bands

Check that `bands` partition `[0, Inf)` in order: the first starts at 0, each starts where the
last ended, the last is open-ended, and the names are distinct.
"""
function validate_age_bands(bands)
    isempty(bands) && throw(ArgumentError("need at least one age band"))
    allunique(b.name for b in bands) ||
        throw(ArgumentError("age band names must be distinct, got $([b.name for b in bands])"))
    first(bands).lower_months == 0 ||
        throw(ArgumentError("the first age band must start at 0 months, got $(first(bands).lower_months)"))
    for (previous, next) in zip(bands, Iterators.drop(bands, 1))
        previous.upper_months == next.lower_months || throw(
            ArgumentError(
                "age bands must be contiguous: $(previous.name) ends at $(previous.upper_months) " *
                    "months and $(next.name) starts at $(next.lower_months)",
            )
        )
    end
    isinf(last(bands).upper_months) ||
        throw(ArgumentError("the last age band must be open-ended (upper_months = Inf)"))
    return bands
end

"""
    AgeMixing(bands, populations, contacts)

A population on age `bands` ([`AgeBand`](@ref)s partitioning `[0, Inf)`): `populations[g]`,
absolute or as shares, and `contacts`, a NamedTuple of matrices by setting such as
`(; school, other)`, where `contacts.school[g, h]` is the mean daily number of school contacts
one person in band `g` has with people in band `h`. The settings are disjoint parts of contact;
a single matrix is the one setting `all`.
"""
struct AgeMixing{C <: NamedTuple}
    bands::Vector{AgeBand}
    populations::Vector{Float64}
    contacts::C
    function AgeMixing(bands::AbstractVector{AgeBand}, populations::AbstractVector{<:Real}, contacts::NamedTuple)
        validate_age_bands(bands)
        n = length(bands)
        length(populations) == n || throw(DimensionMismatch("$(length(populations)) populations for $n age bands"))
        all(p -> isfinite(p) && p > 0, populations) || throw(ArgumentError("populations must be positive"))
        isempty(contacts) && throw(ArgumentError("need at least one contact setting"))
        matrices = map(contacts) do m
            size(m) == (n, n) || throw(DimensionMismatch("contact matrices must be $n × $n for $n age bands"))
            all(x -> isfinite(x) && x >= 0, m) || throw(ArgumentError("contact rates must be finite and non-negative"))
            Matrix{Float64}(m)
        end
        return new{typeof(matrices)}(collect(bands), Float64.(populations), matrices)
    end
end
AgeMixing(bands, populations, contacts::AbstractMatrix) = AgeMixing(bands, populations, (; all = contacts))

"Spectral radius of the contacts over all settings together."
contact_spectral_radius(mixing::AgeMixing) = maximum(abs, eigen(sum(values(mixing.contacts))).values)

# The share of base band `base` that falls inside model band `band`, with ages spread evenly within
# a finite base band. The open top base band holds no age density to divide by.
function _overlap_fraction(band::AgeBand, base::AgeBand)
    lo, hi = max(band.lower_months, base.lower_months), min(band.upper_months, base.upper_months)
    hi > lo || return 0.0
    isinf(base.upper_months) || return (hi - lo) / (base.upper_months - base.lower_months)
    band.lower_months <= base.lower_months && isinf(band.upper_months) || throw(
        ArgumentError(
            "age band $(band.name) would divide the open-ended base band $(base.name) " *
                "(from $(base.lower_months) months); the oldest model band must start at or before it",
        )
    )
    return 1.0
end

function _base_contacts(path::AbstractString, key::AbstractString, index::Dict{Symbol, Int})
    contacts = fill(NaN, length(index), length(index))
    for (place, contactor, contactee, rate) in
        _read_table(path, "location,contactor,contactee,contacts_per_day", "age contact matrix")
        lowercase(place) == key || continue
        g, h = get(index, Symbol(contactor), 0), get(index, Symbol(contactee), 0)
        g > 0 && h > 0 || error(
            "$(basename(path)) names an age band absent from the distribution table for `$key`: $contactor, $contactee",
        )
        contacts[g, h] = parse(Float64, rate)
    end
    all(c -> isfinite(c) && c >= 0, contacts) || throw(
        ArgumentError(
            "$(basename(path)) does not give a finite, non-negative contact rate for every pair of base age bands of `$key`",
        )
    )
    return contacts
end

_contact_paths(path::AbstractString) = (; all = path)
_contact_paths(paths::NamedTuple) = paths

"""
    load_age_mixing(location, bands; distribution_path, contact_path, population = 1.0) -> AgeMixing

The [`AgeMixing`](@ref) of `location` on the model's `bands`, regrouped from two tables on a
common base grid: `location,age_band,lower_months,upper_months,share` and
`location,contactor,contactee,contacts_per_day`. `contact_path` is one table, the setting `all`,
or a NamedTuple of tables by setting. The populations are the shares times `population`.

A model band is any union of base bands or any part of a finite one. Within a finite base band
ages are taken as evenly spread and every member is given the band's per-capita contact rates,
so a divided band's contacts with a part are in proportion to the part. The regrouping keeps the
total contacts between two bands symmetric whenever the base tables are.

Errors when the location is absent, a base band is missing, or the bands do not partition
`[0, Inf)` on the base grid.
"""
function load_age_mixing(
        location, bands; distribution_path::AbstractString, contact_path, population::Real = 1.0,
    )
    validate_age_bands(bands)
    key = lowercase(String(location))
    base, base_share = AgeBand[], Float64[]
    for (place, name, lower, upper, share) in
        _read_table(distribution_path, "location,age_band,lower_months,upper_months,share", "age distribution")
        lowercase(place) == key || continue
        push!(base, AgeBand(name, parse(Float64, lower), parse(Float64, upper)))
        push!(base_share, parse(Float64, share))
    end
    isempty(base) && throw(ArgumentError("location `$key` is absent from $(basename(distribution_path))"))
    allunique(b.name for b in base) ||
        error("$(basename(distribution_path)) repeats an age band for `$key`: $([b.name for b in base])")
    all(s -> isfinite(s) && s > 0, base_share) ||
        error("$(basename(distribution_path)) has a non-positive share for `$key`")
    base_share ./= sum(base_share)

    # fraction[g, b]: the share of base band `b` inside model band `g`.
    fraction = [_overlap_fraction(band, b) for band in bands, b in base]
    for (b, column) in zip(base, eachcol(fraction))
        isapprox(sum(column), 1.0; atol = 1.0e-12) ||
            throw(ArgumentError("the age bands do not cover base band $(b.name) of `$key` exactly once"))
    end
    drawn = fraction .* base_share'
    shares = vec(sum(drawn; dims = 2))
    all(>(0), shares) || throw(ArgumentError("an age band holds no population for `$key`: $shares"))
    # A member of band g drawn from base b makes `base_contacts[b, c]` contacts with base c, of
    # which the part inside band h is `fraction[h, c]`.
    index = Dict(b.name => i for (i, b) in enumerate(base))
    contacts = map(path -> (drawn * _base_contacts(path, key, index) * fraction') ./ shares, _contact_paths(contact_path))
    return AgeMixing(bands, population .* shares, contacts)
end

"""
    load_setting_contacts(location, bands; matrix_path, population_path, settings = (:school, :other)) -> AgeMixing

The [`AgeMixing`](@ref) of `location` read on the tables' own age groups, named by `bands`: the
contact layers `settings` from `location,layer,row_age,col_age,value` and the populations from
`location,age_group,population`. Every age pair must be present in every layer.
"""
function load_setting_contacts(
        location, bands; matrix_path::AbstractString, population_path::AbstractString,
        settings::Tuple{Vararg{Symbol}} = (:school, :other),
    )
    validate_age_bands(bands)
    key = lowercase(String(location))
    position = Dict(String(b.name) => g for (g, b) in enumerate(bands))
    n = length(bands)
    layers = NamedTuple{settings}(ntuple(_ -> fill(NaN, n, n), length(settings)))
    for (place, layer, row, col, value) in
        _read_table(matrix_path, "location,layer,row_age,col_age,value", "setting contact matrices")
        lowercase(place) == key || continue
        Symbol(layer) in settings || error("`$matrix_path` has the layer `$layer`, not one of $settings")
        g, h = get(position, row, 0), get(position, col, 0)
        g > 0 && h > 0 || continue
        layers[Symbol(layer)][g, h] = parse(Float64, value)
    end
    for (layer, m) in pairs(layers)
        any(isnan, m) && error("`$matrix_path` lacks $(count(isnan, m)) `$layer` entries for `$key` over $([b.name for b in bands])")
    end
    populations = fill(NaN, n)
    for (place, age, value) in _read_table(population_path, "location,age_group,population", "age populations")
        lowercase(place) == key && haskey(position, age) || continue
        populations[position[age]] = parse(Float64, value)
    end
    any(isnan, populations) && error("`$population_path` lacks populations for `$key` over $([b.name for b in bands])")
    return AgeMixing(bands, populations, layers)
end
