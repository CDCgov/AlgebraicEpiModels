# Mass-action kinetics analysed once into sparse operators: classify each transition, record it
# by kind, factor the catalysed weights, then assemble the terms `vectorfield_factored` evaluates.

"""The key index of the fixed rates, whose values are folded into the coefficients."""
const _FIXED = 0

"""
    Reaction

One transition under mass action: flux `coefficient * p[key] * Π x[j]^m` over `inputs`, added
to the state with `stoichiometry`. Both lists are `(species, count)` pairs in species order.
"""
struct Reaction
    key::Int
    coefficient::Float64
    inputs::Vector{Tuple{Int, Int}}
    stoichiometry::Vector{Tuple{Int, Int}}
end

"""Multiplications and additions per call when `r` is evaluated as it is."""
_operations(r::Reaction) = length(r.inputs) + 1 + 2 * length(r.stoichiometry)

"""
    ZeroOrderTerms

Column `b` of `matrix` (species × operator) is added at rate `p[key[b]]`.
"""
struct ZeroOrderTerms
    key::Vector{Int}
    matrix::SparseMatrixCSC{Float64, Int}
end

"""
    FirstOrderTerms

Operator `b`, at rate `p[key[b]]`, owns the columns `columns[b]` of `matrix` (species ×
column); column `c` is scaled by `x[input[c]]`.
"""
struct FirstOrderTerms
    key::Vector{Int}
    columns::Vector{UnitRange{Int}}
    input::Vector{Int}
    matrix::SparseMatrixCSC{Float64, Int}
end

"""
    CatalysedTerms

Catalysed second-order transitions, grouped into classes: a class is a substrate with one net
stoichiometry (a column of the species × class `net_effect`), whatever the catalyst. The class
fluxes are `x[substrate] .* (L z)` with `z = K(p) Π x`:

- `right`: the rows of `Π`, as the columns of a species × factor matrix.
- `left`: `L`, a class × factor matrix.
- `kernel`, `kernel_right`, `kernel_constant`, `kernel_terms`: the sparse rows of `K`. Row `ℓ`
  owns the entries `kernel[ℓ]`; entry `q` multiplies column `kernel_right[q]` of `right` by
  `kernel_constant[q] + Σ scale · p[key]` over column `q` of the key × entry `kernel_terms`.
"""
struct CatalysedTerms
    substrate::Vector{Int}
    net_effect::SparseMatrixCSC{Int, Int}
    right::SparseMatrixCSC{Float64, Int}
    left::SparseMatrixCSC{Float64, Int}
    kernel::Vector{UnitRange{Int}}
    kernel_right::Vector{Int}
    kernel_constant::Vector{Float64}
    kernel_terms::SparseMatrixCSC{Float64, Int}
end

"""
    KineticReport

Counts from [`kinetic_structure`](@ref): transitions by kind, the catalysed `classes` and
`catalysts_per_class`, the factor and kernel sizes, `keys_not_rank_one` and
`max_row_directions`, the multiplications and additions per call of the factored and flat
forms, and `max_weight_error`, the largest relative difference between a declared weight and
its factored value.
"""
Base.@kwdef struct KineticReport
    species::Int
    transitions::Int
    rate_keys::Int
    fixed_keys::Int
    zero_order::Int
    first_order::Int
    catalysed::Int
    residual::Int
    inactive::Int
    classes::Int
    catalysts_per_class::Tuple{Int, Int}
    left_factors::Int
    right_factors::Int
    kernel_entries::Int
    keys_not_rank_one::Int
    max_row_directions::Int
    factored_operations::Int
    flat_operations::Int
    max_weight_error::Float64
end

"""
The mass-action kinetics of a Petri net in factored form, as built by [`kinetic_structure`](@ref)
and evaluated by [`vectorfield_factored`](@ref).

With `x` the state and `p` the rates by key, the field is

    dx = b + A x + Σ_ℓ z_ℓ · (L_ℓ x) + residual,      z = K(p) Π x

# Fields
- `species`: flattened species names, in net order.
- `rate_keys`: the keys `p` must supply, in order of first appearance.
- `fixed_keys`: the keys whose rate was folded into the constants.
- `zero_order`: `b`, as `ZeroOrderTerms`.
- `first_order`: `A x`, as `FirstOrderTerms`.
- `catalysed`: the factored second-order part, as `CatalysedTerms`.
- `residual`: transitions kept as they are (uncatalysed second order, and third order and
  above), as `Reaction`s.
- `report`: counts from the analysis, as a `KineticReport`.

Each of these types is documented under its own name, for example `?AlgebraicEpiMech.Reaction`.

A key index of `0` in any of these stands for the fixed rates.
"""
struct KineticStructure
    species::Vector{Symbol}
    rate_keys::Vector{Symbol}
    fixed_keys::Vector{Symbol}
    zero_order::ZeroOrderTerms
    first_order::FirstOrderTerms
    catalysed::CatalysedTerms
    residual::Vector{Reaction}
    report::KineticReport
end

function Base.show(io::IO, ::MIME"text/plain", ks::KineticStructure)
    r = ks.report
    println(io, "KineticStructure: $(r.species) species, $(r.transitions) transitions")
    println(io, "  rates: $(r.rate_keys) keyed, $(r.fixed_keys) fixed")
    println(
        io, "  transitions: $(r.zero_order) zero order, $(r.first_order) first order, " *
            "$(r.catalysed) catalysed, $(r.residual) residual, $(r.inactive) inactive"
    )
    println(
        io, "  catalysed: $(r.classes) classes with $(r.catalysts_per_class[1]) to " *
            "$(r.catalysts_per_class[2]) catalysts each"
    )
    println(
        io, "  factors: $(r.left_factors) left, $(r.right_factors) right, " *
            "$(r.kernel_entries) kernel entries; $(r.keys_not_rank_one) keys not rank one " *
            "(most row directions: $(r.max_row_directions))"
    )
    return print(
        io, "  operations per call: $(r.factored_operations) factored, $(r.flat_operations) flat"
    )
end

# ============================================================================
# Classify: one reaction and one kind per transition
# ============================================================================

"""The default rate class: every transition is its own key, with weight one."""
_unit_rate_class(name) = (name, 1.0)

"""A `rate_class` result as a flattened key and a finite weight."""
_rate_class(class::Pair) = _rate_class((class.first, class.second))
function _rate_class(class::Tuple{Any, Real})
    key, weight = class
    isfinite(weight) || throw(ArgumentError("rate class `$key` has a non-finite weight $weight"))
    return flatten_symbols(key), Float64(weight)
end
_rate_class(class) = throw(
    ArgumentError("`rate_class` must return `(key, weight)` or `key => weight`; got $(repr(class))"),
)

"""Each transition's key index and coefficient, and the rate keys in order of first appearance."""
function _resolve_keys(pn, rate_class, fixed_values)
    rate_keys = Symbol[]
    key_index = Dict{Symbol, Int}()
    fixed_seen = Set{Symbol}()
    resolved = map(1:nt(pn)) do i
        key, weight = _rate_class(rate_class(tname(pn, i)))
        if haskey(fixed_values, key)
            push!(fixed_seen, key)
            return (_FIXED, weight * fixed_values[key])
        end
        k = get!(key_index, key) do
            push!(rate_keys, key)
            length(rate_keys)
        end
        return (k, weight)
    end
    unknown = setdiff(keys(fixed_values), fixed_seen)
    isempty(unknown) || throw(
        ArgumentError("fixed rate(s) $(sort!(collect(unknown))) are not rate keys of the Petri net"),
    )
    return rate_keys, resolved
end

"""`(index, multiplicity)` of each distinct index, in index order."""
function _counted(indices)
    counts = Dict{Int, Int}()
    for j in indices
        counts[j] = get(counts, j, 0) + 1
    end
    return sort!([(j, m) for (j, m) in counts]; by = first)
end

"""Transition `i` as a `Reaction`, and the inputs it also outputs."""
function _reaction(pn, i, key, coefficient)
    ins = _counted(inputs(pn, i))
    outs = Dict(_counted(outputs(pn, i)))
    net = mergewith!(+, Dict{Int, Int}(j => -m for (j, m) in ins), outs)
    stoichiometry = sort!([(j, c) for (j, c) in net if c != 0]; by = first)
    return Reaction(key, coefficient, ins, stoichiometry), [j for (j, _) in ins if haskey(outs, j)]
end

# How a transition enters the factored field.
abstract type _Kind end
struct _Inactive <: _Kind end
struct _ZeroOrder <: _Kind end
struct _FirstOrder <: _Kind end
struct _Residual <: _Kind end
"""A second-order transition with an input it also outputs: a candidate catalyst."""
struct _Catalysed <: _Kind
    conserved::Vector{Int}
end

"""The kind of reaction `r`, given the inputs it also outputs."""
function _kind(r::Reaction, conserved)
    (r.coefficient == 0 || isempty(r.stoichiometry)) && return _Inactive()
    order = sum(last, r.inputs; init = 0)
    order == 0 && return _ZeroOrder()
    order == 1 && return _FirstOrder()
    order == 2 && !isempty(conserved) && return _Catalysed(conserved)
    return _Residual()
end

# ============================================================================
# Record: coefficients gathered by key
# ============================================================================

"""Coefficients gathered by key, before they are packed into sparse operators."""
struct _Builder
    source::Dict{Int, Dict{Int, Float64}}               # key => species => coefficient
    linear::Dict{Int, Dict{Int, Dict{Int, Float64}}}    # key => input => species => coefficient
    bilinear::Dict{Int, Dict{Int, Dict{Int, Float64}}}  # key => class => catalyst => weight
    class_index::Dict{Tuple{Int, Vector{Tuple{Int, Int}}}, Int}
    substrate::Vector{Int}
    net_effect::Vector{Vector{Tuple{Int, Int}}}
    residual::Vector{Reaction}
end
_Builder() = _Builder((T() for T in fieldtypes(_Builder))...)

"""The entry of `d` at `k`, created empty if missing."""
_entry!(d::AbstractDict, k) = get!(valtype(d), d, k)

"""Add `scale * c` into `into[j]` for every `(j, c)` of `entries`."""
function _add_scaled!(into, entries, scale)
    for (j, c) in entries
        into[j] = get(into, j, 0.0) + scale * c
    end
    return into
end

_record!(::_Builder, ::_Inactive, r, _) = nothing
_record!(b::_Builder, ::_Residual, r, _) = push!(b.residual, r)
_record!(b::_Builder, ::_ZeroOrder, r, _) = _add_scaled!(_entry!(b.source, r.key), r.stoichiometry, r.coefficient)
_record!(b::_Builder, ::_FirstOrder, r, _) =
    _add_scaled!(_entry!(_entry!(b.linear, r.key), first(only(r.inputs))), r.stoichiometry, r.coefficient)

"""
Record a catalysed flux `x[substrate] * x[catalyst]` in its class. When both inputs are
conserved, the catalyst is the one that `catalyses` more of the net.
"""
function _record!(b::_Builder, kind::_Catalysed, r, catalyses)
    catalyst = argmax(j -> (catalyses[j], j), kind.conserved)
    s = length(r.inputs) == 1 ? catalyst : only(j for (j, _) in r.inputs if j != catalyst)
    class = get!(b.class_index, (s, r.stoichiometry)) do
        push!(b.substrate, s)
        push!(b.net_effect, r.stoichiometry)
        length(b.substrate)
    end
    return _add_scaled!(_entry!(_entry!(b.bilinear, r.key), class), ((catalyst, 1),), r.coefficient)
end

# ============================================================================
# Factor: low-rank catalysed weights
# ============================================================================

"""The nonzero entries of `d`, in index order."""
_sorted_entries(d::AbstractDict{Int, Float64}) = sort!([(j, w) for (j, w) in d if w != 0]; by = first)

_key_label(rate_keys, k) = k == _FIXED ? "the fixed rates" : "rate key `$(rate_keys[k])`"

"""
A sparse vector reduced to its direction: its lead entry, the entries divided by it, and a
signature equal for vectors proportional to within `sigdigits`.
"""
function _direction(entries::Vector{Tuple{Int, Float64}}, sigdigits, label)
    lead = entries[1][2]
    direction = [(j, w / lead) for (j, w) in entries]
    all(((_, w),) -> isfinite(w) && abs(w) >= floatmin(Float64), direction) || throw(
        ArgumentError(
            "cannot factor the weights of $label: a ratio between two of them is outside the " *
                "normal Float64 range (about 1e-308 to 1e308)",
        ),
    )
    signature = [(j, round(w; sigdigits)) for (j, w) in direction]
    return lead, direction, signature
end

"""The index of `direction` among `factors` by `signature`, appended if new."""
function _factor_index!(factors, index, direction, signature)
    return get!(index, signature) do
        push!(factors, direction)
        length(factors)
    end
end

"""
Factors of the catalysed weights: per key, rows proportional to one another share a right
factor and the classes they reach share a left factor; both are merged across keys.
"""
struct _Factors
    right::Vector{Vector{Tuple{Int, Float64}}}
    left::Vector{Vector{Tuple{Int, Float64}}}
    kernel_constant::Dict{Tuple{Int, Int}, Float64}                    # (left, right) => scale
    kernel_terms::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}   # (left, right) => (key, scale)
    row_directions::Vector{Int}
    max_weight_error::Float64
end

function _factor(bilinear, rate_keys, sigdigits)
    right, left = Vector{Tuple{Int, Float64}}[], Vector{Tuple{Int, Float64}}[]
    right_index = Dict{Vector{Tuple{Int, Float64}}, Int}()
    left_index = Dict{Vector{Tuple{Int, Float64}}, Int}()
    kernel_constant = Dict{Tuple{Int, Int}, Float64}()
    kernel_terms = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()
    row_directions = Int[]
    max_weight_error = 0.0
    for k in sort!(collect(keys(bilinear)))
        rows, label = bilinear[k], _key_label(rate_keys, k)
        by_direction = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for class in sort!(collect(keys(rows)))
            entries = _sorted_entries(rows[class])
            isempty(entries) && continue
            lead, direction, signature = _direction(entries, sigdigits, label)
            r = _factor_index!(right, right_index, direction, signature)
            push!(_entry!(by_direction, r), (class, lead))
        end
        push!(row_directions, length(by_direction))
        rebuilt = Dict{Tuple{Int, Int}, Float64}()
        for r in sort!(collect(keys(by_direction)))
            scale, direction, signature = _direction(by_direction[r], sigdigits, label)
            l = _factor_index!(left, left_index, direction, signature)
            if k == _FIXED
                kernel_constant[(l, r)] = get(kernel_constant, (l, r), 0.0) + scale
            else
                push!(_entry!(kernel_terms, (l, r)), (k, scale))
            end
            for (class, a) in left[l], (c, w) in right[r]
                rebuilt[(class, c)] = get(rebuilt, (class, c), 0.0) + scale * a * w
            end
        end
        for (class, row) in rows, (c, w) in row
            w == 0 && continue
            max_weight_error = max(max_weight_error, abs(get(rebuilt, (class, c), 0.0) - w) / abs(w))
        end
    end
    max_weight_error <= 10.0^(2 - sigdigits) ||
        error("factored weights differ from the declared ones by $max_weight_error")
    return _Factors(right, left, kernel_constant, kernel_terms, row_directions, max_weight_error)
end

# ============================================================================
# Assemble: sparse terms and the report
# ============================================================================

"""A sparse `m × length(columns)` matrix with column `j` holding the `(row, value)` pairs `columns[j]`."""
function _sparse_columns(m::Int, columns::Vector{Vector{Tuple{Int, T}}}) where {T}
    I = Int[i for column in columns for (i, _) in column]
    J = Int[j for (j, column) in enumerate(columns) for _ in column]
    V = T[v for column in columns for (_, v) in column]
    return sparse(I, J, V, m, length(columns))
end

function _zero_order_terms(S, source)
    key = sort!(collect(keys(source)))
    return ZeroOrderTerms(key, _sparse_columns(S, [_sorted_entries(source[k]) for k in key]))
end

function _first_order_terms(S, linear)
    key = sort!(collect(keys(linear)))
    columns, input, entries = UnitRange{Int}[], Int[], Vector{Tuple{Int, Float64}}[]
    for k in key
        start = length(input) + 1
        for s in sort!(collect(keys(linear[k])))
            column = _sorted_entries(linear[k][s])
            isempty(column) && continue
            push!(input, s)
            push!(entries, column)
        end
        push!(columns, start:length(input))
    end
    return FirstOrderTerms(key, columns, input, _sparse_columns(S, entries))
end

function _catalysed_terms(S, n_keys, b::_Builder, f::_Factors)
    pairs = sort!(collect(union(keys(f.kernel_constant), keys(f.kernel_terms))))
    kernel_left = first.(pairs)
    terms = [get(f.kernel_terms, pair, Tuple{Int, Float64}[]) for pair in pairs]
    return CatalysedTerms(
        b.substrate, _sparse_columns(S, b.net_effect), _sparse_columns(S, f.right),
        _sparse_columns(length(b.substrate), f.left),
        [searchsorted(kernel_left, l) for l in eachindex(f.left)], last.(pairs),
        Float64[get(f.kernel_constant, pair, 0.0) for pair in pairs], _sparse_columns(n_keys, terms),
    )
end

"""Multiplications and additions per call of the factored form."""
function _operations(z::ZeroOrderTerms, a::FirstOrderTerms, c::CatalysedTerms, residual)
    kernel = sum(2 * length(nzrange(c.right, r)) + 2 for r in c.kernel_right; init = 0)
    classes = sum(2 + 2 * length(nzrange(c.net_effect, class)) for class in rowvals(c.left); init = 0)
    return 2 * nnz(z.matrix) + length(a.input) + 2 * nnz(a.matrix) + kernel +
        2 * nnz(c.kernel_terms) + classes + sum(_operations, residual; init = 0)
end

function _report(pn, rate_keys, fixed_values, terms, b::_Builder, f::_Factors, kinds, reactions)
    tally(K) = count(k -> k isa K, kinds)
    catalysts = [Set{Int}() for _ in b.substrate]
    for rows in values(b.bilinear), (class, row) in rows
        union!(catalysts[class], keys(row))
    end
    return KineticReport(;
        species = ns(pn), transitions = nt(pn),
        rate_keys = length(rate_keys), fixed_keys = length(fixed_values),
        zero_order = tally(_ZeroOrder), first_order = tally(_FirstOrder),
        catalysed = tally(_Catalysed), residual = tally(_Residual), inactive = tally(_Inactive),
        classes = length(b.substrate),
        catalysts_per_class = isempty(catalysts) ? (0, 0) : extrema(length, catalysts),
        left_factors = length(f.left), right_factors = length(f.right),
        kernel_entries = length(terms[3].kernel_right),
        keys_not_rank_one = count(>(1), f.row_directions),
        max_row_directions = maximum(f.row_directions; init = 0),
        factored_operations = _operations(terms...),
        flat_operations = sum(_operations, reactions; init = 0),
        max_weight_error = f.max_weight_error,
    )
end

"""
    kinetic_structure(pn; rate_class, fixed = (;), sigdigits = 12) -> KineticStructure

Analyse the mass-action kinetics of `pn` once, from its arcs and its rate classes, into the
factored form [`vectorfield_factored`](@ref) evaluates.

# Arguments
- `rate_class`: a function from a transition's name (as `tname` gives it, so a tuple for a
  typed product) to `(key, weight)` or `key => weight`. The transition's rate is
  `weight * p[key]`, and `key` is flattened with [`flatten_symbols`](@ref). By default every
  transition is its own class with weight one and the keys are the flattened transition names,
  which is the contract of [`vectorfield_flat`](@ref).
- `fixed`: rates by key (a `NamedTuple` or a dictionary) that never change. They are folded
  into the constants and `p` must not supply them.
- `sigdigits`: weight vectors that agree to this many significant digits once scaled share a
  factor.

# Analysis
1. Zero-order and first-order transitions become a constant vector and sparse linear
   operators, one per key, with the fixed keys summed into a single constant operator.
2. A second-order transition with an input that is also an output is catalysed. Such
   transitions are grouped into classes by substrate and net stoichiometry, so a class's flux is
   `x[substrate] * Σ_c B[class, c] x[c]`.
3. Each key's part of `B` is split by row direction: rows proportional to one another share a
   right factor, so a rank-one part costs one inner product and anything else keeps its rows.
4. Right factors, and the left factors that carry them back to classes, are merged across keys.
5. Every other transition is kept in a flat loop.

The `report` field holds the counts; see `KineticReport`.
"""
function kinetic_structure(
        pn::AbstractPetriNet; rate_class = _unit_rate_class, fixed = (;), sigdigits::Integer = 12,
    )
    S = ns(pn)
    species = [flatten_symbols(sname(pn, j)) for j in 1:S]
    _require_unique_flattened_names(species, "species")
    rate_class === _unit_rate_class &&
        _require_unique_flattened_names([flatten_symbols(tname(pn, i)) for i in 1:nt(pn)], "transition")
    fixed_values = Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in pairs(fixed))
    rate_keys, resolved = _resolve_keys(pn, rate_class, fixed_values)

    analysed = [_reaction(pn, i, resolved[i]...) for i in 1:nt(pn)]
    reactions = first.(analysed)
    kinds = [_kind(r, conserved) for (r, conserved) in analysed]
    catalyses = zeros(Int, S)
    for kind in kinds
        kind isa _Catalysed && (catalyses[kind.conserved] .+= 1)
    end
    builder = _Builder()
    foreach((kind, r) -> _record!(builder, kind, r, catalyses), kinds, reactions)
    factors = _factor(builder.bilinear, rate_keys, sigdigits)

    terms = (
        _zero_order_terms(S, builder.source), _first_order_terms(S, builder.linear),
        _catalysed_terms(S, length(rate_keys), builder, factors), builder.residual,
    )
    report = _report(pn, rate_keys, fixed_values, terms, builder, factors, kinds, reactions)
    return KineticStructure(species, rate_keys, sort!(collect(keys(fixed_values))), terms..., report)
end

# ============================================================================
# Evaluate
# ============================================================================

"""
    vectorfield_factored(structure::KineticStructure) -> vectorfield!(du, u, p, t)
    vectorfield_factored(pn; kwargs...)

The mass-action vector field of a Petri net from its [`kinetic_structure`](@ref). `u` and `du`
are indexed by flattened species name as for [`vectorfield_flat`](@ref), and `p` holds one rate
per key of `structure.rate_keys`, by name. A rate may be a function of `(u, t)`.

The two fields agree to rounding, not bit for bit, because the factored form sums in a
different order. With the default rate classes `p` is keyed by transition name and the call is
interchangeable with [`vectorfield_flat`](@ref).
"""
function vectorfield_factored(structure::KineticStructure)
    S = length(structure.species)
    u_positions = _NamePositions()
    du_positions = _NamePositions()
    p_positions = _NamePositions()
    return function vectorfield!(du, u, p, t)
        x = _values_by_name(u_positions, structure.species, u)
        rates = _rates_by_key(p_positions, structure, p)
        du_m = _slots_by_name(du_positions, structure.species, du)
        Ty = valtype(du)
        A = promote_type(Float64, Ty)
        acc = _zeroed_accumulator(A, Ty, du_m, S)
        _accumulate!(acc, A, structure.zero_order, x, rates, u, t)
        _accumulate!(acc, A, structure.first_order, x, rates, u, t)
        _accumulate!(acc, A, structure.catalysed, x, rates, u, t)
        _accumulate!(acc, A, structure.residual, x, rates, u, t)
        _writeback_accumulator!(acc, du_m, S)
        return du
    end
end

vectorfield_factored(pn::AbstractPetriNet; kwargs...) =
    vectorfield_factored(kinetic_structure(pn; kwargs...))

"""Throw if `p` supplies a fixed rate, which would be ignored."""
function _require_no_fixed_keys(structure::KineticStructure, names)
    supplied = intersect(structure.fixed_keys, names)
    isempty(supplied) || throw(
        ArgumentError(
            "rate(s) $supplied are fixed in the kinetic structure, so the values in `p` " *
                "would be ignored",
        ),
    )
    return nothing
end

"""`p` in key order, checked for fixed keys once per argument type (every call for a dictionary)."""
function _rates_by_key(cache::_NamePositions, structure::KineticStructure, p)
    entry = @atomic :acquire cache.entry
    if entry === nothing || entry[1] !== typeof(p)
        _require_no_fixed_keys(structure, propertynames(p))
    end
    return _values_by_name(cache, structure.rate_keys, p)
end
function _rates_by_key(cache::_NamePositions, structure::KineticStructure, p::AbstractDict)
    _require_no_fixed_keys(structure, keys(p))
    return _values_by_name(cache, structure.rate_keys, p)
end

"""The rate of key `k` as an `A`: one for the fixed rates, already folded into the coefficients."""
Base.@propagate_inbounds _rate(::Type{A}, rates, k::Int, u, t) where {A} =
    k == _FIXED ? one(A) : convert(A, AlgebraicPetri.valueat(rates[k], u, t))

@inline function _accumulate!(acc, ::Type{A}, z::ZeroOrderTerms, x, rates, u, t) where {A}
    @inbounds for b in eachindex(z.key)
        rate = _rate(A, rates, z.key[b], u, t)
        for e in nzrange(z.matrix, b)
            acc[rowvals(z.matrix)[e]] += rate * nonzeros(z.matrix)[e]
        end
    end
    return acc
end

@inline function _accumulate!(acc, ::Type{A}, a::FirstOrderTerms, x, rates, u, t) where {A}
    @inbounds for b in eachindex(a.key)
        rate = _rate(A, rates, a.key[b], u, t)
        for col in a.columns[b]
            flow = rate * x[a.input[col]]
            for e in nzrange(a.matrix, col)
                acc[rowvals(a.matrix)[e]] += nonzeros(a.matrix)[e] * flow
            end
        end
    end
    return acc
end

@inline function _accumulate!(acc, ::Type{A}, c::CatalysedTerms, x, rates, u, t) where {A}
    @inbounds for l in eachindex(c.kernel)
        z = zero(A)
        for q in c.kernel[l]
            coefficient = convert(A, c.kernel_constant[q])
            for e in nzrange(c.kernel_terms, q)
                coefficient += nonzeros(c.kernel_terms)[e] * _rate(A, rates, rowvals(c.kernel_terms)[e], u, t)
            end
            pool = zero(A)
            for e in nzrange(c.right, c.kernel_right[q])
                pool += nonzeros(c.right)[e] * x[rowvals(c.right)[e]]
            end
            z += coefficient * pool
        end
        for e in nzrange(c.left, l)
            class = rowvals(c.left)[e]
            flux = nonzeros(c.left)[e] * z * x[c.substrate[class]]
            for s in nzrange(c.net_effect, class)
                acc[rowvals(c.net_effect)[s]] += flux * nonzeros(c.net_effect)[s]
            end
        end
    end
    return acc
end

@inline function _accumulate!(acc, ::Type{A}, residual::Vector{Reaction}, x, rates, u, t) where {A}
    @inbounds for r in residual
        flux = r.coefficient * _rate(A, rates, r.key, u, t)
        for (j, m) in r.inputs
            flux *= x[j]^m
        end
        for (j, c) in r.stoichiometry
            acc[j] += flux * c
        end
    end
    return acc
end
