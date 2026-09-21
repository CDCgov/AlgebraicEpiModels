"""
Recursively flatten a tuple of symbols and join them with underscores.

# Examples
```julia
flatten_symbols((:x, :y, :z)) # returns :x_y_z
flatten_symbols(((:x, :y), :z)) # returns :x_y_z
flatten_symbols((((:a, :b), :c), :d)) # returns :a_b_c_d
```
"""
function flatten_symbols(name::Tuple)
    # Recursively flatten all elements
    flattened = _flatten_recursive(name)
    # Join all symbols with underscores
    symbol_strings = [string(sym) for sym in flattened]
    return Symbol(join(symbol_strings, "_"))
end

function flatten_symbols(name::Symbol)
    return name
end

# Handle recursive flattening of tuples
function _flatten_recursive(name::Tuple)
    return mapreduce(vcat, name) do element
        _flatten_recursive(element)
    end
end

# Handle single symbol case
_flatten_recursive(name::Symbol) = name

"""
Generate an ODE vectorfield function from a Petri net using mass action kinetics as per
`AlgebraicPetri.vectorfield`. The only difference is that species and transition names
are flattened using `flatten_symbols`, converting nested tuples into single symbols
joined by underscores.

# Arguments
- `pn::AbstractPetriNet`: A Petri net representing the reaction network

# Returns
A function `(du, u, p, t) -> du` where:
- `du`: Array/Dict to store computed derivatives
- `u`: Current state (species concentrations), indexed by flattened species names
- `p`: Parameters (rate constants), indexed by flattened transition names
- `t`: Current time

This function complies with SciML conventions for in-place ODE vectorfields.

# Performance

Everything that can be resolved from the net alone — flattened names, the mass-action input
lists, the nonzero stoichiometry — is resolved ONCE at build time, and the right-hand side is a
sparse loop over arcs rather than a dense loop over `transitions × species`. Name lookup into the
arguments is also done once per concrete argument type: `u`, `du` and `p` may be anything whose
`propertynames` lists the flattened names in the same order as its integer indexing (`LVector`,
`NamedTuple`, ...), and that position map is cached on first sight of the type; an `AbstractDict`
is looked up by name on every call. At the 50-state geographic model (2700 transitions × 300
species) this is the difference between ~13 ms and ~40 µs per evaluation. The results are bit-for-bit
identical to the dense resolve-everything-per-call form this replaced — same factor and
summation order — which `test/test_vectorfield_flat.jl` keeps as a local reference and
compares against exactly.
"""
function vectorfield_flat(pn::AbstractPetriNet)
    S = ns(pn)
    T = nt(pn)
    species_syms = [flatten_symbols(sname(pn, j)) for j in 1:S]
    transition_syms = [flatten_symbols(tname(pn, i)) for i in 1:T]

    tm = TransitionMatrices(pn)
    # Mass-action inputs as `(species, multiplicity)`, ascending species: the factor is `u_j^k`
    # applied in species order — the reference's `prod(u_j^input[i, j] for j)` minus its exact
    # `u^0 = 1` factors — so the products agree bit for bit for any multiplicity.
    inputs = [Tuple{Int, Int}[] for _ in 1:T]
    # Net stoichiometry `output - input`, nonzero entries only, ascending species (a zero entry
    # contributes an exact `+0.0` in the dense form).
    stoichiometry = [Tuple{Int, Int}[] for _ in 1:T]
    for i in 1:T, j in 1:S
        k = tm.input[i, j]
        k == 0 || push!(inputs[i], (j, k))
        c = tm.output[i, j] - k
        c == 0 || push!(stoichiometry[i], (j, c))
    end

    u_positions = _NamePositions()
    du_positions = _NamePositions()
    p_positions = _NamePositions()

    return function vectorfield!(du, u, p, t)
        u_m = _values_by_name(u_positions, species_syms, u)
        p_m = _values_by_name(p_positions, transition_syms, p)
        du_m = _slots_by_name(du_positions, species_syms, du)
        # Same numeric path as the dense form: each rate converted to `valtype(du)`, then the
        # contributions summed from a Float64 zero (`sum(...; init = 0.0)`) and written once. For
        # Float64 and Dual-over-Float64 outputs that accumulator IS `du`'s element type, so the
        # sums go straight into `du` with no per-call allocation; a narrower output type (Float32)
        # keeps the wider accumulator and converts on the final write. `du` must not alias `u`.
        Ty = valtype(du)
        acc = _accumulator(promote_type(Float64, Ty), Ty, du_m, S)
        @inbounds for k in 1:S
            acc[k] = zero(eltype(acc))
        end
        @inbounds for i in 1:T
            factor = one(eltype(u_m))
            for (j, k) in inputs[i]
                factor *= u_m[j]^k
            end
            rate = convert(Ty, AlgebraicPetri.valueat(p_m[i], u, t) * factor)
            for (j, c) in stoichiometry[i]
                acc[j] += rate * c
            end
        end
        _finish!(acc, du_m, S)
        return du
    end
end

# Position of each flattened name inside an argument, resolved once per concrete argument type.
# Single-entry and lock-free: the right-hand side is called from threaded ensemble propagation, so
# readers take an acquire load and a miss publishes an immutable `(type, positions)` pair with a
# release store — two threads racing on the same miss compute identical content.
mutable struct _NamePositions
    @atomic entry::Union{Nothing, Tuple{DataType, Vector{Int}}}
end
_NamePositions() = _NamePositions(nothing)

function _positions(cache::_NamePositions, syms::Vector{Symbol}, x)
    Tx = typeof(x)
    entry = @atomic :acquire cache.entry
    if entry !== nothing && entry[1] === Tx
        return entry[2]
    end
    names = collect(propertynames(x))
    positions = Vector{Int}(undef, length(syms))
    for (k, s) in enumerate(syms)
        i = findfirst(==(s), names)
        i === nothing && throw(
            ArgumentError(
                "`$s` is not a name of the $(Tx) argument; the vectorfield needs every " *
                    "flattened species and transition name of the net to be present",
            ),
        )
        positions[k] = i
    end
    @atomic :release cache.entry = (Tx, positions)
    return positions
end

# An argument read in net order without copying it: `v[k]` is the value of the k-th net name.
# Materializing the gather would cost one allocation of `length(syms)` per right-hand-side call
# — 2700 rates per call on the geographic model, inside threaded ensemble propagation.
struct _ByName{X}
    x::X
    positions::Vector{Int}
end
Base.@propagate_inbounds Base.getindex(v::_ByName, k::Int) = v.x[v.positions[k]]
Base.@propagate_inbounds Base.setindex!(v::_ByName, val, k::Int) = (v.x[v.positions[k]] = val)
Base.eltype(::Type{_ByName{X}}) where {X} = eltype(X)
Base.length(v::_ByName) = length(v.positions)

# The same for a dictionary, which indexes only by key.
struct _ByKey{X}
    x::X
    syms::Vector{Symbol}
end
Base.getindex(v::_ByKey, k::Int) = v.x[v.syms[k]]
Base.setindex!(v::_ByKey, val, k::Int) = (v.x[v.syms[k]] = val)
Base.eltype(::Type{_ByKey{X}}) where {X} = valtype(X)
Base.length(v::_ByKey) = length(v.syms)

# Fast path: integer indexing through the cached positions (LVector, NamedTuple, ...).
_values_by_name(cache::_NamePositions, syms::Vector{Symbol}, x) =
    _ByName(x, _positions(cache, syms, x))
# By-name path for dictionaries, which index only by key.
_values_by_name(::_NamePositions, syms::Vector{Symbol}, x::AbstractDict) = [x[s] for s in syms]

# Writable slots of `du` in net order, for the accumulation.
_slots_by_name(cache::_NamePositions, syms::Vector{Symbol}, du) =
    _ByName(du, _positions(cache, syms, du))
_slots_by_name(::_NamePositions, syms::Vector{Symbol}, du::AbstractDict) = _ByKey(du, syms)

# The accumulator is `du` itself when its element type already is the dense form's
# accumulation type, and a temporary of that type otherwise (both cases resolved by dispatch).
_accumulator(::Type{A}, ::Type{A}, du_m, S) where {A} = du_m
_accumulator(::Type{A}, ::Type{Ty}, du_m, S) where {A, Ty} = zeros(A, S)
_finish!(acc::_ByName, du_m::_ByName, S) = nothing
_finish!(acc::_ByKey, du_m::_ByKey, S) = nothing
_finish!(acc, du_m, S) = (
    @inbounds for k in 1:S
        du_m[k] = acc[k]
    end
)
