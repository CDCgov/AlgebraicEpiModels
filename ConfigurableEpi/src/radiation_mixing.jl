# ============================================================================
# Radiation-model contact mixing
# ============================================================================
#
# Geography in this codebase is CROSS-LOCATION CONTACT, not migration: nobody moves between
# compartments, but susceptibles in `x` can be infected by infectives in `y`. The mixing weights come
# from a CALLER-OWNED radiation-flows file (this package ships no data; the conventional file name
# is `RADIATION_ARTIFACT`): the radiation mobility model for US states parameterised by population,
# with the out-of-state fluxes rescaled.
#
# That rescaling is the point of the file's shape. It divides out the radiation model's one
# unidentified quantity — the TOTAL flux — leaving a pure destination distribution. So the matrix
# answers "given that someone leaves state x, where do they go?" (analytic, fixed by the populations,
# no free parameters) and `A_com` answers "how much do they leave?". The magnitude is the only thing
# left to infer, which is why `A_com` needs an external prior rather than being read off this file;
# see `geographic_seir.default_priors`.
#
# Two properties of that file drive the code below:
#
#   * it is ALREADY row-normalized over all 52 destinations, and
#   * its diagonal is exactly zero — it is a matrix of out-of-origin shares.
#
# So selecting a handful of locations leaves rows summing to far less than 1 (six western states
# retain only the flow between themselves), which is why re-normalization after selection is
# mandatory rather than tidy-up. It is also why `A_com = 1` means no within-location transmission
# at all: `M` has no diagonal to contribute. That is a limit of the algebra, not an operating
# point — `A_com` is inferred on `(0, 1)`.

const RADIATION_ARTIFACT = "radiation_flows.csv2"

"""
    load_radiation_matrix(locations; path) -> Matrix{Float64}

The row-normalized contact matrix `M` for `locations`, in the order given, read from the
caller-owned `origin,destination,flow` table at `path` (required: this package ships no data).

`M[x, y]` is the share of location `x`'s out-of-location contact that is with location `y`.
`M` has a zero diagonal (inherited from the source file) and rows summing to 1 after selection.

Errors rather than guesses when a location is absent from the file, or when a selected row has no
retained flow at all — an all-zero row cannot be normalized, and silently leaving it zero would
make that location epidemiologically isolated without saying so.
"""
function load_radiation_matrix(locations; path::AbstractString)
    keys_wanted = [lowercase(String(l)) for l in locations]
    allunique(keys_wanted) || throw(
        ArgumentError("locations must be unique, got $(keys_wanted)")
    )
    L = length(keys_wanted)
    L >= 1 || throw(ArgumentError("need at least one location"))
    index = Dict(k => i for (i, k) in enumerate(keys_wanted))

    # A plain line reader rather than CSV.jl, matching `load_indoor_activity_climatology`: `src/`
    # deliberately does not depend on CSV (it is in `test/aqua.jl`'s stale-deps ignore list, because
    # only the run script uses it).
    isfile(path) || error("radiation flows not found at `$(path)`")
    lines = readlines(path)
    length(lines) > 1 || error("radiation flows `$(path)` is empty")
    strip(lines[1]) == "origin,destination,flow" || error(
        "radiation flows `$(path)` has unexpected header `$(lines[1])` " *
            "(expected `origin,destination,flow`)"
    )

    flows = zeros(Float64, L, L)
    seen_origins = Set{String}()
    for (n, line) in enumerate(@view lines[2:end])
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == 3 || error(
            "radiation flows `$(path)` line $(n + 1): expected 3 fields, got $(length(fields))"
        )
        origin = lowercase(String(strip(fields[1])))
        haskey(index, origin) || continue
        push!(seen_origins, origin)
        destination = lowercase(String(strip(fields[2])))
        haskey(index, destination) || continue
        flows[index[origin], index[destination]] = parse(Float64, fields[3])
    end

    missing_origins = setdiff(Set(keys_wanted), seen_origins)
    isempty(missing_origins) || throw(
        ArgumentError(
            "locations absent from $(basename(path)): $(sort(collect(missing_origins)))"
        )
    )

    # The source diagonal is already zero; assert rather than assume, since a self-flow would
    # silently become within-location contact that `A_com` is not supposed to control.
    for x in 1:L
        iszero(flows[x, x]) || throw(
            ArgumentError(
                "$(basename(path)) has a non-zero self-flow for $(keys_wanted[x]) " *
                    "($(flows[x, x])); the radiation matrix is expected to be off-diagonal only",
            )
        )
    end

    for x in 1:L
        total = sum(view(flows, x, :))
        total > 0 || throw(
            ArgumentError(
                "location $(keys_wanted[x]) has no retained flow to any of " *
                    "$(keys_wanted); it would be isolated from the joint model. Widen the " *
                    "location set or check $(basename(path)).",
            )
        )
        flows[x, :] ./= total
    end
    return flows
end

"""
    contact_matrix(M, a_com) -> Matrix{Float64}

`T = (1 - a_com) I + a_com M`: the mixing amplitude `a_com` interpolates between purely
within-location contact (`a_com = 0`, giving the identity) and the normalized radiation matrix
(`a_com = 1`).

Built once per parameter set rather than per ODE step in the hot path — the rate closure inlines
the same expression elementwise so it can pick up a candidate's `A_com` without allocating.
"""
function contact_matrix(M::AbstractMatrix, a_com::Real)
    L = size(M, 1)
    T = (1 - a_com) .* Matrix(1.0I, L, L) .+ a_com .* M
    return T
end
