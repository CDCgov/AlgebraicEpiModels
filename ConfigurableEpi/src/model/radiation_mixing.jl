# Cross-location contact from a caller-owned radiation-flows file: `M[x, y]` is the share of
# location `x`'s out-of-location contact that is with `y` (zero diagonal, rows re-normalised over
# the selected locations), and `contact_matrix` mixes it with within-location contact.

"""
    load_radiation_matrix(locations; path) -> Matrix{Float64}

Row-normalised contact matrix for `locations` (in order) from an `origin,destination,flow` table.
Errors when a location is absent, has a self-flow, or retains no flow to the others.
"""
function load_radiation_matrix(locations; path::AbstractString)
    wanted = [lowercase(String(l)) for l in locations]
    allunique(wanted) || throw(ArgumentError("locations must be unique, got $wanted"))
    isempty(wanted) && throw(ArgumentError("need at least one location"))
    index = Dict(k => i for (i, k) in enumerate(wanted))
    isfile(path) || error("radiation flows not found at `$path`")
    lines = readlines(path)
    length(lines) > 1 || error("radiation flows `$path` is empty")
    strip(lines[1]) == "origin,destination,flow" ||
        error("radiation flows `$path` has header `$(lines[1])`, expected `origin,destination,flow`")

    L = length(wanted)
    flows = zeros(L, L)
    seen = Set{String}()
    for (n, line) in enumerate(@view lines[2:end])
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == 3 || error("radiation flows `$path` line $(n + 1): expected 3 fields, got $(length(fields))")
        origin = lowercase(String(strip(fields[1])))
        haskey(index, origin) || continue
        push!(seen, origin)
        destination = lowercase(String(strip(fields[2])))
        haskey(index, destination) || continue
        flow = parse(Float64, fields[3])
        isfinite(flow) && flow >= 0 ||
            error("radiation flows `$path` line $(n + 1): flow must be finite and non-negative, got $flow")
        flows[index[origin], index[destination]] = flow
    end
    missing_origins = setdiff(Set(wanted), seen)
    isempty(missing_origins) ||
        throw(ArgumentError("locations absent from $(basename(path)): $(sort!(collect(missing_origins)))"))
    for x in 1:L
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
