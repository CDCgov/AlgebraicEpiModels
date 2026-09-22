# Linking a DataFrame of observations (long or wide) to the observation vectors the filters read.

"""
    ObservationSchema(names, time_column)

The observation names, in the order the measurement model emits them, and the time column.
"""
struct ObservationSchema{N}
    names::NTuple{N, Symbol}
    time_column::Symbol
    function ObservationSchema(names::NTuple{N, Symbol}, time_column::Symbol) where {N}
        N >= 1 || throw(ArgumentError("a schema needs at least one observation"))
        allunique(names) || throw(ArgumentError("duplicate observation names: $(filter(n -> count(==(n), names) > 1, unique(names)))"))
        return new{N}(names, time_column)
    end
end
ObservationSchema(names::AbstractVector{Symbol}, time_column::Symbol) = ObservationSchema(Tuple(names), time_column)

n_observations(::ObservationSchema{N}) where {N} = N
observation_names(schema::ObservationSchema) = schema.names

"""
    build_observation_schema(obs_specs; time_column = :date)

The schema named by a tuple of observation specs, in their order.
"""
build_observation_schema(obs_specs::Tuple{Vararg{ObservationSpec}}; time_column::Symbol = :date) =
    ObservationSchema(Tuple(spec.name for spec in obs_specs), time_column)

"""
    DataLink(schema[, value_column = :value, pivot_column = nothing])

How a DataFrame maps onto a schema: already wide (one column per observation name), or long with
a `pivot_column` whose values are the observation names and a `value_column`.
"""
struct DataLink{N}
    schema::ObservationSchema{N}
    value_column::Symbol
    pivot_column::Union{Nothing, Symbol}
end
DataLink(schema::ObservationSchema) = DataLink(schema, :value, nothing)
DataLink(schema::ObservationSchema, value_column::Symbol) = DataLink(schema, value_column, nothing)

function _require_columns(df, columns)
    for col in columns
        col in propertynames(df) || throw(ArgumentError("DataFrame missing column: $col"))
    end
    return nothing
end

"""
    pivot_to_wide(df, link::DataLink) -> DataFrame

Wide frame with the time column followed by one column per schema name, sorted by time. A wide
input is validated and sorted; a long one is unstacked on `link.pivot_column`.
"""
function pivot_to_wide(df::DataFrame, link::DataLink)
    schema, time_col = link.schema, link.schema.time_column
    if link.pivot_column === nothing
        _require_columns(df, (time_col, schema.names...))
        return sort!(copy(df), time_col)
    end
    _require_columns(df, (time_col, link.pivot_column, link.value_column))
    present = Set(Symbol.(df[!, link.pivot_column]))
    for name in schema.names
        name in present ||
            throw(ArgumentError("schema expects observation '$name' but it is not in $(link.pivot_column); found $(collect(present))"))
    end
    grouped = groupby(df, time_col)
    wide = DataFrame(time_col => [first(g[!, time_col]) for g in grouped])
    for name in schema.names
        wide[!, name] = Union{Float64, Missing}[
            (i = findfirst(==(name), Symbol.(g[!, link.pivot_column])); i === nothing ? missing : g[i, link.value_column])
                for g in grouped
        ]
    end
    return sort!(wide, time_col)
end

"""
    require_complete_grid(df, link) -> df

Assert exactly one row for every (observation, time) cell of a long frame, reporting every
missing and duplicated cell at once (the pivot alone would take the first duplicate silently).
"""
function require_complete_grid(df::DataFrame, link::DataLink{N}) where {N}
    link.pivot_column === nothing && return df
    time_col, pivot_col = link.schema.time_column, link.pivot_column
    _require_columns(df, (time_col, pivot_col))
    counts = Dict{Tuple{Symbol, Any}, Int}()
    for i in 1:nrow(df)
        key = (Symbol(df[i, pivot_col]), df[i, time_col])
        counts[key] = get(counts, key, 0) + 1
    end
    times = sort(unique(df[!, time_col]))
    cells = [(name, t, get(counts, (name, t), 0)) for name in link.schema.names for t in times]
    missing_cells = [(n, t) for (n, t, c) in cells if c == 0]
    duplicate_cells = [(n, t, c) for (n, t, c) in cells if c > 1]
    isempty(missing_cells) && isempty(duplicate_cells) && return df
    parts = String[]
    isempty(missing_cells) ||
        push!(parts, "missing $(length(missing_cells)) cell(s): " * join(("($n, $t)" for (n, t) in missing_cells), ", "))
    isempty(duplicate_cells) ||
        push!(parts, "duplicated $(length(duplicate_cells)) cell(s): " * join(("($n, $t) x$c" for (n, t, c) in duplicate_cells), ", "))
    throw(
        ArgumentError(
            "incomplete $pivot_col-by-$time_col grid over $(length(times)) time point(s) and $N observation(s): " *
                join(parts, "; "),
        )
    )
end

"""
    build_observations(df, link::DataLink; T = Float64) -> (; y, times)
    build_observations(df, obs_specs; time_column = :date, value_column = :value, pivot_column = nothing, T = Float64)

Observation vectors `y::Vector{SVector{N, T}}` in schema order, one per time point, and the times.
Missing cells throw.
"""
function build_observations(df::DataFrame, link::DataLink{N}; T::Type = Float64) where {N}
    wide = pivot_to_wide(df, link)
    times = wide[!, link.schema.time_column]
    y = map(1:nrow(wide)) do t
        SVector{N, T}(
            ntuple(Val(N)) do i
                v = wide[t, link.schema.names[i]]
                ismissing(v) && throw(ArgumentError("missing value at time $(times[t]) for $(link.schema.names[i])"))
                T(v)
            end
        )
    end
    return (; y, times)
end

function build_observations(
        df::DataFrame, obs_specs::Tuple{Vararg{ObservationSpec}};
        time_column::Symbol = :date, value_column::Symbol = :value, pivot_column::Union{Nothing, Symbol} = nothing,
        T::Type = Float64,
    )
    link = DataLink(build_observation_schema(obs_specs; time_column), value_column, pivot_column)
    return build_observations(df, link; T)
end
