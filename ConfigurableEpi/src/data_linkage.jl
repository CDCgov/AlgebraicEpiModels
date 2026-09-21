# ============================================================================
# DATA LINKAGE - Connect DataFrames to UKF observations
# ============================================================================
#
# This module provides an API to link epidemiological data in DataFrames
# to the observation vectors expected by LowLevelParticleFilters.
#
# Key concepts:
# 1. ObservationSchema: Describes what the model expects (names, order)
# 2. DataLink: Maps DataFrame columns to observation schema
# 3. build_observations: Transforms DataFrame to Vector{SVector{ny}}
#
# The UKF expects y as a Vector of observation vectors, where each element
# is the observation at time t. For multivariate observations (e.g.,
# hospitalizations by location or age group), the order in each observation
# vector must match the order expected by the measurement model.
#
# ============================================================================

"""
    ObservationSchema

Describes the observation structure expected by a measurement model.

The schema defines:
1. What observations are expected (by name)
2. In what order they appear in the observation vector
3. What time column to use

This is built automatically from `ObservationSpec` tuples, or can be
constructed manually for custom configurations.

# Fields
- `names`: Tuple of observation names (Symbols) in the order they appear
- `time_column`: Symbol for the time/date column in the DataFrame

# Example
```julia
# Two-location model observing hospitalizations in each
schema = ObservationSchema((:ma, :ny), :date)

# Age-stratified model
schema = ObservationSchema((:child_hosp, :adult_hosp, :elderly_hosp), :report_date)
```
"""
struct ObservationSchema{N}
    names::NTuple{N, Symbol}
    time_column::Symbol

    function ObservationSchema(names::NTuple{N, Symbol}, time_column::Symbol) where {N}
        N >= 1 || throw(ArgumentError("Must have at least 1 observation"))
        uniques = unique(names)
        if length(uniques) != N
            duplicates = filter(n -> count(==(n), names) > 1, uniques)
            throw(ArgumentError("Duplicate observation names: $(duplicates)"))
        end
        return new{N}(names, time_column)
    end
end

# Convenience constructor from Vector
function ObservationSchema(names::Vector{Symbol}, time_column::Symbol)
    return ObservationSchema(Tuple(names), time_column)
end

"""
    n_observations(schema::ObservationSchema)

Return the number of observations in the schema.
"""
n_observations(::ObservationSchema{N}) where {N} = N

"""
    observation_names(schema::ObservationSchema)

Return the observation names as a tuple.
"""
observation_names(schema::ObservationSchema) = schema.names

"""
    build_observation_schema(obs_specs::Tuple{Vararg{ObservationSpec}}; time_column=:date)

Build ObservationSchema from measurement model observation specs.

# Arguments
- `obs_specs`: Tuple of SignalObservationSpec or AggregatedSignalSpec
- `time_column`: Name of the time column in the DataFrame

# Returns
ObservationSchema matching the order of obs_specs.

# Example
```julia
obs_specs = (
    SignalObservationSpec(1, noise, :ma),
    SignalObservationSpec(2, noise, :ny),
)
schema = build_observation_schema(obs_specs; time_column=:report_date)
# schema.names = (:ma, :ny)
```
"""
function build_observation_schema(
        obs_specs::Tuple{Vararg{ObservationSpec}};
        time_column::Symbol = :date
    )
    names = Tuple(spec.name for spec in obs_specs)
    return ObservationSchema(names, time_column)
end

# ============================================================================
# Data Link - Maps DataFrame to observations
# ============================================================================

"""
    DataLink

Configuration for linking a DataFrame to an observation schema.

Handles transformation from long-format data (one row per observation-time pair)
to the wide-format observation vectors expected by the UKF.

# Fields
- `schema`: ObservationSchema defining expected structure
- `value_column`: Column containing observation values
- `pivot_column`: Optional column for pivoting (e.g., :location, :age_group)

# Two data formats are supported:

## 1. Wide format (one column per observation)
DataFrame already has columns matching schema.names:
```
date       | ma  | ny  | ca
2024-01-01 | 100 | 200 | 150
2024-01-02 | 105 | 210 | 160
```
Use: `DataLink(schema)` - no pivoting needed.

## 2. Long format (one row per observation-time pair)
```
date       | location | value
2024-01-01 | ma       | 100
2024-01-01 | ny       | 200
2024-01-01 | ca       | 150
2024-01-02 | ma       | 105
...
```
Use: `DataLink(schema, :value, :location)` - pivots on location column.
"""
struct DataLink{N}
    schema::ObservationSchema{N}
    value_column::Symbol
    pivot_column::Union{Nothing, Symbol}

    function DataLink(
            schema::ObservationSchema{N},
            value_column::Symbol = :value,
            pivot_column::Union{Nothing, Symbol} = nothing
        ) where {N}
        return new{N}(schema, value_column, pivot_column)
    end
end

# ============================================================================
# Pivot long to wide format
# ============================================================================

"""
    pivot_to_wide(df::DataFrame, link::DataLink) -> DataFrame

Convert long-format DataFrame to wide format for observation extraction.

# Arguments
- `df`: Long-format DataFrame with time, pivot_column, and value_column
- `link`: DataLink specifying columns and schema

# Returns
Wide-format DataFrame with one row per time point and columns for each observation.
"""
function pivot_to_wide(df::DataFrame, link::DataLink{N}) where {N}
    if isnothing(link.pivot_column)
        # Already wide format, just validate
        return _validate_wide_format(df, link)
    end

    schema = link.schema
    time_col = schema.time_column
    value_col = link.value_column
    pivot_col = link.pivot_column

    # Check required columns exist
    required = [time_col, pivot_col, value_col]
    for col in required
        col in Symbol.(names(df)) || throw(ArgumentError("DataFrame missing column: $col"))
    end

    # Get unique pivot values in the data
    pivot_values = unique(df[!, pivot_col])

    # Validate all expected names are in pivot values
    for name in schema.names
        name in Symbol.(pivot_values) || throw(
            ArgumentError(
                "Schema expects observation '$name' but not found in $pivot_col column. " *
                    "Found: $(pivot_values)"
            )
        )
    end

    # Pivot using unstack-like operation via groupby + combine
    # Group by time, then create named columns from pivot values
    grouped = groupby(df, time_col)

    # Build wide DataFrame using combine to avoid map over GroupedDataFrame
    # First get unique times in order
    times = combine(grouped, time_col => first => time_col)[!, time_col]
    wide_df = DataFrame(time_col => times)

    # Add column for each observation in schema order by iterating groups
    for name in schema.names
        values = Vector{Union{Float64, Missing}}(undef, length(grouped))
        for (i, g) in enumerate(grouped)
            row = findfirst(==(name), Symbol.(g[!, pivot_col]))
            values[i] = isnothing(row) ? missing : g[row, value_col]
        end
        wide_df[!, name] = values
    end

    return sort!(wide_df, time_col)
end

"""
    require_complete_grid(df, link) -> df

Assert that `df` carries exactly one row for every (pivot value, time) cell the schema names, and
return it unchanged.

`pivot_to_wide` cannot do this itself, in either direction. A **missing** cell becomes `missing` and
is only reported later by `build_observations`, one cell at a time, so a run with a ragged grid dies
on its first hole rather than telling you the shape of the problem. A **duplicate** cell is worse:
the pivot takes `findfirst`, so the extra rows are silently discarded and the run proceeds on
arbitrarily chosen data.

Both are reported here in one error listing every offending cell, because the fix is upstream in the
slice and you want the whole picture to fix it once.
"""
function require_complete_grid(df::DataFrame, link::DataLink{N}) where {N}
    isnothing(link.pivot_column) && return df
    schema = link.schema
    time_col = schema.time_column
    pivot_col = link.pivot_column
    for col in (time_col, pivot_col)
        col in Symbol.(names(df)) || throw(ArgumentError("DataFrame missing column: $col"))
    end

    counts = Dict{Tuple{Symbol, Any}, Int}()
    for i in 1:nrow(df)
        key = (Symbol(df[i, pivot_col]), df[i, time_col])
        counts[key] = get(counts, key, 0) + 1
    end

    times = sort(unique(df[!, time_col]))
    missing_cells = Tuple{Symbol, Any}[]
    duplicate_cells = Tuple{Symbol, Any, Int}[]
    for name in schema.names, t in times
        n = get(counts, (name, t), 0)
        n == 0 && push!(missing_cells, (name, t))
        n > 1 && push!(duplicate_cells, (name, t, n))
    end

    if !isempty(missing_cells) || !isempty(duplicate_cells)
        parts = String[]
        isempty(missing_cells) || push!(
            parts,
            "missing $(length(missing_cells)) cell(s): " *
                join(("($(n), $(t))" for (n, t) in missing_cells), ", "),
        )
        isempty(duplicate_cells) || push!(
            parts,
            "duplicated $(length(duplicate_cells)) cell(s): " *
                join(("($(n), $(t)) x$(c)" for (n, t, c) in duplicate_cells), ", "),
        )
        throw(
            ArgumentError(
                "incomplete $(pivot_col)-by-$(time_col) grid over " *
                    "$(length(times)) time point(s) and $(N) observation(s) — " *
                    join(parts, "; "),
            )
        )
    end
    return df
end

"""
    _validate_wide_format(df::DataFrame, link::DataLink) -> DataFrame

Validate wide-format DataFrame has required columns.
"""
function _validate_wide_format(df::DataFrame, link::DataLink{N}) where {N}
    schema = link.schema

    # Check time column
    schema.time_column in Symbol.(names(df)) ||
        throw(ArgumentError("DataFrame missing time column: $(schema.time_column)"))

    # Check observation columns
    for name in schema.names
        name in Symbol.(names(df)) ||
            throw(ArgumentError("DataFrame missing observation column: $name"))
    end

    return sort!(copy(df), schema.time_column)
end

# ============================================================================
# Build observation vectors for UKF
# ============================================================================

"""
    build_observations(df::DataFrame, link::DataLink{N}; T=Float64) -> (y, times)

Build observation vectors for UKF from DataFrame.

# Arguments
- `df`: DataFrame with observation data (long or wide format)
- `link`: DataLink specifying data structure and schema
- `T`: Element type for observation vectors (default: Float64)

# Returns
Tuple of:
- `y`: Vector{SVector{N,T}} of observations, one per time point
- `times`: Vector of time values (dates or numbers)

# Example
```julia
# Long format data
df = DataFrame(
    date = repeat([Date(2024,1,1), Date(2024,1,2)], inner=2),
    location = repeat([:ma, :ny], 2),
    value = [100, 200, 105, 210]
)

schema = ObservationSchema((:ma, :ny), :date)
link = DataLink(schema, :value, :location)
y, times = build_observations(df, link)
# y[1] = SVector(100.0, 200.0)  # ma, ny at t=1
# y[2] = SVector(105.0, 210.0)  # ma, ny at t=2
```

# Note
Missing values will throw an error. Handle missing data before calling this function.
"""
function build_observations(df::DataFrame, link::DataLink{N}; T::Type = Float64) where {N}
    # Pivot if needed
    wide_df = pivot_to_wide(df, link)

    schema = link.schema
    time_col = schema.time_column

    n_times = nrow(wide_df)
    times = wide_df[!, time_col]

    # Build observation vectors in schema order
    y = Vector{SVector{N, T}}(undef, n_times)

    for t in 1:n_times
        obs = ntuple(Val(N)) do i
            val = wide_df[t, schema.names[i]]
            ismissing(val) &&
                throw(ArgumentError("Missing value at time $(times[t]) for $(schema.names[i])"))
            T(val)
        end
        y[t] = SVector{N}(obs)
    end

    return (y = y, times = times)
end

# ============================================================================
# Convenience: Direct from obs_specs
# ============================================================================

"""
    build_observations(df::DataFrame, obs_specs::Tuple{Vararg{ObservationSpec}};
                       time_column=:date, value_column=:value, pivot_column=nothing, T=Float64)

Convenience function to build observations directly from measurement model specs.

# Arguments
- `df`: DataFrame with observation data
- `obs_specs`: Tuple of observation specifications from measurement model
- `time_column`: Name of time/date column
- `value_column`: Name of value column (for long format)
- `pivot_column`: Name of pivot column (for long format), or nothing for wide format
- `T`: Element type

# Returns
Tuple of (y, times) as in `build_observations(df, link)`.

# Example
```julia
obs_specs = (
    SignalObservationSpec(1, noise, :ma),
    SignalObservationSpec(2, noise, :ny),
)

# Long format
y, times = build_observations(df, obs_specs;
    time_column=:date, value_column=:cases, pivot_column=:location)
```
"""
function build_observations(
        df::DataFrame,
        obs_specs::Tuple{Vararg{ObservationSpec}};
        time_column::Symbol = :date,
        value_column::Symbol = :value,
        pivot_column::Union{Nothing, Symbol} = nothing,
        T::Type = Float64
    )
    schema = build_observation_schema(obs_specs; time_column = time_column)

    link = if isnothing(pivot_column)
        DataLink(schema)
    else
        DataLink(schema, value_column, pivot_column)
    end

    return build_observations(df, link; T = T)
end

# ============================================================================
# Univariate convenience
# ============================================================================

"""
    build_observations(df::DataFrame, obs_name::Symbol, time_column::Symbol; T=Float64)

Convenience for univariate observations (single column).

# Arguments
- `df`: DataFrame with observation data
- `obs_name`: Name of the observation column
- `time_column`: Name of time/date column
- `T`: Element type

# Returns
Tuple of (y, times) where y is Vector{SVector{1,T}}.

# Example
```julia
df = DataFrame(date = Date.(2024, 1, 1:10), cases = rand(100:200, 10))
y, times = build_observations(df, :cases, :date)
```
"""
function build_observations(
        df::DataFrame,
        obs_name::Symbol,
        time_column::Symbol;
        T::Type = Float64
    )
    schema = ObservationSchema((obs_name,), time_column)
    link = DataLink(schema)
    return build_observations(df, link; T = T)
end

# ============================================================================
# Validation helpers
# ============================================================================

"""
    validate_observations(y::Vector{SVector{N,T}}, schema::ObservationSchema{N}) where {N,T}

Validate observation vector matches schema dimensions.

# Returns
`true` if valid, throws ArgumentError otherwise.
"""
function validate_observations(
        y::Vector{SVector{N, T}},
        schema::ObservationSchema{M}
    ) where {N, T, M}
    N == M || throw(
        ArgumentError(
            "Observation dimension mismatch: got $N, schema expects $M"
        )
    )
    return true
end

"""
    validate_observations(y::Vector, expected_n_obs::Int)

Validate observation vector dimensions against expected count.
"""
function validate_observations(y::Vector{SVector{N, T}}, expected_n_obs::Int) where {N, T}
    N == expected_n_obs || throw(
        ArgumentError(
            "Observation dimension mismatch: got $N, expected $expected_n_obs"
        )
    )
    return true
end

# ============================================================================
# Summary/diagnostic functions
# ============================================================================

"""
    summarize_observations(y::Vector{SVector{N,T}}, schema::ObservationSchema{N}) where {N,T}

Print summary statistics for observation data.
"""
function summarize_observations(
        y::Vector{SVector{N, T}},
        schema::ObservationSchema{N}
    ) where {N, T}
    println("Observation Summary:")
    println("  Time points: $(length(y))")
    println("  Dimensions: $N")
    println("  Observations:")
    for (i, name) in enumerate(schema.names)
        values = [y[t][i] for t in 1:length(y)]
        println("    $name: min=$(minimum(values)), max=$(maximum(values)), mean=$(sum(values) / length(values))")
    end
    return
end

"""
    observation_as_matrix(y::Vector{SVector{N,T}}) where {N,T}

Convert observation vector to Matrix (T × N) for analysis.

Rows are time points, columns are observation dimensions.
"""
function observation_as_matrix(y::Vector{SVector{N, T}}) where {N, T}
    n_times = length(y)
    mat = Matrix{T}(undef, n_times, N)
    for t in 1:n_times
        for i in 1:N
            mat[t, i] = y[t][i]
        end
    end
    return mat
end
