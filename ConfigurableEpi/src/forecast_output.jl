# ============================================================================
# DRAW-LEVEL FORECAST OUTPUT
# ============================================================================
#
# The seven-column core intentionally matches EpiAutoGP's `samples.parquet`
# producer in cfa-stf-routine-forecasting. Keeping the constructor and writer
# here, rather than in a submodel runner, gives every ConfigurableEpi model the
# same deployment boundary. Backtests may append metadata columns (for example
# `origin`); a single-origin production run needs no extension columns.
# ============================================================================

const FORECAST_SAMPLE_COLUMNS = (
    :date,
    Symbol(".value"),
    Symbol(".draw"),
    Symbol(".variable"),
    :resolution,
    :geo_value,
    :disease,
)

function _append_sample_metadata!(frame, metadata::NamedTuple)
    n = nrow(frame)
    existing = Set(Symbol.(names(frame)))
    for (name, value) in pairs(metadata)
        name in existing && throw(
            ArgumentError("sample metadata column `$name` conflicts with the routine schema")
        )
        if value isa AbstractVector
            length(value) == n || throw(
                DimensionMismatch(
                    "sample metadata column `$name` has $(length(value)) values for $n rows"
                )
            )
            frame[!, name] = collect(value)
        else
            frame[!, name] = fill(value, n)
        end
        push!(existing, name)
    end
    return frame
end

"""
    forecast_sample_rows(samples, target_dates;
        geo_value, disease, variable, resolution, metadata=(;)) -> DataFrame

Convert coherent forecast trajectories `samples[horizon, draw]` to the exact
seven-column long-sample contract produced by EpiAutoGP in
`cfa-stf-routine-forecasting`. Rows are draw-major: every target date for draw
1, then every target date for draw 2, and so on. Draw identifiers are `Int32`
for cross-language compatibility with the routine R and Polars postprocessors.

`metadata` appends model-agnostic scalar or row-length columns after the routine
columns. This is useful for a multi-origin backtest (`metadata=(origin=date,)`),
while a production run can omit it and receive the exact EpiAutoGP schema.
"""
function forecast_sample_rows(
        samples::AbstractMatrix,
        target_dates;
        geo_value::AbstractString,
        disease::AbstractString,
        variable::AbstractString,
        resolution::AbstractString,
        metadata::NamedTuple = (;),
    )
    n_ahead, n_draws = size(samples)
    n_ahead > 0 || throw(ArgumentError("forecast samples must contain at least one horizon"))
    n_draws > 0 || throw(ArgumentError("forecast samples must contain at least one draw"))
    length(target_dates) == n_ahead || throw(
        DimensionMismatch(
            "forecast samples have $n_ahead horizons but $(length(target_dates)) target dates"
        )
    )
    n_draws <= typemax(Int32) || throw(
        ArgumentError("forecast sample count $n_draws exceeds the Int32 draw contract")
    )

    dates = Date.(collect(target_dates))
    n = n_ahead * n_draws
    frame = DataFrame(
        :date => repeat(dates, n_draws),
        Symbol(".value") => Float64.(vec(samples)),
        Symbol(".draw") => repeat(Int32.(1:n_draws); inner = n_ahead),
        Symbol(".variable") => fill(String(variable), n),
        :resolution => fill(String(resolution), n),
        :geo_value => fill(String(geo_value), n),
        :disease => fill(String(disease), n),
    )
    return _append_sample_metadata!(frame, metadata)
end

"""
    forecast_sample_rows(samples, target_dates;
        geo_values, diseases, variables, resolution, metadata=(;)) -> DataFrame

Multi-signal sibling for `samples[horizon, draw, signal]`. The three label
vectors identify each signal in model order; each signal is formatted with the
same routine contract as the matrix method.
"""
function forecast_sample_rows(
        samples::AbstractArray{<:Real, 3},
        target_dates;
        geo_values,
        diseases,
        variables,
        resolution::AbstractString,
        metadata::NamedTuple = (;),
    )
    n_signals = size(samples, 3)
    n_signals > 0 || throw(ArgumentError("forecast samples must contain at least one signal"))
    for (label, values) in (
            ("geo_values", geo_values),
            ("diseases", diseases),
            ("variables", variables),
        )
        length(values) == n_signals || throw(
            DimensionMismatch(
                "$label has $(length(values)) entries for $n_signals forecast signals"
            )
        )
    end
    frames = [
        forecast_sample_rows(
            view(samples, :, :, signal), target_dates;
            geo_value = String(geo_values[signal]),
            disease = String(diseases[signal]),
            variable = String(variables[signal]),
            resolution,
            metadata,
        ) for signal in 1:n_signals
    ]
    return reduce((left, right) -> append!(left, right), frames)
end

function _validate_forecast_sample_table(table)
    table_names = Symbol.(names(table))
    missing_columns = setdiff(collect(FORECAST_SAMPLE_COLUMNS), table_names)
    isempty(missing_columns) || throw(
        ArgumentError(
            "forecast sample table is missing routine column(s): " *
                join(string.(missing_columns), ", ")
        )
    )
    eltype(table[!, :date]) == Date || throw(
        ArgumentError(
            "forecast sample `date` must have element type Date; got " *
                string(eltype(table[!, :date]))
        )
    )
    eltype(table[!, Symbol(".draw")]) == Int32 || throw(
        ArgumentError(
            "forecast sample `.draw` must have element type Int32; got " *
                string(eltype(table[!, Symbol(".draw")]))
        )
    )
    nrow(table) > 0 || throw(ArgumentError("forecast sample table must not be empty"))
    return table_names
end

_quote_duckdb_string(value::AbstractString) =
    "'" * replace(value, "'" => "''") * "'"

"""
    write_forecast_samples(path, table_or_tables) -> path

Write one forecast sample table, or an iterable of schema-compatible tables, to
`samples.parquet` using the same DuckDB registration and `COPY ... FORMAT
parquet` route as EpiAutoGP. Multiple inputs are combined inside DuckDB, which
lets callers retain compact sample matrices and construct one origin at a time
instead of first concatenating a very large Julia `DataFrame`.
"""
write_forecast_samples(path::AbstractString, table::DataFrame) =
    write_forecast_samples(path, (table,))

function write_forecast_samples(path::AbstractString, tables)
    mkpath(dirname(path))
    con = connect(DB, ":memory:")
    expected_names = nothing
    wrote_table = false
    try
        for table in tables
            table_names = _validate_forecast_sample_table(table)
            if expected_names === nothing
                expected_names = table_names
            elseif table_names != expected_names
                throw(
                    ArgumentError(
                        "forecast sample batches must have identical columns; expected " *
                            "$(expected_names), got $(table_names)"
                    )
                )
            end
            register_data_frame(con, table, "forecast_sample_batch")
            try
                if wrote_table
                    execute(
                        con,
                        "INSERT INTO forecast_samples SELECT * FROM forecast_sample_batch",
                    )
                else
                    execute(
                        con,
                        "CREATE TABLE forecast_samples AS SELECT * FROM forecast_sample_batch",
                    )
                    wrote_table = true
                end
            finally
                unregister_data_frame(con, "forecast_sample_batch")
            end
        end
        wrote_table || throw(ArgumentError("at least one forecast sample table is required"))
        execute(
            con,
            "COPY forecast_samples TO $(_quote_duckdb_string(path)) (FORMAT parquet)",
        )
    finally
        close(con)
    end
    return path
end
