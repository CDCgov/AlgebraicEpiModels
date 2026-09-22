# Draw-level forecast output in the seven-column `samples.parquet` contract of
# cfa-stf-routine-forecasting (the EpiAutoGP producer), written through DuckDB.

const FORECAST_SAMPLE_COLUMNS =
    (:date, Symbol(".value"), Symbol(".draw"), Symbol(".variable"), :resolution, :geo_value, :disease)

function _append_sample_metadata!(frame, metadata::NamedTuple)
    n = nrow(frame)
    for (name, value) in pairs(metadata)
        name in propertynames(frame) &&
            throw(ArgumentError("sample metadata column `$name` conflicts with the routine schema"))
        if value isa AbstractVector
            length(value) == n ||
                throw(DimensionMismatch("sample metadata column `$name` has $(length(value)) values for $n rows"))
            frame[!, name] = collect(value)
        else
            frame[!, name] = fill(value, n)
        end
    end
    return frame
end

"""
    forecast_sample_rows(samples, target_dates; geo_value, disease, variable, resolution, metadata = (;))
    forecast_sample_rows(samples, target_dates; geo_values, diseases, variables, resolution, metadata = (;))

Long draw-level rows from `samples[horizon, draw]` (or `[horizon, draw, signal]` with one label
per signal): draw-major, with `Int32` draw ids. `metadata` appends scalar or row-length columns
after the routine ones (a backtest's `origin`, say).
"""
function forecast_sample_rows(
        samples::AbstractMatrix, target_dates;
        geo_value::AbstractString, disease::AbstractString, variable::AbstractString, resolution::AbstractString,
        metadata::NamedTuple = (;),
    )
    n_ahead, n_draws = size(samples)
    n_ahead > 0 && n_draws > 0 || throw(ArgumentError("forecast samples must contain at least one horizon and one draw"))
    length(target_dates) == n_ahead ||
        throw(DimensionMismatch("forecast samples have $n_ahead horizons but $(length(target_dates)) target dates"))
    n_draws <= typemax(Int32) || throw(ArgumentError("forecast sample count $n_draws exceeds the Int32 draw contract"))
    n = n_ahead * n_draws
    frame = DataFrame(
        :date => repeat(Date.(collect(target_dates)), n_draws),
        Symbol(".value") => Float64.(vec(samples)),
        Symbol(".draw") => repeat(Int32.(1:n_draws); inner = n_ahead),
        Symbol(".variable") => fill(String(variable), n),
        :resolution => fill(String(resolution), n),
        :geo_value => fill(String(geo_value), n),
        :disease => fill(String(disease), n),
    )
    return _append_sample_metadata!(frame, metadata)
end

function forecast_sample_rows(
        samples::AbstractArray{<:Real, 3}, target_dates;
        geo_values, diseases, variables, resolution::AbstractString, metadata::NamedTuple = (;),
    )
    n_signals = size(samples, 3)
    n_signals > 0 || throw(ArgumentError("forecast samples must contain at least one signal"))
    for (label, values) in (("geo_values", geo_values), ("diseases", diseases), ("variables", variables))
        length(values) == n_signals ||
            throw(DimensionMismatch("$label has $(length(values)) entries for $n_signals forecast signals"))
    end
    frames = [
        forecast_sample_rows(
                view(samples, :, :, s), target_dates;
                geo_value = String(geo_values[s]), disease = String(diseases[s]), variable = String(variables[s]),
                resolution, metadata,
            ) for s in 1:n_signals
    ]
    return reduce(append!, frames)
end

function _validate_forecast_sample_table(table)
    columns = propertynames(table)
    missing_columns = setdiff(collect(FORECAST_SAMPLE_COLUMNS), columns)
    isempty(missing_columns) ||
        throw(ArgumentError("forecast sample table is missing routine column(s): " * join(string.(missing_columns), ", ")))
    eltype(table[!, :date]) == Date ||
        throw(ArgumentError("forecast sample `date` must have element type Date; got $(eltype(table[!, :date]))"))
    eltype(table[!, Symbol(".draw")]) == Int32 ||
        throw(ArgumentError("forecast sample `.draw` must have element type Int32; got $(eltype(table[!, Symbol(".draw")]))"))
    nrow(table) > 0 || throw(ArgumentError("forecast sample table must not be empty"))
    return columns
end

_quote_duckdb_string(value::AbstractString) = "'" * replace(value, "'" => "''") * "'"

"""
    write_forecast_samples(path, table_or_tables) -> path

Write one sample table, or an iterable of schema-identical tables, to `samples.parquet` through
DuckDB (`COPY ... FORMAT parquet`), combining the batches inside DuckDB.
"""
write_forecast_samples(path::AbstractString, table::DataFrame) = write_forecast_samples(path, (table,))

function write_forecast_samples(path::AbstractString, tables)
    mkpath(dirname(path))
    con = connect(DB, ":memory:")
    expected = nothing
    try
        for table in tables
            columns = _validate_forecast_sample_table(table)
            expected === nothing || columns == expected ||
                throw(ArgumentError("forecast sample batches must have identical columns; expected $expected, got $columns"))
            register_data_frame(con, table, "forecast_sample_batch")
            try
                execute(
                    con,
                    expected === nothing ? "CREATE TABLE forecast_samples AS SELECT * FROM forecast_sample_batch" :
                        "INSERT INTO forecast_samples SELECT * FROM forecast_sample_batch",
                )
            finally
                unregister_data_frame(con, "forecast_sample_batch")
            end
            expected = columns
        end
        expected === nothing && throw(ArgumentError("at least one forecast sample table is required"))
        execute(con, "COPY forecast_samples TO $(_quote_duckdb_string(path)) (FORMAT parquet)")
    finally
        close(con)
    end
    return path
end
