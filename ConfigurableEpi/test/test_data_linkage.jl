using ConfigurableEpi, Test
using DataFramesMeta
using StaticArrays: SVector
using Dates: Date, Day

@testset "Data Linkage" begin
    @testset "ObservationSchema" begin
        # Basic construction
        schema = ObservationSchema((:ma, :ny), :date)
        @test n_observations(schema) == 2
        @test observation_names(schema) == (:ma, :ny)
        @test schema.time_column == :date

        # From vector
        schema_vec = ObservationSchema([:a, :b, :c], :time)
        @test n_observations(schema_vec) == 3
        @test observation_names(schema_vec) == (:a, :b, :c)

        # Must have at least 1 observation
        @test_throws ArgumentError ObservationSchema((), :date)

        # Names must be unique
        @test_throws ArgumentError ObservationSchema((:a, :a, :b), :date)
    end

    @testset "DataLink construction" begin
        schema = ObservationSchema((:obs1, :obs2), :date)

        # Default (wide format)
        link1 = DataLink(schema)
        @test link1.schema === schema
        @test link1.value_column == :value
        @test link1.pivot_column === nothing

        # Long format with pivot
        link2 = DataLink(schema, :cases, :location)
        @test link2.value_column == :cases
        @test link2.pivot_column == :location
    end

    @testset "build_observations - wide format" begin
        schema = ObservationSchema((:ma, :ny), :date)
        link = DataLink(schema)

        # Wide format DataFrame
        df = DataFrame(
            date = Date.(["2024-01-01", "2024-01-02", "2024-01-03"]),
            ma = [100.0, 110.0, 120.0],
            ny = [200.0, 210.0, 220.0]
        )

        y, times = build_observations(df, link)

        @test length(y) == 3
        @test y[1] ≈ SVector(100.0, 200.0)
        @test y[2] ≈ SVector(110.0, 210.0)
        @test y[3] ≈ SVector(120.0, 220.0)
        @test times == Date.(["2024-01-01", "2024-01-02", "2024-01-03"])
    end

    @testset "build_observations - long format" begin
        schema = ObservationSchema((:ma, :ny), :date)
        link = DataLink(schema, :value, :location)

        # Long format DataFrame
        df = DataFrame(
            date = repeat(Date.(["2024-01-01", "2024-01-02"]), inner = 2),
            location = repeat([:ma, :ny], 2),
            value = [100.0, 200.0, 110.0, 210.0]
        )

        y, times = build_observations(df, link)

        @test length(y) == 2
        @test y[1] ≈ SVector(100.0, 200.0)  # ma, ny at t=1
        @test y[2] ≈ SVector(110.0, 210.0)  # ma, ny at t=2
        @test times == Date.(["2024-01-01", "2024-01-02"])
    end

    @testset "build_observations - order matches schema" begin
        # Schema specifies order: ny first, then ma
        schema = ObservationSchema((:ny, :ma), :date)
        link = DataLink(schema, :value, :location)

        df = DataFrame(
            date = repeat([Date("2024-01-01")], 2),
            location = [:ma, :ny],  # Data has ma first
            value = [100.0, 200.0]  # ma=100, ny=200
        )

        y, times = build_observations(df, link)

        # Output should follow schema order: ny first
        @test y[1] ≈ SVector(200.0, 100.0)  # ny=200, ma=100
    end

    @testset "build_observations - missing column error" begin
        schema = ObservationSchema((:ma, :ny), :date)
        link = DataLink(schema)

        df = DataFrame(
            date = [Date("2024-01-01")],
            ma = [100.0]            # missing :ny column
        )

        @test_throws ArgumentError build_observations(df, link)
    end

    @testset "build_observations - missing time column error" begin
        schema = ObservationSchema((:ma,), :date)
        link = DataLink(schema)

        df = DataFrame(
            time = [Date("2024-01-01")],  # wrong column name
            ma = [100.0]
        )

        @test_throws ArgumentError build_observations(df, link)
    end

    @testset "require_complete_grid" begin
        schema = ObservationSchema((:ny, :ca), :date)
        link = DataLink(schema, :counts, :location)
        dates = Date.(["2024-01-01", "2024-01-08"])
        complete = DataFrame(
            date = repeat(dates, 2),
            location = repeat(["ny", "ca"]; inner = 2),
            counts = [10.0, 11.0, 90.0, 91.0],
        )
        # A complete grid passes through untouched, so it composes as a guard.
        @test require_complete_grid(complete, link) === complete

        # A HOLE. `build_observations` would report only the first one it reaches; this names every
        # offending cell in a single error, because the fix is upstream in the slice.
        holed = complete[Not(4), :]
        err = try
            require_complete_grid(holed, link)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("missing 1 cell", err.msg)
        @test occursin("ca", err.msg)
        @test occursin("2024-01-08", err.msg)

        # A DUPLICATE is worse than a hole: `pivot_to_wide` takes `findfirst`, so without this check
        # the extra row is silently discarded and the run proceeds on arbitrarily chosen data.
        duped = vcat(complete, DataFrame(date = [dates[1]], location = ["ny"], counts = [999.0]))
        err2 = try
            require_complete_grid(duped, link)
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("duplicated 1 cell", err2.msg)
        @test occursin("x2", err2.msg)

        # Both at once are reported together.
        err3 = try
            require_complete_grid(
                vcat(
                    holed, DataFrame(
                        date = [dates[1]], location = ["ny"], counts = [999.0],
                    )
                ), link
            )
        catch e
            e
        end
        @test occursin("missing", err3.msg) && occursin("duplicated", err3.msg)

        # Wide-format links have no grid to check.
        wide_link = DataLink(schema, :value, nothing)
        @test require_complete_grid(complete, wide_link) === complete
    end

    @testset "build_observations - missing value in data" begin
        schema = ObservationSchema((:ma, :ny), :date)
        link = DataLink(schema)

        df = DataFrame(
            date = [Date("2024-01-01")],
            ma = [100.0],
            ny = [missing]
        )

        @test_throws ArgumentError build_observations(df, link)
    end

    @testset "build_observation_schema from obs_specs" begin
        # Create mock observation specs
        noise = NegBinomialNoise(0.1, 10.0)

        obs_specs = (
            SignalObservationSpec(1, noise; name = :ma),
            SignalObservationSpec(2, noise; name = :ny),
        )

        schema = build_observation_schema(obs_specs; time_column = :report_date)

        @test n_observations(schema) == 2
        @test observation_names(schema) == (:ma, :ny)
        @test schema.time_column == :report_date
    end

    @testset "build_observations from obs_specs directly" begin
        noise = NegBinomialNoise(0.1, 10.0)

        obs_specs = (
            SignalObservationSpec(1, noise; name = :loc_a),
            SignalObservationSpec(2, noise; name = :loc_b),
        )

        # Long format data
        df = DataFrame(
            report_date = repeat([Date("2024-01-01"), Date("2024-01-02")], inner = 2),
            location = repeat([:loc_a, :loc_b], 2),
            cases = [100.0, 150.0, 110.0, 160.0]
        )

        y,
            times = build_observations(
            df, obs_specs;
            time_column = :report_date,
            value_column = :cases,
            pivot_column = :location
        )

        @test length(y) == 2
        @test y[1] ≈ SVector(100.0, 150.0)  # loc_a, loc_b at t=1
        @test y[2] ≈ SVector(110.0, 160.0)  # loc_a, loc_b at t=2
    end

    # A reporting gap must become predict-only slots on the `step_days` grid, never a shorter
    # series: `fit_forecast!` takes one entry per slot and `missing` where nothing was reported.
    @testset "reindex_to_grid fills absent grid dates with missing" begin
        d0 = Date("2024-09-20")
        df = DataFrame(date = d0 .+ Day.([0, 1, 5, 6]), counts = [1.0, 2.0, 6.0, 7.0], extra = 1:4)
        grid = reindex_to_grid(df, 1)
        @test grid.date == collect(d0:Day(1):(d0 + Day(6)))
        @test names(grid) == ["date", "counts"]   # other columns are dropped
        @test isequal(grid.counts, [1.0, 2.0, missing, missing, missing, 6.0, 7.0])
        @test eltype(grid.counts) == Union{Missing, Float64}
        # A gap-free frame comes back unchanged.
        full = DataFrame(date = d0 .+ Day.(0:3), counts = [1.0, 2.0, 3.0, 4.0])
        @test isequal(reindex_to_grid(full, 1).counts, full.counts)
        # `start`/`stop` extend the grid with missing slots (a trailing gap before a report date).
        ext = reindex_to_grid(df, 1; start = d0 - Day(1), stop = d0 + Day(8))
        @test nrow(ext) == 10 && ismissing(ext.counts[1]) && all(ismissing, ext.counts[9:10])
        # Weekly grid and several value columns.
        weekly = DataFrame(
            date = d0 .+ Day.([0, 7, 21]), counts = [1.0, 2.0, 4.0], raw_counts = [1, 2, 4],
            reporting_fraction = [1.0, 1.0, 0.5],
        )
        wk = reindex_to_grid(weekly, 7; value_cols = (:counts, :raw_counts, :reporting_fraction))
        @test wk.date == d0 .+ Day.([0, 7, 14, 21])
        @test isequal(wk.raw_counts, [1, 2, missing, 4]) && isequal(wk.reporting_fraction, [1.0, 1.0, missing, 0.5])
        # Joint (multi-signal) rows keep their vector values.
        joint = DataFrame(date = d0 .+ Day.([0, 14]), counts = [[1.0, 2.0], [3.0, 4.0]])
        jg = reindex_to_grid(joint, 7)
        @test eltype(jg.counts) == Union{Missing, Vector{Float64}}
        @test isequal(jg.counts, [[1.0, 2.0], missing, [3.0, 4.0]])
        # Errors: off-grid date, duplicate, unsorted, inverted range, bad step, empty input.
        @test_throws ArgumentError reindex_to_grid(DataFrame(date = d0 .+ Day.([0, 3]), counts = [1.0, 2.0]), 7)
        @test_throws ArgumentError reindex_to_grid(DataFrame(date = [d0, d0], counts = [1.0, 2.0]), 1)
        @test_throws ArgumentError reindex_to_grid(DataFrame(date = [d0 + Day(1), d0], counts = [1.0, 2.0]), 1)
        @test_throws ArgumentError reindex_to_grid(df, 1; start = d0 + Day(9))
        @test_throws ArgumentError reindex_to_grid(df, 0)
        @test_throws ArgumentError reindex_to_grid(DataFrame(date = Date[], counts = Float64[]), 1)
        @test_throws ArgumentError reindex_to_grid(DataFrame(date = [d0], value = [1.0]), 1)
    end
end
