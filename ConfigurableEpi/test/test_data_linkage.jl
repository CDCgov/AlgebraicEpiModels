using ConfigurableEpi, Test
using DataFramesMeta
using StaticArrays: SVector
using Dates: Date

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
end
