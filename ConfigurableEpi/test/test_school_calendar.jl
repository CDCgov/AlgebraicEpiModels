using Test
using ConfigurableEpi
using Dates: Date

@testset "School calendar" begin
    @testset "SchoolCalendar validates its days" begin
        cal = SchoolCalendar(Date(2024, 1, 1), [1.0, 0.5, 0.0])
        @test ConfigurableEpi.last_date(cal) == Date(2024, 1, 3)
        @test_throws ArgumentError SchoolCalendar(Date(2024, 1, 1), Float64[])
        @test_throws ArgumentError SchoolCalendar(Date(2024, 1, 1), [1.2])
        @test_throws ArgumentError SchoolCalendar(Date(2024, 1, 1), [NaN])
    end

    @testset "load_school_calendar reads consecutive days" begin
        mktempdir() do dir
            path = joinpath(dir, "school_calendar.csv")
            write(path, "date,school_open_fraction\n2024-12-23,1.0\n2024-12-24,0.5\n\n2024-12-25,0.0\n")
            cal = load_school_calendar(path)
            @test cal.first_date == Date(2024, 12, 23)
            @test cal.values == [1.0, 0.5, 0.0]
            write(path, "date,school_open_fraction\n2024-12-23,1.0\n2024-12-25,0.0\n")
            @test_throws ErrorException load_school_calendar(path)
            write(path, "date,open\n2024-12-23,1.0\n")
            @test_throws ErrorException load_school_calendar(path)
            write(path, "date,school_open_fraction\n")
            @test_throws ErrorException load_school_calendar(path)
        end
    end

    @testset "AnchoredCalendar is piecewise constant per day and held outside its range" begin
        cal = SchoolCalendar(Date(2024, 1, 1), [0.1, 0.2, 0.3, 0.4])
        anchored = AnchoredCalendar(cal, Date(2024, 1, 2))
        @test anchored(0.0) == 0.2
        @test anchored(0.99) == 0.2
        @test anchored(1.0) == 0.3
        @test anchored(-1.0) == 0.1
        @test anchored(-50.0) == 0.1
        @test anchored(2.5) == 0.4
        @test anchored(400.0) == 0.4

        @test assert_calendar_covers(cal, Date(2024, 1, 1), Date(2024, 1, 4)) === nothing
        @test_throws ErrorException assert_calendar_covers(cal, Date(2023, 12, 31), Date(2024, 1, 4))
        @test_throws ErrorException assert_calendar_covers(cal, Date(2024, 1, 1), Date(2024, 1, 5))
    end
end
