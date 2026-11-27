using Test
using ConfigurableEpi
using LinearAlgebra: Diagonal

@testset "Age mixing" begin
    @testset "AgeBand and the partition check" begin
        @test AgeBand("infant", 0, 12).name == :infant
        @test_throws ArgumentError AgeBand(:bad, 12, 12)
        @test_throws ArgumentError AgeBand(:bad, -1, 12)
        @test_throws ArgumentError AgeBand(:bad, Inf, Inf)
        ok = [AgeBand(:a, 0, 12), AgeBand(:b, 12, Inf)]
        @test validate_age_bands(ok) === ok
        @test_throws ArgumentError validate_age_bands(AgeBand[])
        @test_throws ArgumentError validate_age_bands([AgeBand(:a, 1, 12), AgeBand(:b, 12, Inf)])
        @test_throws ArgumentError validate_age_bands([AgeBand(:a, 0, 12), AgeBand(:b, 24, Inf)])
        @test_throws ArgumentError validate_age_bands([AgeBand(:a, 0, 12), AgeBand(:b, 12, 24)])
        @test_throws ArgumentError validate_age_bands([AgeBand(:a, 0, 12), AgeBand(:a, 12, Inf)])
    end

    @testset "AgeMixing validates its bands, populations and settings" begin
        bands = [AgeBand(:young, 0, 240), AgeBand(:old, 240, Inf)]
        school, other = [0.0 1.0; 2.0 0.0], [3.0 1.0; 2.0 4.0]
        m = AgeMixing(bands, [100.0, 50.0], (; school, other))
        @test keys(m.contacts) == (:school, :other)
        @test m.contacts.school == school && m.contacts.other == other
        # school + other = [3 2; 4 4], eigenvalues (7 ± √33) / 2
        @test contact_spectral_radius(m) ≈ (7 + sqrt(33)) / 2
        @test AgeMixing(bands, [1, 1], other).contacts == (; all = other)

        @test_throws ArgumentError AgeMixing(reverse(bands), [1.0, 1.0], other)
        @test_throws DimensionMismatch AgeMixing(bands, [1.0], other)
        @test_throws DimensionMismatch AgeMixing(bands, [1.0, 1.0], (; school, other = ones(3, 3)))
        @test_throws ArgumentError AgeMixing(bands, [1.0, 1.0], (; school = -school))
        @test_throws ArgumentError AgeMixing(bands, [1.0, 1.0], (; school = fill(NaN, 2, 2)))
        @test_throws ArgumentError AgeMixing(bands, [1.0, 0.0], other)
        @test_throws ArgumentError AgeMixing(bands, [1.0, 1.0], (;))
    end

    mktempdir() do dir
        distribution_path = joinpath(dir, "age_distribution.csv")
        contact_path = joinpath(dir, "age_contact_matrix.csv")
        school_path = joinpath(dir, "age_school_contact_matrix.csv")
        # Three base bands: a first year, years 1-4, and an open adult band. `N .* C` is symmetric.
        write(
            distribution_path,
            "location,age_band,lower_months,upper_months,share\n" *
                "xx,age_0,0,12,0.1\nxx,age_1_4,12,60,0.3\nxx,age_5_plus,60,inf,0.6\n" *
                "yy,age_0,0,12,1.0\n",
        )
        contact_table(rates) = "location,contactor,contactee,contacts_per_day\n" * join(
            (
                "xx,$g,$h,$(rates[i, j])" for (i, g) in enumerate(("age_0", "age_1_4", "age_5_plus"))
                    for (j, h) in enumerate(("age_0", "age_1_4", "age_5_plus"))
            ), "\n",
        )
        write(contact_path, contact_table([1.0 3.0 6.0; 1.0 4.0 2.0; 1.0 1.0 8.0]))
        write(school_path, contact_table([0.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]))
        load(bands; location = "xx", kwargs...) = load_age_mixing(location, bands; distribution_path, contact_path, kwargs...)
        base = [AgeBand(:age_0, 0, 12), AgeBand(:age_1_4, 12, 60), AgeBand(:age_5_plus, 60, Inf)]

        @testset "the base grid comes back unchanged" begin
            result = load(base)
            @test result isa AgeMixing
            @test result.bands == base
            @test result.populations ≈ [0.1, 0.3, 0.6]
            @test result.contacts.all ≈ [1.0 3.0 6.0; 1.0 4.0 2.0; 1.0 1.0 8.0]
            @test load(base; location = "XX").contacts == result.contacts   # case-insensitive
            @test load(base; population = 1000).populations ≈ [100.0, 300.0, 600.0]
        end

        @testset "a union of base bands is the population-weighted mean" begin
            result = load([AgeBand(:child, 0, 60), AgeBand(:adult, 60, Inf)])
            @test result.populations ≈ [0.4, 0.6]
            # A child's contacts with children: (0.1 * (1 + 3) + 0.3 * (1 + 4)) / 0.4.
            @test result.contacts.all[1, 1] ≈ (0.1 * 4.0 + 0.3 * 5.0) / 0.4
            @test result.contacts.all[1, 2] ≈ (0.1 * 6.0 + 0.3 * 2.0) / 0.4
            @test result.contacts.all[2, 1] ≈ 2.0
            @test result.contacts.all[2, 2] ≈ 8.0
        end

        @testset "a divided band keeps its per-capita rates and splits contacts with it" begin
            bands = [
                AgeBand(:m0_2, 0, 3), AgeBand(:m3_11, 3, 12), AgeBand(:age_1_4, 12, 60),
                AgeBand(:age_5_plus, 60, Inf),
            ]
            result = load(bands)
            contacts = result.contacts.all
            @test result.populations ≈ [0.025, 0.075, 0.3, 0.6]
            # Both parts of the first year contact the older bands at the parent's rate, and
            # contacts with the first year divide by the parts' size (a quarter, three quarters).
            @test contacts[1, 3:4] ≈ [3.0, 6.0]
            @test contacts[2, 3:4] ≈ [3.0, 6.0]
            @test contacts[3, 1:2] ≈ [0.25, 0.75]
            @test contacts[1, 1:2] ≈ [0.25, 0.75]
            # Regrouping preserves each band's total contacts and the symmetry of `N .* C`.
            @test sum(contacts[3, :]) ≈ 1.0 + 4.0 + 2.0
            total = Diagonal(result.populations) * contacts
            @test total ≈ total'
        end

        @testset "several settings regroup on the same bands" begin
            bands = [AgeBand(:child, 0, 60), AgeBand(:adult, 60, Inf)]
            result = load(bands; contact_path = (; all = contact_path, school = school_path))
            @test keys(result.contacts) == (:all, :school)
            @test result.contacts.all == load(bands).contacts.all
            @test result.contacts.school ≈ [0.3 * 2.0 / 0.4 0.0; 0.0 1.0]
        end

        @testset "refusals" begin
            @test_throws ArgumentError load(base; location = "zz")
            # The open base band cannot be divided.
            @test_throws ArgumentError load([AgeBand(:young, 0, 120), AgeBand(:old, 120, Inf)])
            # `yy` has a distribution row but no contact rows.
            @test_throws ArgumentError load([AgeBand(:all, 0, Inf)]; location = "yy")
            @test_throws ErrorException load(base; distribution_path = joinpath(dir, "nope.csv"))
            bad = joinpath(dir, "bad.csv")
            write(bad, "location,band,share\nxx,age_0,1.0\n")
            @test_throws ErrorException load(base; distribution_path = bad)
            write(bad, "location,age_band,lower_months,upper_months,share\n")
            @test_throws ErrorException load(base; distribution_path = bad)
            write(bad, "location,contactor,contactee,contacts_per_day\nxx,age_0,age_9,1.0\n")
            @test_throws ErrorException load(base; contact_path = bad)
            write(bad, "location,contactor,contactee,contacts_per_day\nxx,age_0,age_0\n")
            @test_throws ErrorException load(base; contact_path = bad)
        end
    end

    @testset "load_setting_contacts reads one location on the tables' age groups" begin
        mktempdir() do dir
            matrix_path = joinpath(dir, "contact_matrices.csv")
            population_path = joinpath(dir, "age_populations.csv")
            groups = ("0-17", "18-64", "65+")
            school = [6.0 0.5 0.0; 0.25 0.5 0.0; 0.0 0.0 0.0]
            other = [3.0 4.0 0.5; 2.0 6.0 1.0; 1.0 4.0 2.0]
            rows = ["location,layer,row_age,col_age,value"]
            for (layer, m) in (("school", school), ("other", other)), (i, g) in enumerate(groups), (j, h) in enumerate(groups)
                push!(rows, "us,$layer,$g,$h,$(m[i, j])", "ny,$layer,$g,$h,$(2 * m[i, j])")
            end
            push!(rows, "us,school,80+,0-17,1.0")   # a group the model does not use
            write(matrix_path, join(rows, "\n"))
            write(population_path, "location,age_group,population\nus,0-17,100\nus,18-64,200\nus,65+,50\nny,0-17,10\n")
            bands = [AgeBand("0-17", 0, 216), AgeBand("18-64", 216, 780), AgeBand("65+", 780, Inf)]
            load(location = "us"; kwargs...) = load_setting_contacts(location, bands; matrix_path, population_path, kwargs...)

            us = load()
            @test us.bands == bands
            @test us.contacts == (; school, other)
            @test us.populations == [100.0, 200.0, 50.0]
            flow = us.populations .* (us.contacts.school .+ us.contacts.other)
            @test flow ≈ flow'
            @test load("US").contacts == us.contacts
            @test load(; settings = (:other, :school)).contacts == (; other, school)

            # `ny` has every contact but only one population.
            @test_throws ErrorException load("ny")
            @test_throws ErrorException load("atlantis")
            @test_throws ErrorException load(; settings = (:school,))   # an unnamed layer
            @test_throws ErrorException load_setting_contacts(
                "us", [AgeBand("0-17", 0, 216), AgeBand("18+", 216, Inf)]; matrix_path, population_path,
            )
        end
    end
end
