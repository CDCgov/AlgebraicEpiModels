# Unit tests for ContactStratification and its convenience constructors
# This generalizes the old AgeStratification tests to work with any ContactStratification

@testsnippet ContactStratificationSetup begin
    using AlgebraicEpiMech

    # Various ContactStratification instances
    age_single = AgeStratification([:adult])
    age_two = AgeStratification([:child, :adult])
    age_multi = AgeStratification([:child, :adult, :elderly])

    geo_two = GeographicStratification([:urban, :rural])
    geo_multi = GeographicStratification([:urban, :suburban, :rural])

    # Direct ContactStratification for testing the base type
    risk_strat = ContactStratification([:low, :high], :risk)
end

@testitem "ContactStratification constructor" setup = [ContactStratificationSetup] begin
    @testset "Basic construction" begin
        # Single stratum
        @test age_single.stratum_names == [:adult]
        @test age_single.label == :age
        @test length(age_single.stratum_names) == 1

        # Two strata
        @test age_two.stratum_names == [:child, :adult]
        @test age_two.label == :age
        @test length(age_two.stratum_names) == 2

        # Multiple strata
        @test age_multi.stratum_names == [:child, :adult, :elderly]
        @test age_multi.label == :age
        @test length(age_multi.stratum_names) == 3

        # Direct ContactStratification
        @test risk_strat.stratum_names == [:low, :high]
        @test risk_strat.label == :risk
    end

    @testset "Validation" begin
        # Empty stratum vector should error
        @test_throws ErrorException ContactStratification(Symbol[], :test)
        @test_throws ErrorException AgeStratification(Symbol[])
        @test_throws ErrorException GeographicStratification(Symbol[])

        # Duplicate stratum names should error
        @test_throws ErrorException ContactStratification([:a, :b, :a], :test)
        @test_throws ErrorException AgeStratification([:child, :adult, :child])
        @test_throws ErrorException GeographicStratification([:urban, :urban])
    end
end

@testitem "AgeStratification convenience constructor" setup = [ContactStratificationSetup] begin
    @testset "Splatted construction" begin
        # Two age groups via splatting
        age_strat = AgeStratification(:child, :adult)
        @test age_strat.stratum_names == [:child, :adult]
        @test age_strat.label == :age
        @test length(age_strat.stratum_names) == 2

        # Three age groups via splatting
        age_strat_3 = AgeStratification(:child, :adult, :elderly)
        @test age_strat_3.stratum_names == [:child, :adult, :elderly]
        @test age_strat_3.label == :age
        @test length(age_strat_3.stratum_names) == 3

        # Single age group via splatting
        age_strat_1 = AgeStratification(:population)
        @test age_strat_1.stratum_names == [:population]
        @test age_strat_1.label == :age
        @test length(age_strat_1.stratum_names) == 1
    end

    @testset "String to Symbol conversion" begin
        # Convenience constructor should convert strings to symbols
        age_strat_2_splat = AgeStratification("child", "adult")
        @test age_strat_2_splat.stratum_names == age_two.stratum_names
        @test age_strat_2_splat.label == :age

        # Mixed string and symbol input
        age_strat_mixed = AgeStratification("child", :adult, "elderly")
        @test age_strat_mixed.stratum_names == age_multi.stratum_names
        @test age_strat_mixed.label == :age
    end

    @testset "Validation in convenience constructor" begin
        # Duplicate names should still error
        @test_throws ErrorException AgeStratification(:child, :adult, :child)
    end
end

@testitem "GeographicStratification convenience constructor" setup = [ContactStratificationSetup] begin
    @testset "Splatted construction" begin
        # Two geographic regions via splatting
        geo_strat = GeographicStratification(:urban, :rural)
        @test geo_strat.stratum_names == [:urban, :rural]
        @test geo_strat.label == :geography
        @test length(geo_strat.stratum_names) == 2

        # Three geographic regions via splatting
        geo_strat_3 = GeographicStratification(:urban, :suburban, :rural)
        @test geo_strat_3.stratum_names == [:urban, :suburban, :rural]
        @test geo_strat_3.label == :geography
        @test length(geo_strat_3.stratum_names) == 3
    end

    @testset "String to Symbol conversion" begin
        # Convenience constructor should convert strings to symbols
        geo_strat_splat = GeographicStratification("urban", "rural")
        @test geo_strat_splat.stratum_names == geo_two.stratum_names
        @test geo_strat_splat.label == :geography
    end
end

@testitem "ContactStratification type relationships" setup = [ContactStratificationSetup] begin
    @testset "All convenience constructors return ContactStratification" begin
        @test age_single isa ContactStratification
        @test age_two isa ContactStratification
        @test geo_two isa ContactStratification
        @test risk_strat isa ContactStratification
    end

    @testset "All stratifications are Stratification subtypes" begin
        @test age_single isa Stratification
        @test geo_two isa Stratification
        @test risk_strat isa Stratification
    end

    @testset "Labels distinguish stratification types" begin
        @test age_two.label == :age
        @test geo_two.label == :geography
        @test risk_strat.label == :risk

        # Labels can be arbitrary
        custom = ContactStratification([:a, :b], :custom_label)
        @test custom.label == :custom_label
    end
end
