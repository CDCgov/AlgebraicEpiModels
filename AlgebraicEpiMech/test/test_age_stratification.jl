# DEPRECATED: This file tests the old AgeStratification API
# New tests should use test_contact_stratification.jl which covers
# ContactStratification and all its convenience constructors (AgeStratification,
# GeographicStratification, etc.)
#
# This file is kept temporarily for backward compatibility but uses the new
# ContactStratification field names (stratum_names instead of age_group_names)

@testsnippet AgeStratificationSetup begin
    using AlgebraicEpiMech

    age_strat = AgeStratification([:adult])
    age_strat_2 = AgeStratification([:child, :adult])
    age_strat_multi = AgeStratification([:child, :adult, :elderly])
end

@testitem "AgeStratification constructor (deprecated - see test_contact_stratification.jl)" setup = [AgeStratificationSetup] begin
    @testset "Basic construction" begin
        # Single age group
        @test age_strat.stratum_names == [:adult]
        @test age_strat.label == :age
        @test length(age_strat.stratum_names) == 1

        # Two age groups
        @test age_strat_2.stratum_names == [:child, :adult]
        @test age_strat_2.label == :age
        @test length(age_strat_2.stratum_names) == 2

        # Multiple age groups
        @test age_strat_multi.stratum_names == [:child, :adult, :elderly]
        @test age_strat_multi.label == :age
        @test length(age_strat_multi.stratum_names) == 3
    end

    @testset "Validation" begin
        # Empty age group vector should error
        @test_throws ErrorException AgeStratification(Symbol[])

        # Duplicate age group names should error
        @test_throws ErrorException AgeStratification([:child, :adult, :child])
        @test_throws ErrorException AgeStratification([:adult, :adult])
    end
end

@testitem "AgeStratification convenience constructor (deprecated - see test_contact_stratification.jl)" setup = [AgeStratificationSetup] begin
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
        @test age_strat_2_splat.stratum_names == age_strat_2.stratum_names
        @test age_strat_2_splat.label == :age

        # Mixed string and symbol input
        age_strat_mixed = AgeStratification("child", :adult, "elderly")
        @test age_strat_mixed.stratum_names == age_strat_multi.stratum_names
        @test age_strat_mixed.label == :age
    end

    @testset "Validation in convenience constructor" begin
        # Duplicate names should still error
        @test_throws ErrorException AgeStratification(:child, :adult, :child)
    end
end
