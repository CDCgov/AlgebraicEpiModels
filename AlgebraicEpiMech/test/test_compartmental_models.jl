@testitem "Compartmental model constructors" begin
    using AlgebraicEpiMech

    @testset "SI model" begin
        # Single stage
        model = SI()
        @test model.number_I_stages == 1
        @test model.number_of_states == 2  # S + I

        # Multi-stage
        model_multi = SI(number_I_stages = 3)
        @test model_multi.number_I_stages == 3
        @test model_multi.number_of_states == 4  # S + I1 + I2 + I3
    end

    @testset "SEI model" begin
        # Single stage
        model = SEI()
        @test model.number_E_stages == 1
        @test model.number_I_stages == 1
        @test model.number_of_states == 3  # S + E + I

        # Multi-stage
        model_multi = SEI(number_E_stages = 2, number_I_stages = 3)
        @test model_multi.number_E_stages == 2
        @test model_multi.number_I_stages == 3
        @test model_multi.number_of_states == 6  # S + E1 + E2 + I1 + I2 + I3
    end

    @testset "SIR model" begin
        # Single stage
        model = SIR()
        @test model.number_I_stages == 1
        @test model.number_of_states == 3  # S + I + R

        # Multi-stage
        model_multi = SIR(number_I_stages = 2)
        @test model_multi.number_I_stages == 2
        @test model_multi.number_of_states == 4  # S + I1 + I2 + R
    end

    @testset "SEIR model" begin
        model = SEIR(number_E_stages = 2, number_I_stages = 2)
        @test model.number_E_stages == 2
        @test model.number_I_stages == 2
        @test model.number_of_states == 6  # S + E1 + E2 + I1 + I2 + R
    end

    @testset "SIS model" begin
        model = SIS(number_I_stages = 3)
        @test model.number_I_stages == 3
        @test model.number_of_states == 4  # S + I1 + I2 + I3
    end

    @testset "SEIS model" begin
        model = SEIS(number_E_stages = 1, number_I_stages = 2)
        @test model.number_E_stages == 1
        @test model.number_I_stages == 2
        @test model.number_of_states == 4  # S + E + I1 + I2
    end

    @testset "SEIRS model" begin
        model = SEIRS()
        @test model.number_E_stages == 1
        @test model.number_I_stages == 1
        @test model.number_of_states == 4  # S + E + I + R
    end

    @testset "Type hierarchy" begin
        @test SI() isa CompartmentalModel
        @test SEI() isa CompartmentalModel
        @test SIR() isa CompartmentalModel
        @test SEIR() isa CompartmentalModel
        @test SIS() isa CompartmentalModel
        @test SEIS() isa CompartmentalModel
        @test SEIRS() isa CompartmentalModel
    end

    @testset "Stage counts must be positive" begin
        @test_throws ArgumentError SI(number_I_stages = 0)
        @test_throws ArgumentError SIR(number_I_stages = -1)
        @test_throws ArgumentError SIS(number_I_stages = 0)
        @test_throws ArgumentError SEI(number_E_stages = 0)
        @test_throws ArgumentError SEI(number_I_stages = -1)
        @test_throws ArgumentError SEIR(number_E_stages = -1)
        @test_throws ArgumentError SEIR(number_I_stages = 0)
        @test_throws ArgumentError SEIS(number_E_stages = 0)
        @test_throws ArgumentError SEIS(number_I_stages = -1)
        @test_throws ArgumentError SEIRS(number_E_stages = -1)
        @test_throws ArgumentError SEIRS(number_I_stages = 0)
    end
end
