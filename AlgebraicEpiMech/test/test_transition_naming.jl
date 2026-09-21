# Unit tests for transition naming from UWD structure
# Tests that create_model generates unique transition names for ODE rate assignment

@testsnippet TransitionNamingSetup begin
    using AlgebraicPetri
    using Catlab
    using AlgebraicEpiMech

    # Create schema for testing
    one_pop_schema = OnePopulationSchema(population_type = :Individual)
end

# Test simple SIR model transition naming
@testitem "SIR model has unique transition names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SIR())
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 2 transitions
    @test length(tnames) == 2

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern
    @test :transmission_S_I in tnames
    @test :I_to_R in tnames
end

# Test multi-stage SI model transition naming
@testitem "Multi-stage SI model has unique transmission names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SI(number_I_stages = 3))
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 5 transitions: 3 transmissions + 2 progressions
    @test length(tnames) == 5

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern for transmissions
    @test :transmission_S_I1 in tnames
    @test :transmission_S_I2 in tnames
    @test :transmission_S_I3 in tnames

    # Check expected naming pattern for progressions
    @test :I1_to_I2 in tnames
    @test :I2_to_I3 in tnames
end

# Test SEIR model transition naming
@testitem "SEIR model has unique transition names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SEIR())
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 3 transitions
    @test length(tnames) == 3

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern
    @test :transmission_S_I in tnames
    @test :E_to_I in tnames
    @test :I_to_R in tnames
end

# Test multi-stage SEIR model transition naming
@testitem "Multi-stage SEIR model has unique transition names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(
        one_pop_schema,
        SEIR(number_E_stages = 2, number_I_stages = 3)
    )
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 8 transitions:
    # - 3 transmissions (one per I stage)
    # - 1 E progression (E1->E2)
    # - 2 I progressions (I1->I2, I2->I3)
    # - 1 E to I transition (E2->I1)
    # - 1 I to R transition (I3->R)
    @test length(tnames) == 8

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern for transmissions
    @test :transmission_S_I1 in tnames
    @test :transmission_S_I2 in tnames
    @test :transmission_S_I3 in tnames

    # Check expected naming pattern for progressions
    @test :E1_to_E2 in tnames
    @test :I1_to_I2 in tnames
    @test :I2_to_I3 in tnames
    @test :E2_to_I1 in tnames
    @test :I3_to_R in tnames
end

# Test SEI model transition naming
@testitem "SEI model has unique transition names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SEI())
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 2 transitions
    @test length(tnames) == 2

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern
    @test :transmission_S_I in tnames
    @test :E_to_I in tnames
end

# Test SIS model transition naming
@testitem "SIS model has unique transition names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SIS())
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 2 transitions
    @test length(tnames) == 2

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern
    @test :transmission_S_I in tnames
    @test :I_to_S in tnames
end

# Test SEIS model transition naming
@testitem "SEIS model has unique transition names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SEIS())
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 3 transitions
    @test length(tnames) == 3

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern
    @test :transmission_S_I in tnames
    @test :E_to_I in tnames
    @test :I_to_S in tnames
end

# Test SEIRS model transition naming
@testitem "SEIRS model has unique transition names" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SEIRS())
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should have 4 transitions
    @test length(tnames) == 4

    # All transition names should be unique
    @test length(unique(tnames)) == length(tnames)

    # Check expected naming pattern
    @test :transmission_S_I in tnames
    @test :E_to_I in tnames
    @test :I_to_R in tnames
    @test :R_to_S in tnames
end

# Test that transition names enable unambiguous ODE rate assignment
@testitem "Multi-stage model enables unique rate parameter assignment" setup = [TransitionNamingSetup] begin
    using LabelledArrays

    typed_model = create_model(one_pop_schema, SI(number_I_stages = 3))
    pn = dom(typed_model)

    tnames = AlgebraicPetri.tnames(pn)

    # Should be able to create a parameter vector with unique names
    # This would fail if transition names were not unique
    param_dict = Dict(tname => 0.1 for tname in tnames)

    # Verify we can access each transition uniquely
    @test haskey(param_dict, :transmission_S_I1)
    @test haskey(param_dict, :transmission_S_I2)
    @test haskey(param_dict, :transmission_S_I3)
    @test haskey(param_dict, :I1_to_I2)
    @test haskey(param_dict, :I2_to_I3)

    # Verify all 5 unique parameters can be set
    @test length(param_dict) == 5
end

# Test that multi-input transitions follow the naming pattern
@testitem "Multi-input transitions follow box_name_input1_input2 pattern" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SIR())
    pn = dom(typed_model)

    # Find the transmission transition (has 2 inputs)
    transmission_idx = findfirst(
        t -> AlgebraicPetri.tname(pn, t) == :transmission_S_I, 1:AlgebraicPetri.nt(pn)
    )
    @test transmission_idx !== nothing

    # Get its inputs
    inputs = AlgebraicPetri.inputs(pn, transmission_idx)
    @test length(inputs) == 2

    # The name should reflect both inputs (sorted in reverse alphabetical order)
    input_names = sort([string(AlgebraicPetri.sname(pn, s)) for s in inputs], rev = true)
    @test input_names == ["S", "I"]

    # The transition name should follow the pattern
    @test AlgebraicPetri.tname(pn, transmission_idx) == :transmission_S_I
end

# Test that single-input transitions follow input_to_output pattern
@testitem "Single-input transitions follow input_to_output pattern" setup = [TransitionNamingSetup] begin
    typed_model = create_model(one_pop_schema, SIR())
    pn = dom(typed_model)

    # Find the recovery transition (I -> R)
    recovery_idx = findfirst(
        t -> AlgebraicPetri.tname(pn, t) == :I_to_R, 1:AlgebraicPetri.nt(pn)
    )
    @test recovery_idx !== nothing

    # Get its inputs and outputs
    inputs = AlgebraicPetri.inputs(pn, recovery_idx)
    outputs = AlgebraicPetri.outputs(pn, recovery_idx)

    @test length(inputs) == 1
    @test length(outputs) == 1

    # The name should reflect input_to_output
    input_name = AlgebraicPetri.sname(pn, inputs[1])
    output_name = AlgebraicPetri.sname(pn, outputs[1])

    @test AlgebraicPetri.tname(pn, recovery_idx) == Symbol(input_name, "_to_", output_name)
end
