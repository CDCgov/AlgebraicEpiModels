# Unit tests for create_model_uwd.jl
# Tests the exported create_model_uwd methods for all compartmental model types

@testsnippet ModelUWDSetup begin
    using AlgebraicPetri
    using Catlab
    using AlgebraicEpiMech

    # Create typing instances for testing
    one_pop_typing = OnePopulationTyping(population_type = :Individual)
    ui_typing = UninfectedInfectedTyping(
        uninfected_type = :Susceptible,
        infected_type = :Infectious
    )
    uninfected_infected_typing = UninfectedInfectedTyping(
        uninfected_type = :Susceptible,
        infected_type = :Infectious
    )
end

# Tests for SI model UWD construction
@testitem "create_model_uwd(SI) creates correct structure" setup = [ModelUWDSetup] begin
    model = SI()
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S and I junctions (2 total)
    @test length(junctions(uwd)) == 2

    # Should have 1 box (infection)
    @test length(boxes(uwd)) == 1

    # Should return a valid RelationDiagram
    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(SI) with multi-stage I" setup = [ModelUWDSetup] begin
    model = SI(number_I_stages = 3)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S and I1, I2, I3 junctions (4 total)
    @test length(junctions(uwd)) == 4

    # Should have 5 boxes (each mechanism creates 2 boxes: infection (2), I1->I2 (2), I2->I3 (1 unidirectional))
    @test length(boxes(uwd)) == 5
end

@testitem "create_model_uwd(SI) works with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    model = SI()
    uwd = create_model_uwd(ui_typing, model)

    # Should create valid UWD with typing-specific types
    @test length(junctions(uwd)) == 2
    @test length(boxes(uwd)) == 1
    @test uwd isa RelationDiagram
end

# Tests for SEI model UWD construction
@testitem "create_model_uwd(SEI) creates correct structure" setup = [ModelUWDSetup] begin
    model = SEI()
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E, and I junctions (3 total)
    @test length(junctions(uwd)) == 3

    # Should have 2 boxes (exposure, progression E->I)
    @test length(boxes(uwd)) == 2

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(SEI) with multi-stage E and I" setup = [ModelUWDSetup] begin
    model = SEI(number_E_stages = 2, number_I_stages = 3)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E1, E2, I1, I2, I3 junctions (6 total)
    @test length(junctions(uwd)) == 6

    # Boxes: exposure S->E from each Infected group (3), E1->E2 (1), E2->I1 (1), I1->I2 (1), I2->I3 (1),
    @test length(boxes(uwd)) == 7
end

@testitem "create_model_uwd(SEI) works with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    model = SEI()
    uwd = create_model_uwd(ui_typing, model)

    # Should create valid UWD with typing-specific types
    @test length(junctions(uwd)) == 3
    @test length(boxes(uwd)) == 2
    @test uwd isa RelationDiagram
end

# Tests for SIR model UWD construction
@testitem "create_model_uwd(SIR) creates correct structure" setup = [ModelUWDSetup] begin
    model = SIR()
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, I, and R junctions (3 total)
    @test length(junctions(uwd)) == 3

    # Should have 2 boxes (infection, recovery)
    @test length(boxes(uwd)) == 2

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(SIR) with multi-stage I" setup = [ModelUWDSetup] begin
    model = SIR(number_I_stages = 3)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, I1, I2, I3, R junctions (5 total)
    @test length(junctions(uwd)) == 5

    # Boxes: infection from each infectious group (3), I1->I2 (1), I2->I3 (1), I3->R (1)
    @test length(boxes(uwd)) == 6
end

@testitem "create_model_uwd(SIR) works with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    model = SIR()
    uwd = create_model_uwd(ui_typing, model)

    # Should create valid UWD
    @test length(junctions(uwd)) == 3
    @test length(boxes(uwd)) == 2
    @test uwd isa RelationDiagram
end

# Tests for SEIR model UWD construction
@testitem "create_model_uwd(SEIR) creates correct structure" setup = [ModelUWDSetup] begin
    model = SEIR()
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E, I, and R junctions (4 total)
    @test length(junctions(uwd)) == 4

    # Should have 3 boxes (exposure, E->I progression, recovery)
    @test length(boxes(uwd)) == 3

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(SEIR) with multi-stage E and I" setup = [ModelUWDSetup] begin
    model = SEIR(number_E_stages = 2, number_I_stages = 3)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E1, E2, I1, I2, I3, R junctions (7 total)
    @test length(junctions(uwd)) == 7

    # Boxes: exposure (2), E1->E2 (1), E2->I1 (1), I1->I2 (1), I2->I3 (1), I3->R (1), S->E1
    @test length(boxes(uwd)) == 8
end

@testitem "create_model_uwd(SEIR) works with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    model = SEIR()
    uwd = create_model_uwd(ui_typing, model)

    # Should create valid UWD
    @test length(junctions(uwd)) == 4
    @test length(boxes(uwd)) == 3
    @test uwd isa RelationDiagram
end

# Tests for SIS model UWD construction
@testitem "create_model_uwd(SIS) creates correct structure" setup = [ModelUWDSetup] begin
    model = SIS()
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S and I junctions (same as SI)
    @test length(junctions(uwd)) == 2

    # Should have 2 boxes (infection, reversion back to S)
    @test length(boxes(uwd)) == 2

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(SIS) with multi-stage I" setup = [ModelUWDSetup] begin
    model = SIS(number_I_stages = 3)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, I1, I2, I3 junctions (4 total)
    @test length(junctions(uwd)) == 4

    # Boxes: infection (2), I1->I2 (1), I2->I3 (1), I3->S (1), S->I1
    @test length(boxes(uwd)) == 6
end

@testitem "create_model_uwd(SIS) works with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    model = SIS()
    uwd = create_model_uwd(ui_typing, model)

    # Should create valid UWD
    @test length(junctions(uwd)) == 2
    @test length(boxes(uwd)) == 2
    @test uwd isa RelationDiagram
end

# Tests for SEIS model UWD construction
@testitem "create_model_uwd(SEIS) creates correct structure" setup = [ModelUWDSetup] begin
    model = SEIS()
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E, and I junctions (same as SEI)
    @test length(junctions(uwd)) == 3

    # Should have 3 boxes (exposure, E->I progression, reversion to S)
    @test length(boxes(uwd)) == 3

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(SEIS) with multi-stage E and I" setup = [ModelUWDSetup] begin
    model = SEIS(number_E_stages = 2, number_I_stages = 3)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E1, E2, I1, I2, I3 junctions (6 total)
    @test length(junctions(uwd)) == 6

    # Boxes: exposure (3), E1->E2 (1), E2->I1 (1), I1->I2 (1), I2->I3 (1), I3->S (1)
    @test length(boxes(uwd)) == 8
end

@testitem "create_model_uwd(SEIS) works with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    model = SEIS()
    uwd = create_model_uwd(ui_typing, model)

    # Should create valid UWD
    @test length(junctions(uwd)) == 3
    @test length(boxes(uwd)) == 3
    @test uwd isa RelationDiagram
end

# Tests for SEIRS model UWD construction
@testitem "create_model_uwd(SEIRS) creates correct structure" setup = [ModelUWDSetup] begin
    model = SEIRS()
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E, I, and R junctions (4 total)
    @test length(junctions(uwd)) == 4

    # Should have 4 boxes (exposure, E->I progression, recovery, waning immunity R->S)
    @test length(boxes(uwd)) == 4

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(SEIRS) with multi-stage E and I" setup = [ModelUWDSetup] begin
    model = SEIRS(number_E_stages = 2, number_I_stages = 3)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have S, E1, E2, I1, I2, I3, R junctions (7 total)
    @test length(junctions(uwd)) == 7

    # Boxes: exposure (3), E1->E2 (1), E2->I1 (1), I1->I2 (1), I2->I3 (1), I3->R (1), R->S (1)
    @test length(boxes(uwd)) == 9
end

@testitem "create_model_uwd(SEIRS) works with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    model = SEIRS()
    uwd = create_model_uwd(ui_typing, model)

    # Should create valid UWD
    @test length(junctions(uwd)) == 4
    @test length(boxes(uwd)) == 4
    @test uwd isa RelationDiagram
end

# Test error handling for unimplemented combinations
@testitem "create_model_uwd throws error for unimplemented combinations" setup = [ModelUWDSetup] begin
    # Create a custom typing type that has no implementations
    struct CustomSchema <: EpidemiologicalTyping end
    custom_schema = CustomSchema()

    # Should throw error for any model with unimplemented typing
    @test_throws ErrorException create_model_uwd(custom_schema, SI())
    @test_throws ErrorException create_model_uwd(custom_schema, SEIR())
end

# Tests for NoCrossImmunity multistrain model UWD construction
@testitem "create_model_uwd(NoCrossImmunity) with auto-generated strain names" setup = [ModelUWDSetup] begin
    model = NoCrossImmunity(2)
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have 2 strain junctions (no Uninfected in multistrain UWD)
    @test length(junctions(uwd)) == 2

    # Should have 2 * 3 = 6 boxes (per strain: transmission, disease, reversion)
    @test length(boxes(uwd)) == 6

    # Check box names - transmission, disease, reversion for each strain
    box_names = [subpart(uwd, b, :name) for b in boxes(uwd)]
    @test count(==(:transmission), box_names) == 2
    @test count(==(:disease), box_names) == 2
    @test count(==(:reversion), box_names) == 2
end

@testitem "create_model_uwd(NoCrossImmunity) with custom strain names" setup = [ModelUWDSetup] begin
    model = NoCrossImmunity([:h1n1, :h3n2, :seasonal])
    uwd = create_model_uwd(one_pop_typing, model)

    # Should have 3 strain junctions (no Uninfected in multistrain UWD)
    @test length(junctions(uwd)) == 3

    # Should have 3 * 3 = 9 boxes
    @test length(boxes(uwd)) == 9

    # Check junction variables are the strain names
    junction_vars = [subpart(uwd, j, :variable) for j in junctions(uwd)]
    @test :h1n1 in junction_vars
    @test :h3n2 in junction_vars
    @test :seasonal in junction_vars
end

# Tests for CompleteCrossImmunity multistrain model UWD construction
@testitem "create_model_uwd(CompleteCrossImmunity) with auto-generated strain names" setup = [ModelUWDSetup] begin
    model = CompleteCrossImmunity(2)
    uwd = create_model_uwd(uninfected_infected_typing, model)

    # Should have 3 junctions: 1 shared susceptible + 2 strain-specific infected
    @test length(junctions(uwd)) == 3

    # Should have 2 * 3 = 6 boxes (per strain: transmission, disease, reversion)
    @test length(boxes(uwd)) == 6

    # Check junction variables: one :susceptible and two strain names
    junction_vars = [subpart(uwd, j, :variable) for j in junctions(uwd)]
    @test :susceptible in junction_vars
    @test :strain_1 in junction_vars
    @test :strain_2 in junction_vars

    # Check box names - transmission, disease, reversion for each strain
    box_names = [subpart(uwd, b, :name) for b in boxes(uwd)]
    @test count(==(:transmission), box_names) == 2
    @test count(==(:disease), box_names) == 2
    @test count(==(:reversion), box_names) == 2
end

@testitem "create_model_uwd(CompleteCrossImmunity) with custom strain names" setup = [ModelUWDSetup] begin
    model = CompleteCrossImmunity([:wild_type, :variant])
    uwd = create_model_uwd(uninfected_infected_typing, model)

    # Should have 3 junctions: 1 shared susceptible + 2 strain-specific infected
    @test length(junctions(uwd)) == 3

    # Should have 2 * 3 = 6 boxes
    @test length(boxes(uwd)) == 6

    # Check junction variables
    junction_vars = [subpart(uwd, j, :variable) for j in junctions(uwd)]
    @test :susceptible in junction_vars
    @test :wild_type in junction_vars
    @test :variant in junction_vars
end

# Tests for AgeStratification UWD construction
@testitem "create_model_uwd(AgeStratification) creates correct structure with OnePopulationTyping" setup = [ModelUWDSetup] begin
    age_strat = AgeStratification([:child, :adult])
    uwd = create_model_uwd(one_pop_typing, age_strat)

    # Should have 2 age group junctions (single junction per age group for OnePopulationTyping)
    @test length(junctions(uwd)) == 2

    # Should have n² + 3n boxes: 4 transmission + 2 disease + 2 reversion + 2 waning = 10
    @test length(boxes(uwd)) == 10

    # Check transmission boxes are present
    box_names = [subpart(uwd, b, :name) for b in boxes(uwd)]
    @test count(==(:transmission), box_names) == 4
    @test count(==(:disease), box_names) == 2
    @test count(==(:reversion), box_names) == 2
    @test count(==(:waning), box_names) == 2

    # Check junction variables are the age group names (each appears once)
    junction_vars = [subpart(uwd, j, :variable) for j in junctions(uwd)]
    @test :child in junction_vars
    @test :adult in junction_vars

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(AgeStratification) creates correct structure with UninfectedInfectedTyping" setup = [ModelUWDSetup] begin
    age_strat = AgeStratification([:child, :adult, :elderly])
    uwd = create_model_uwd(ui_typing, age_strat)

    # Should have 6 age group junctions (uninfected + infected for each of 3 age groups)
    @test length(junctions(uwd)) == 6

    # Should have n² + 3n boxes: 9 transmission + 3 disease + 3 reversion + 3 waning = 18
    @test length(boxes(uwd)) == 18

    # Check box type counts
    box_names = [subpart(uwd, b, :name) for b in boxes(uwd)]
    @test count(==(:transmission), box_names) == 9
    @test count(==(:disease), box_names) == 3
    @test count(==(:reversion), box_names) == 3

    # Check junction variables have unique names with _U and _I suffixes
    junction_vars = [subpart(uwd, j, :variable) for j in junctions(uwd)]
    @test :child_U in junction_vars
    @test :child_I in junction_vars
    @test :adult_U in junction_vars
    @test :adult_I in junction_vars
    @test :elderly_U in junction_vars
    @test :elderly_I in junction_vars

    @test uwd isa RelationDiagram
end

@testitem "create_model_uwd(AgeStratification) with single age group" setup = [ModelUWDSetup] begin
    age_strat = AgeStratification([:population])
    uwd = create_model_uwd(one_pop_typing, age_strat)

    # Should have 1 junction for single age group with OnePopulationTyping
    @test length(junctions(uwd)) == 1

    # Should have n² + 3n = 1 + 3 = 4 boxes
    @test length(boxes(uwd)) == 4

    # Check box types
    box_names = [subpart(uwd, b, :name) for b in boxes(uwd)]
    @test count(==(:transmission), box_names) == 1
    @test count(==(:disease), box_names) == 1

    @test uwd isa RelationDiagram
end
