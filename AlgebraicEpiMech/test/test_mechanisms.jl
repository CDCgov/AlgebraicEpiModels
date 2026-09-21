@testsnippet MechanismSetup begin
    using AlgebraicPetri
    using Catlab
    using AlgebraicEpiMech

    # Helper function to create a basic UWD for testing mechanisms
    function create_test_uwd(schema, num_ports::Int)
        port_types = fill(
            schema isa OnePopulationSchema ? schema.population_type : schema.infected_type,
            num_ports
        )
        uwd = RelationDiagram(port_types)
        return uwd
    end

    # Helper to count boxes in a UWD
    count_boxes(uwd) = length(boxes(uwd))

    # Helper to count junctions in a UWD
    count_junctions(uwd) = length(junctions(uwd))

    # Create test schemas
    one_pop_schema = OnePopulationSchema()
    ui_schema = UninfectedInfectedSchema()
end

@testitem "Infection mechanism (direct to I)" setup = [MechanismSetup] begin
    # OnePopulationSchema: S + I → I + I
    @testset "OnePopulationSchema" begin
        uwd = create_test_uwd(one_pop_schema, 2)
        S_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :S)
        I_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :I)
        first_I_junction = add_junction!(
            uwd, one_pop_schema.population_type, variable = :I1
        )

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, first_I_junction, one_pop_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, I1
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end

    # UninfectedInfectedSchema: S + I → I + I
    @testset "UninfectedInfectedSchema" begin
        uwd = RelationDiagram([ui_schema.uninfected_type, ui_schema.infected_type])
        S_junction = add_junction!(uwd, ui_schema.uninfected_type, variable = :S)
        I_junction = add_junction!(uwd, ui_schema.infected_type, variable = :I)
        first_I_junction = add_junction!(uwd, ui_schema.infected_type, variable = :I1)

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, first_I_junction, ui_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, I1
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end
end

@testitem "Infection mechanism (exposure to E)" setup = [MechanismSetup] begin
    # OnePopulationSchema: S + I → E + I
    @testset "OnePopulationSchema" begin
        uwd = create_test_uwd(one_pop_schema, 3)
        S_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :S)
        I_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :I)
        E_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :E)

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, E_junction, one_pop_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, E
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end

    # UninfectedInfectedSchema: S + I → E + I
    @testset "UninfectedInfectedSchema" begin
        uwd = RelationDiagram(
            [
                ui_schema.uninfected_type, ui_schema.infected_type, ui_schema.infected_type,
            ]
        )
        S_junction = add_junction!(uwd, ui_schema.uninfected_type, variable = :S)
        I_junction = add_junction!(uwd, ui_schema.infected_type, variable = :I)
        E_junction = add_junction!(uwd, ui_schema.infected_type, variable = :E)

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, E_junction, ui_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, E
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end
end

@testitem "Disease progression mechanism" setup = [MechanismSetup] begin
    # OnePopulationSchema: X → Y
    @testset "OnePopulationSchema" begin
        uwd = create_test_uwd(one_pop_schema, 2)
        from_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :E)
        to_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :I)

        initial_boxes = count_boxes(uwd)
        add_disease_progression!(uwd, from_junction, to_junction, one_pop_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :disease
    end

    # UninfectedInfectedSchema: X → Y (infected type)
    @testset "UninfectedInfectedSchema" begin
        uwd = create_test_uwd(ui_schema, 2)
        from_junction = add_junction!(uwd, ui_schema.infected_type, variable = :E)
        to_junction = add_junction!(uwd, ui_schema.infected_type, variable = :I)

        initial_boxes = count_boxes(uwd)
        add_disease_progression!(uwd, from_junction, to_junction, ui_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :disease
    end
end

@testitem "Uninfected density progression mechanism" setup = [MechanismSetup] begin
    # OnePopulationSchema: X → Y
    @testset "OnePopulationSchema" begin
        uwd = create_test_uwd(one_pop_schema, 2)
        from_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :R)
        to_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_uninfected_density_progression!(uwd, from_junction, to_junction, one_pop_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :waning
    end

    # UninfectedInfectedSchema: X → Y (uninfected type)
    @testset "UninfectedInfectedSchema" begin
        uwd = RelationDiagram([ui_schema.uninfected_type, ui_schema.uninfected_type])
        from_junction = add_junction!(uwd, ui_schema.uninfected_type, variable = :R)
        to_junction = add_junction!(uwd, ui_schema.uninfected_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_uninfected_density_progression!(uwd, from_junction, to_junction, ui_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :waning
    end
end

@testitem "Reversion progression mechanism" setup = [MechanismSetup] begin
    # OnePopulationSchema: infected → uninfected
    @testset "OnePopulationSchema" begin
        uwd = create_test_uwd(one_pop_schema, 2)
        from_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :I)
        to_junction = add_junction!(uwd, one_pop_schema.population_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_reversion_progression!(uwd, from_junction, to_junction, one_pop_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :reversion
    end

    # UninfectedInfectedSchema: infected → uninfected (different types)
    @testset "UninfectedInfectedSchema" begin
        uwd = RelationDiagram([ui_schema.infected_type, ui_schema.uninfected_type])
        from_junction = add_junction!(uwd, ui_schema.infected_type, variable = :I)
        to_junction = add_junction!(uwd, ui_schema.uninfected_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_reversion_progression!(uwd, from_junction, to_junction, ui_schema)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :reversion
    end
end
