@testsnippet MechanismSetup begin
    using AlgebraicPetri
    using Catlab
    using AlgebraicEpiMech

    # Helper function to create a basic UWD for testing mechanisms
    function create_test_uwd(typing, num_ports::Int)
        port_types = fill(
            typing isa OnePopulationTyping ? typing.population_type : typing.infected_type,
            num_ports
        )
        uwd = RelationDiagram(port_types)
        return uwd
    end

    # Helper to count boxes in a UWD
    count_boxes(uwd) = length(boxes(uwd))

    # Helper to count junctions in a UWD
    count_junctions(uwd) = length(junctions(uwd))

    # Create test typings
    one_pop_typing = OnePopulationTyping()
    ui_typing = UninfectedInfectedTyping()
end

@testitem "Infection mechanism (direct to I)" setup = [MechanismSetup] begin
    # OnePopulationTyping: S + I → I + I
    @testset "OnePopulationTyping" begin
        uwd = create_test_uwd(one_pop_typing, 2)
        S_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :S)
        I_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :I)
        first_I_junction = add_junction!(
            uwd, one_pop_typing.population_type, variable = :I1
        )

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, first_I_junction, one_pop_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, I1
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end

    # UninfectedInfectedTyping: S + I → I + I
    @testset "UninfectedInfectedTyping" begin
        uwd = RelationDiagram([ui_typing.uninfected_type, ui_typing.infected_type])
        S_junction = add_junction!(uwd, ui_typing.uninfected_type, variable = :S)
        I_junction = add_junction!(uwd, ui_typing.infected_type, variable = :I)
        first_I_junction = add_junction!(uwd, ui_typing.infected_type, variable = :I1)

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, first_I_junction, ui_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, I1
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end
end

@testitem "Infection mechanism (exposure to E)" setup = [MechanismSetup] begin
    # OnePopulationTyping: S + I → E + I
    @testset "OnePopulationTyping" begin
        uwd = create_test_uwd(one_pop_typing, 3)
        S_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :S)
        I_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :I)
        E_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :E)

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, E_junction, one_pop_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, E
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end

    # UninfectedInfectedTyping: S + I → E + I
    @testset "UninfectedInfectedTyping" begin
        uwd = RelationDiagram(
            [
                ui_typing.uninfected_type, ui_typing.infected_type, ui_typing.infected_type,
            ]
        )
        S_junction = add_junction!(uwd, ui_typing.uninfected_type, variable = :S)
        I_junction = add_junction!(uwd, ui_typing.infected_type, variable = :I)
        E_junction = add_junction!(uwd, ui_typing.infected_type, variable = :E)

        initial_boxes = count_boxes(uwd)
        add_infection!(uwd, S_junction, I_junction, E_junction, ui_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 3  # S, I, E
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :transmission
    end
end

@testitem "Disease progression mechanism" setup = [MechanismSetup] begin
    # OnePopulationTyping: X → Y
    @testset "OnePopulationTyping" begin
        uwd = create_test_uwd(one_pop_typing, 2)
        from_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :E)
        to_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :I)

        initial_boxes = count_boxes(uwd)
        add_disease_progression!(uwd, from_junction, to_junction, one_pop_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :disease
    end

    # UninfectedInfectedTyping: X → Y (infected type)
    @testset "UninfectedInfectedTyping" begin
        uwd = create_test_uwd(ui_typing, 2)
        from_junction = add_junction!(uwd, ui_typing.infected_type, variable = :E)
        to_junction = add_junction!(uwd, ui_typing.infected_type, variable = :I)

        initial_boxes = count_boxes(uwd)
        add_disease_progression!(uwd, from_junction, to_junction, ui_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :disease
    end
end

@testitem "Uninfected density progression mechanism" setup = [MechanismSetup] begin
    # OnePopulationTyping: X → Y
    @testset "OnePopulationTyping" begin
        uwd = create_test_uwd(one_pop_typing, 2)
        from_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :R)
        to_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_uninfected_density_progression!(uwd, from_junction, to_junction, one_pop_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :waning
    end

    # UninfectedInfectedTyping: X → Y (uninfected type)
    @testset "UninfectedInfectedTyping" begin
        uwd = RelationDiagram([ui_typing.uninfected_type, ui_typing.uninfected_type])
        from_junction = add_junction!(uwd, ui_typing.uninfected_type, variable = :R)
        to_junction = add_junction!(uwd, ui_typing.uninfected_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_uninfected_density_progression!(uwd, from_junction, to_junction, ui_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :waning
    end
end

@testitem "Reversion progression mechanism" setup = [MechanismSetup] begin
    # OnePopulationTyping: infected → uninfected
    @testset "OnePopulationTyping" begin
        uwd = create_test_uwd(one_pop_typing, 2)
        from_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :I)
        to_junction = add_junction!(uwd, one_pop_typing.population_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_reversion_progression!(uwd, from_junction, to_junction, one_pop_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :reversion
    end

    # UninfectedInfectedTyping: infected → uninfected (different types)
    @testset "UninfectedInfectedTyping" begin
        uwd = RelationDiagram([ui_typing.infected_type, ui_typing.uninfected_type])
        from_junction = add_junction!(uwd, ui_typing.infected_type, variable = :I)
        to_junction = add_junction!(uwd, ui_typing.uninfected_type, variable = :S)

        initial_boxes = count_boxes(uwd)
        add_reversion_progression!(uwd, from_junction, to_junction, ui_typing)

        @test count_boxes(uwd) == initial_boxes + 1
        @test count_junctions(uwd) == 2  # from and to
        # Verify box name
        @test subpart(uwd, boxes(uwd)[end], :name) == :reversion
    end
end
