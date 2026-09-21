@testsnippet HelperSetup begin
    using AlgebraicPetri
    using Catlab
    using AlgebraicEpiMech
    import AlgebraicEpiMech as aem

    # Test typings
    one_pop_typing = OnePopulationTyping()
    ui_typing = UninfectedInfectedTyping()
end

@testitem "create_relation_diagram" setup = [HelperSetup] begin
    @testset "OnePopulationTyping" begin
        model = SI(number_I_stages = 3)
        uwd = aem.create_relation_diagram(one_pop_typing, model)

        @test uwd isa RelationDiagram
        @test length(ports(uwd, outer = true)) == 4  # S + 3 I stages
    end

    @testset "UninfectedInfectedTyping" begin
        model = SEI(number_E_stages = 2, number_I_stages = 3)
        uwd = aem.create_relation_diagram(ui_typing, model)

        @test uwd isa RelationDiagram
        @test length(ports(uwd, outer = true)) == 6  # S + 2 E + 3 I
    end

    @testset "Uses model.number_of_states" begin
        model = SEIR(number_E_stages = 2, number_I_stages = 2)
        uwd = aem.create_relation_diagram(one_pop_typing, model)

        # SEIR: 2 (S,R) + 2 (E) + 2 (I) = 6 states
        @test length(ports(uwd, outer = true)) == model.number_of_states
    end
end

@testitem "set_S_junction!" setup = [HelperSetup] begin
    @testset "OnePopulationTyping" begin
        model = SI()
        uwd = aem.create_relation_diagram(one_pop_typing, model)

        S_junction = aem.set_S_junction!(uwd, one_pop_typing)

        @test S_junction isa Integer
        @test length(junctions(uwd)) == 1  # Only S junction added
        # Verify first outer port is connected to S junction
        @test junction(uwd, ports(uwd, outer = true)[1], outer = true) == S_junction
    end

    @testset "UninfectedInfectedTyping" begin
        model = SEI()
        uwd = aem.create_relation_diagram(ui_typing, model)

        S_junction = aem.set_S_junction!(uwd, ui_typing)

        @test S_junction isa Integer
        @test length(junctions(uwd)) == 1  # Only S junction added
        # Verify first outer port is connected to S junction
        @test junction(uwd, ports(uwd, outer = true)[1], outer = true) == S_junction
    end
end

@testitem "variable_name" setup = [HelperSetup] begin
    @testset "Single stage returns base name" begin
        @test aem.variable_name(:E, 1, 1) == :E
        @test aem.variable_name(:I, 1, 1) == :I
    end

    @testset "Multiple stages append stage numbers" begin
        @test aem.variable_name(:E, 1, 3) == :E1
        @test aem.variable_name(:E, 2, 3) == :E2
        @test aem.variable_name(:E, 3, 3) == :E3

        @test aem.variable_name(:I, 1, 4) == :I1
        @test aem.variable_name(:I, 2, 4) == :I2
        @test aem.variable_name(:I, 3, 4) == :I3
        @test aem.variable_name(:I, 4, 4) == :I4
    end

    @testset "Works with any symbol" begin
        @test aem.variable_name(:R, 1, 2) == :R1
        @test aem.variable_name(:Custom, 5, 10) == :Custom5
    end
end

@testitem "add_stages!" setup = [HelperSetup] begin
    @testset "Single stage" begin
        model = SI(number_I_stages = 1)
        uwd = aem.create_relation_diagram(one_pop_typing, model)
        aem.set_S_junction!(uwd, one_pop_typing)

        pop_type = aem.get_infected_type(one_pop_typing)
        # Counter starts at 2 (after S)
        I_junctions, new_counter = aem.add_stages!(uwd, :I, 1, pop_type, 2, one_pop_typing)

        @test length(I_junctions) == 1
        @test new_counter == 3  # Moved past 1 I stage
        # Single stage, no progression mechanisms added
        @test length(boxes(uwd)) == 0
    end

    @testset "Multiple stages with progression" begin
        model = SI(number_I_stages = 3)
        uwd = aem.create_relation_diagram(one_pop_typing, model)
        aem.set_S_junction!(uwd, one_pop_typing)

        pop_type = aem.get_infected_type(one_pop_typing)
        I_junctions, new_counter = aem.add_stages!(uwd, :I, 3, pop_type, 2, one_pop_typing)

        @test length(I_junctions) == 3
        @test new_counter == 5  # Moved past 3 I stages
        # 3 stages → 2 progression mechanisms (I1→I2, I2→I3)
        @test length(boxes(uwd)) == 2
    end

    @testset "Junctions connected to outer ports" begin
        model = SI(number_I_stages = 2)
        uwd = aem.create_relation_diagram(one_pop_typing, model)
        aem.set_S_junction!(uwd, one_pop_typing)

        pop_type = aem.get_infected_type(one_pop_typing)
        I_junctions, _ = aem.add_stages!(uwd, :I, 2, pop_type, 2, one_pop_typing)

        # Verify each junction is connected to sequential outer ports
        for (i, junction_id) in enumerate(I_junctions)
            port_idx = i + 1  # +1 because S is on port 1
            @test junction(uwd, ports(uwd, outer = true)[port_idx], outer = true) ==
                junction_id
        end
    end

    @testset "Works with UninfectedInfectedTyping" begin
        model = SEI(number_E_stages = 2, number_I_stages = 3)
        uwd = aem.create_relation_diagram(ui_typing, model)
        aem.set_S_junction!(uwd, ui_typing)

        pop_type = aem.get_infected_type(ui_typing)
        E_junctions, counter = aem.add_stages!(uwd, :E, 2, pop_type, 2, ui_typing)

        @test length(E_junctions) == 2
        @test counter == 4  # 2 (start) + 2 (E stages) = 4
        # 2 E stages → 1 progression mechanism
        @test length(boxes(uwd)) == 1

        # Add I stages after E
        I_junctions,
            final_counter = aem.add_stages!(uwd, :I, 3, pop_type, counter, ui_typing)

        @test length(I_junctions) == 3
        @test final_counter == 7  # 4 + 3 = 7
        # 1 (E progression) + 2 (I progressions) = 3 total
        @test length(boxes(uwd)) == 3
    end
end

@testitem "setup_basic! for SI" setup = [HelperSetup] begin
    @testset "Single-stage SI" begin
        model = SI()
        uwd = aem.create_relation_diagram(one_pop_typing, model)

        result_uwd, S_junction,
            last_I_junction = aem.setup_basic!(uwd, one_pop_typing, model)

        @test result_uwd === uwd  # Returns same UWD
        @test S_junction isa Integer
        @test last_I_junction isa Integer
        @test length(junctions(uwd)) == 2  # S and I
        # 1 transmission box (S + I → I + I)
        @test length(boxes(uwd)) == 1
    end

    @testset "Multi-stage SI" begin
        model = SI(number_I_stages = 3)
        uwd = aem.create_relation_diagram(one_pop_typing, model)

        result_uwd, S_junction,
            last_I_junction = aem.setup_basic!(uwd, one_pop_typing, model)

        @test length(junctions(uwd)) == 4  # S + I1 + I2 + I3
        # 3 transmission boxes (one per I stage) + 2 progressions (I1→I2, I2→I3)
        @test length(boxes(uwd)) == 5
    end

    @testset "Works with UninfectedInfectedTyping" begin
        model = SI(number_I_stages = 2)
        uwd = aem.create_relation_diagram(ui_typing, model)

        result_uwd, S_junction, last_I_junction = aem.setup_basic!(uwd, ui_typing, model)

        @test length(junctions(uwd)) == 3  # S + I1 + I2
        # 2 transmission boxes + 1 progression
        @test length(boxes(uwd)) == 3
    end
end

@testitem "setup_basic! for SEI" setup = [HelperSetup] begin
    @testset "Single-stage SEI" begin
        model = SEI()
        uwd = aem.create_relation_diagram(one_pop_typing, model)

        result_uwd, S_junction,
            last_I_junction = aem.setup_basic!(uwd, one_pop_typing, model)

        @test result_uwd === uwd
        @test S_junction isa Integer
        @test last_I_junction isa Integer
        @test length(junctions(uwd)) == 3  # S, E, I
        # 1 transmission (S + I → E + I) + 1 progression (E → I)
        @test length(boxes(uwd)) == 2
    end

    @testset "Multi-stage E and I" begin
        model = SEI(number_E_stages = 2, number_I_stages = 3)
        uwd = aem.create_relation_diagram(one_pop_typing, model)

        result_uwd, S_junction,
            last_I_junction = aem.setup_basic!(uwd, one_pop_typing, model)

        @test length(junctions(uwd)) == 6  # S + E1 + E2 + I1 + I2 + I3
        # 3 transmission boxes (one per I stage)
        # + 1 E progression (E1→E2)
        # + 2 I progressions (I1→I2, I2→I3)
        # + 1 E→I progression (E2→I1)
        @test length(boxes(uwd)) == 7
    end

    @testset "Different E and I stage counts" begin
        model = SEI(number_E_stages = 3, number_I_stages = 1)
        uwd = aem.create_relation_diagram(ui_typing, model)

        result_uwd, S_junction, last_I_junction = aem.setup_basic!(uwd, ui_typing, model)

        @test length(junctions(uwd)) == 5  # S + E1 + E2 + E3 + I
        # 1 transmission + 2 E progressions + 1 E→I progression
        @test length(boxes(uwd)) == 4
    end
end
