@testsnippet HelperSetup begin
    using AlgebraicPetri
    using Catlab
    using AlgebraicEpiMech
    import AlgebraicEpiMech as aem

    # Test typings
    one_pop_typing = OnePopulationTyping()
    ui_typing = UninfectedInfectedTyping()
end

@testitem "set_S_junction!" setup = [HelperSetup] begin
    @testset "OnePopulationTyping" begin
        uwd = RelationDiagram(Symbol[])

        S_junction = aem.set_S_junction!(uwd, one_pop_typing)

        @test S_junction isa Integer
        @test length(junctions(uwd)) == 1  # Only S junction added
        @test subpart(uwd, S_junction, :variable) == :S
        @test isempty(ports(uwd, outer = true))
    end

    @testset "UninfectedInfectedTyping" begin
        uwd = RelationDiagram(Symbol[])

        S_junction = aem.set_S_junction!(uwd, ui_typing)

        @test S_junction isa Integer
        @test length(junctions(uwd)) == 1  # Only S junction added
        @test subpart(uwd, S_junction, :variable) == :S
        @test isempty(ports(uwd, outer = true))
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
        uwd = RelationDiagram(Symbol[])
        aem.set_S_junction!(uwd, one_pop_typing)

        pop_type = aem.get_infected_type(one_pop_typing)
        I_junctions = aem.add_stages!(uwd, :I, 1, pop_type, one_pop_typing)

        @test length(I_junctions) == 1
        # Single stage, no progression mechanisms added
        @test length(boxes(uwd)) == 0
    end

    @testset "Multiple stages with progression" begin
        uwd = RelationDiagram(Symbol[])
        aem.set_S_junction!(uwd, one_pop_typing)

        pop_type = aem.get_infected_type(one_pop_typing)
        I_junctions = aem.add_stages!(uwd, :I, 3, pop_type, one_pop_typing)

        @test length(I_junctions) == 3
        # 3 stages → 2 progression mechanisms (I1→I2, I2→I3)
        @test length(boxes(uwd)) == 2
        @test isempty(ports(uwd, outer = true))
    end

    @testset "Works with UninfectedInfectedTyping" begin
        uwd = RelationDiagram(Symbol[])
        aem.set_S_junction!(uwd, ui_typing)

        pop_type = aem.get_infected_type(ui_typing)
        E_junctions = aem.add_stages!(uwd, :E, 2, pop_type, ui_typing)

        @test length(E_junctions) == 2
        # 2 E stages → 1 progression mechanism
        @test length(boxes(uwd)) == 1

        # Add I stages after E
        I_junctions = aem.add_stages!(uwd, :I, 3, pop_type, ui_typing)

        @test length(I_junctions) == 3
        # 1 (E progression) + 2 (I progressions) = 3 total
        @test length(boxes(uwd)) == 3
        @test isempty(ports(uwd, outer = true))
    end

    @testset "Rejects non-positive stage counts before mutating the UWD" begin
        for count in (0, -1)
            uwd = RelationDiagram(Symbol[])
            pop_type = aem.get_infected_type(one_pop_typing)
            @test_throws ArgumentError aem.add_stages!(
                uwd, :I, count, pop_type, one_pop_typing
            )
            @test isempty(junctions(uwd))
            @test isempty(boxes(uwd))
        end
    end
end

@testitem "setup_basic! for SI" setup = [HelperSetup] begin
    @testset "Single-stage SI" begin
        model = SI()
        uwd = RelationDiagram(Symbol[])

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
        uwd = RelationDiagram(Symbol[])

        result_uwd, S_junction,
            last_I_junction = aem.setup_basic!(uwd, one_pop_typing, model)

        @test length(junctions(uwd)) == 4  # S + I1 + I2 + I3
        # 3 transmission boxes (one per I stage) + 2 progressions (I1→I2, I2→I3)
        @test length(boxes(uwd)) == 5
    end

    @testset "Works with UninfectedInfectedTyping" begin
        model = SI(number_I_stages = 2)
        uwd = RelationDiagram(Symbol[])

        result_uwd, S_junction, last_I_junction = aem.setup_basic!(uwd, ui_typing, model)

        @test length(junctions(uwd)) == 3  # S + I1 + I2
        # 2 transmission boxes + 1 progression
        @test length(boxes(uwd)) == 3
    end
end

@testitem "setup_basic! for SEI" setup = [HelperSetup] begin
    @testset "Single-stage SEI" begin
        model = SEI()
        uwd = RelationDiagram(Symbol[])

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
        uwd = RelationDiagram(Symbol[])

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
        uwd = RelationDiagram(Symbol[])

        result_uwd, S_junction, last_I_junction = aem.setup_basic!(uwd, ui_typing, model)

        @test length(junctions(uwd)) == 5  # S + E1 + E2 + E3 + I
        # 1 transmission + 2 E progressions + 1 E→I progression
        @test length(boxes(uwd)) == 4
    end
end
