@testsnippet MechanismSetup begin
    using AlgebraicPetri
    using Catlab

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

@testitem "Strata movement primitive" setup = [MechanismSetup] begin
    using AlgebraicPetri.TypedPetri: oapply_typed

    for typing in (one_pop_typing, ui_typing)
        codomain = type_system(typing)
        for population in AlgebraicPetri.snames(codomain)
            uwd = RelationDiagram(Symbol[])
            young = add_junction!(uwd, population; variable = :young)
            old = add_junction!(uwd, population; variable = :old)
            linear = add_strata_transition!(uwd, young, old, typing)
            reflexive = add_strata_transition!(uwd, old, old, typing)
            @test [linear, reflexive] == collect(boxes(uwd))
            @test uwd[linear, :name] == uwd[reflexive, :name] ==
                AlgebraicEpiMech.population_box(AlgebraicEpiMech.Movement(), typing, population)
            @test junction.(Ref(uwd), ports(uwd, linear)) == [young, old]
            @test junction.(Ref(uwd), ports(uwd, reflexive)) == [old, old]

            pn = dom(oapply_typed(codomain, uwd, [:aging, :identity]))
            @test AlgebraicPetri.inputs(pn, linear) == [young]
            @test AlgebraicPetri.outputs(pn, linear) == [old]
            @test AlgebraicPetri.inputs(pn, reflexive) == AlgebraicPetri.outputs(pn, reflexive) == [old]
        end
    end

    uwd = RelationDiagram(Symbol[])
    u = add_junction!(uwd, ui_typing.uninfected_type; variable = :U)
    i = add_junction!(uwd, ui_typing.infected_type; variable = :I)
    foreign = add_junction!(uwd, :Other; variable = :other)
    @test_throws ArgumentError add_strata_transition!(uwd, u, i, ui_typing)
    @test_throws ArgumentError add_strata_transition!(uwd, foreign, foreign, ui_typing)
    @test isempty(boxes(uwd))  # invalid declarations leave no partial wiring
end

@testitem "Birth and death primitives" setup = [MechanismSetup] begin
    using AlgebraicPetri.TypedPetri: oapply_typed

    for typing in (one_pop_typing, ui_typing)
        codomain = type_system(typing)
        uwd = RelationDiagram(Symbol[])
        newborn = add_junction!(uwd, AlgebraicEpiMech.get_uninfected_type(typing); variable = :newborn)
        birth = add_birth!(uwd, newborn, typing)
        @test uwd[birth, :name] == :birth
        @test junction.(Ref(uwd), ports(uwd, birth)) == [newborn]
        leaving = Pair{Int, Int}[]
        for population in AlgebraicPetri.snames(codomain)
            old = add_junction!(uwd, population; variable = Symbol(:old_, population))
            death = add_death!(uwd, old, typing)
            @test uwd[death, :name] == AlgebraicEpiMech.population_box(AlgebraicEpiMech.Death(), typing, population)
            @test junction.(Ref(uwd), ports(uwd, death)) == [old]
            push!(leaving, death => old)
        end

        names = [:birth; [Symbol(:death, i) for i in eachindex(leaving)]]
        pn = dom(oapply_typed(codomain, uwd, names))
        @test isempty(AlgebraicPetri.inputs(pn, birth))
        @test AlgebraicPetri.outputs(pn, birth) == [newborn]
        for (death, old) in leaving
            @test AlgebraicPetri.inputs(pn, death) == [old]
            @test isempty(AlgebraicPetri.outputs(pn, death))
        end
    end

    uwd = RelationDiagram(Symbol[])
    i = add_junction!(uwd, ui_typing.infected_type; variable = :I)
    foreign = add_junction!(uwd, :Other; variable = :other)
    @test_throws ArgumentError add_birth!(uwd, i, ui_typing)
    @test_throws ArgumentError add_birth!(uwd, foreign, one_pop_typing)
    @test_throws ArgumentError add_death!(uwd, foreign, ui_typing)
    @test isempty(boxes(uwd))  # invalid declarations leave no partial wiring
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
