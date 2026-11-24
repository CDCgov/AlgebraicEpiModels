@testsnippet StratumTransitionSetup begin
    using AlgebraicPetri, Catlab, LabelledArrays
    using AlgebraicPetri.TypedPetri: typed_product
    using SeeToDee: Rk4
    const AEM = AlgebraicEpiMech

    age_graph = [:child => :adult]
    flatnames(pn) = flatten_symbols.(snames(pn))
    flattransitions(pn) = flatten_symbols.(tnames(pn))
    label_parts(s::Symbol) = [s]
    label_parts(s::Tuple) = vcat(label_parts.(s)...)

    function movement_indices(model, typing)
        strata = AEM.box_names(AEM.Movement(), typing)
        return [t for t in 1:nt(dom(model)) if tname(codom(model), model[:T](t)) in strata]
    end

    function unary_arc(pn, t)
        return (
            sname(pn, only(AlgebraicPetri.inputs(pn, t))),
            sname(pn, only(AlgebraicPetri.outputs(pn, t))),
        )
    end

    function incidence_signature(model, normalize)
        pn = dom(model)
        return sort!(
            [
                (
                    tname(codom(model), model[:T](t)),
                    sort!([normalize(sname(pn, s)) for s in AlgebraicPetri.inputs(pn, t)]),
                    sort!([normalize(sname(pn, s)) for s in AlgebraicPetri.outputs(pn, t)]),
                )
                    for t in 1:nt(pn)
            ]
        )
    end

    function solve_steps(pn, u0, rates; days = 30)
        f! = vectorfield_flat(pn)
        f(x, u, p, t) = f!(similar(x), x, p, t)
        step = Rk4(f, 1.0; supersample = 8)
        states = [u0]
        for t in 0:(days - 1)
            push!(states, step(last(states), nothing, rates, Float64(t)))
        end
        return states
    end

    passive(typing, spec) = (default_reflexives(typing, spec)..., :strata)
end

@testitem "Default construction is unchanged by movement types" setup = [StratumTransitionSetup] begin
    one = OnePopulationTyping()
    split = UninfectedInfectedTyping()
    @test tnames(dom(create_model(one, SIR()))) == [:transmission_S_I, :I_to_R]
    @test tnames(dom(create_model(split, SIR()))) == [:transmission_S_I, :I_to_R]
    @test tnames(dom(create_model(one, SEIRS(number_E_stages = 2, number_I_stages = 3)))) == [
        :E1_to_E2, :I1_to_I2, :I2_to_I3, :transmission_S_I1, :transmission_S_I2,
        :transmission_S_I3, :E2_to_I1, :I3_to_R, :R_to_S,
    ]
    @test tnames(dom(create_model(one, AgeStratification([:child, :adult])))) == [
        :child_child, :child_adult, :adult_child, :adult_adult,
        :child, :child, :child, :adult, :adult, :adult,
    ]
    @test tnames(dom(create_model(split, AgeStratification([:child, :adult])))) == [
        :child_U_child_I, :child_U_adult_I, :adult_U_child_I, :adult_U_adult_I,
        :child_I, :child_I, :child_U, :adult_I, :adult_I, :adult_U,
    ]
    @test tnames(dom(create_model(one, GeographicStratification([:ca, :ny])))) ==
        [:ca_ca, :ca_ny, :ny_ca, :ny_ny, :ca, :ca, :ca, :ny, :ny, :ny]
    # Waning reflexives come last, in immune-class order.
    @test tnames(dom(create_model(split, ImmuneHistory([:a, :b])))) == [
        :infect_a_naive_by_naive, :infect_a_naive_by_b, :a_from_naive, :a_from_naive,
        :infect_b_naive_by_naive, :infect_b_naive_by_a, :b_from_naive, :b_from_naive,
        :infect_b_a_by_naive, :infect_b_a_by_a, :b_from_a, :b_from_a,
        :infect_a_b_by_naive, :infect_a_b_by_b, :a_from_b, :a_from_b,
        :U_naive, :U_a, :U_b, :U_a_b,
    ]
    @test tnames(dom(create_model(split, ImmuneHistory([:a, :b]; mode = LatestInfection())))) == [
        :infect_a_naive_by_naive, :infect_a_naive_by_b, :a_from_naive, :a_from_naive,
        :infect_b_naive_by_naive, :infect_b_naive_by_a, :b_from_naive, :b_from_naive,
        :infect_b_a_by_naive, :infect_b_a_by_a, :b_from_a, :b_from_a,
        :infect_a_b_by_naive, :infect_a_b_by_b, :a_from_b, :a_from_b,
        :U_naive, :U_a, :U_b,
    ]
    @test tnames(dom(create_model(one, NoCrossImmunity([:x, :y])))) == [:x, :x, :x, :y, :y, :y]
    @test tnames(dom(create_model(split, CompleteCrossImmunity([:x, :y])))) ==
        [:x, :x, :x, :y, :y, :y]
    # A static stratification adds no movement, with or without a passive disease model.
    static = create_model(one, AgeStratification([:child, :adult]))
    for sir in (create_model(one, SIR()), create_model(one, SIR(); include_reflexives = (:strata,)))
        pn = dom(typed_product(sir, static))
        @test nt(pn) == 6
        @test isempty(movement_indices(typed_product(sir, static), one))
    end
end

@testitem "Stratum movement declarations" setup = [StratumTransitionSetup] begin
    @test isempty(AgeStratification(:child, :adult).transitions)
    @test AgeStratification("child", :adult; transitions = age_graph).transitions == age_graph
    @test GeographicStratification([:urban, :rural]; transitions = [:urban => :rural]).transitions ==
        [:urban => :rural]
    @test ContactStratification([:a, :b], :risk; transitions = [:a => :b, :b => :a]).transitions ==
        [:a => :b, :b => :a]
    @test_throws ArgumentError AgeStratification([:child, :adult]; transitions = [:child => :missing])
    @test_throws ArgumentError AgeStratification([:child, :adult]; transitions = [age_graph; age_graph])
    @test_throws ArgumentError AgeStratification([:child, :adult]; transitions = [:child => :child])
    @test_throws TypeError AgeStratification([:child, :adult]; transitions = ["child" => "adult"])
    @test_throws TypeError AgeStratification([:child, :adult]; transitions = Dict(:aging => age_graph))

    graph = copy(age_graph)
    names = [:child, :adult]
    age = AgeStratification(names; transitions = graph)
    empty!(graph)
    empty!(names)
    @test age.transitions == age_graph
    @test age.stratum_names == [:child, :adult]
end

@testitem "Ageing: exact construction and arcs" setup = [StratumTransitionSetup] begin
    typing = OnePopulationTyping()
    sir = create_model(typing, SIR(); include_reflexives = (:strata,))
    aging = create_model(typing, AgeStratification([:child, :adult]; transitions = age_graph))
    model = typed_product(sir, aging)
    pn = dom(model)
    @test Set(flatnames(pn)) == Set([:S_child, :I_child, :R_child, :S_adult, :I_adult, :R_adult])
    @test Set(flattransitions(pn)) == Set(
        [
            :transmission_S_I_child_child, :transmission_S_I_child_adult,
            :transmission_S_I_adult_child, :transmission_S_I_adult_adult,
            :I_to_R_child, :I_to_R_adult,
            :S_child_to_adult, :I_child_to_adult, :R_child_to_adult,
        ]
    )
    @test (ns(pn), nt(pn)) == (6, 9)
    @test Set(unary_arc(pn, t) for t in movement_indices(model, typing)) ==
        Set([((c, :child), (c, :adult)) for c in (:S, :I, :R)])
    for (c, name) in zip((:S, :I, :R), (:S_child_to_adult, :I_child_to_adult, :R_child_to_adult))
        t = only(findall(==(name), flattransitions(pn)))
        @test unary_arc(pn, t) == ((c, :child), (c, :adult))
    end

    # Without movement reflexives on the disease side, the pullback drops ageing.
    @test isempty(movement_indices(typed_product(create_model(typing, SIR()), aging), typing))

    # Disabling reflexives does not disable declared movement.
    bare_age = create_model(
        typing, AgeStratification(:child, :adult; transitions = age_graph);
        include_reflexives = false
    )
    @test nt(dom(bare_age)) == 5
    @test :child_to_adult in tnames(dom(bare_age))

    # The split typing moves each population through its own movement type.
    split = UninfectedInfectedTyping()
    split_model = typed_product(
        create_model(split, SIR(); include_reflexives = (:strata,)),
        create_model(split, AgeStratification([:child, :adult]; transitions = age_graph)),
    )
    split_pn = dom(split_model)
    @test Set(flattransitions(split_pn)) == Set(
        [
            :transmission_S_I_child_U_child_I, :transmission_S_I_child_U_adult_I,
            :transmission_S_I_adult_U_child_I, :transmission_S_I_adult_U_adult_I,
            :I_to_R_child_I, :I_to_R_adult_I,
            :S_child_U_to_adult_U, :I_child_I_to_adult_I, :R_child_I_to_adult_I,
        ]
    )
    moves = movement_indices(split_model, split)
    @test Set(tname(codom(split_model), split_model[:T](t)) for t in moves) == Set([:strata_U, :strata_I])
    @test length(moves) == 3
end

@testitem "Stratum movement preserves stages and other model factors" setup = [StratumTransitionSetup] begin
    for typing in (
            OnePopulationTyping(), UninfectedInfectedTyping(
                uninfected_type = :Susceptible, infected_type = :Infectious
            ),
        )
        disease = create_model(
            typing, SEIRS(number_E_stages = 2, number_I_stages = 3);
            include_reflexives = (:strata,)
        )
        age = create_model(typing, AgeStratification(:child, :adult; transitions = age_graph))
        core = typed_product(disease, age)
        @test length(movement_indices(core, typing)) == 7
        factors = typing isa OnePopulationTyping ?
            [GeographicStratification(:urban, :rural), NoCrossImmunity([:a, :b])] :
            [
                GeographicStratification(:urban, :rural), CompleteCrossImmunity([:a, :b]),
                ImmuneHistory([:a, :b]), ImmuneHistory([:a, :b]; mode = LatestInfection()),
            ]
        for spec in factors
            # Every other factor is passive to ageing: it lists :strata on top of its defaults.
            factor = create_model(typing, spec; include_reflexives = passive(typing, spec))
            model = typed_product(core, factor)
            pn = dom(model)
            moves = movement_indices(model, typing)
            @test length(moves) == ns(pn) ÷ 2
            @test length(unique(flattransitions(pn))) == nt(pn)
            for t in moves
                source, target = unary_arc(pn, t)
                s, d = label_parts(source), label_parts(target)
                @test s[1] == d[1]  # disease stage
                @test s[2] in (:child, :child_U, :child_I)
                @test d[2] in (:adult, :adult_U, :adult_I)
                @test replace(string(s[2]), "child" => "adult") == string(d[2])
                @test s[3:end] == d[3:end]  # geography, strain, or immune history
            end
            # Parenthesization must not change the transition incidence.
            regrouped = typed_product(disease, typed_product(age, factor))
            normalize(s) = Tuple(label_parts(s))
            @test incidence_signature(model, normalize) == incidence_signature(regrouped, normalize)
            # A factor that is not passive to movement drops it from the product.
            static_factor = create_model(typing, spec)
            @test isempty(movement_indices(typed_product(core, static_factor), typing))
        end
    end
end

@testitem "Stratum movement ODE supports unequal infant bands in declared order" setup = [StratumTransitionSetup] begin
    typing = OnePopulationTyping()
    # Ordering in the names vector does not create, reverse, or reorder graph edges.
    bands = AgeStratification(
        [:older, :m0_3, :m3_9];
        transitions = [:m0_3 => :m3_9, :m3_9 => :older]
    )
    model = typed_product(
        create_model(typing, SIR(); include_reflexives = (:strata,)),
        create_model(typing, bands),
    )
    pn = dom(model)
    rates = Dict(t => 0.0 for t in flattransitions(pn))
    a, b = 1 / 90, 1 / 180
    for compartment in (:S, :I, :R)
        rates[Symbol(compartment, "_m0_3_to_m3_9")] = a
        rates[Symbol(compartment, "_m3_9_to_older")] = b
    end
    u0 = LVector(
        S_m0_3 = 120.0, S_m3_9 = 30.0, S_older = 50.0,
        I_m0_3 = 0.0, I_m3_9 = 0.0, I_older = 0.0,
        R_m0_3 = 0.0, R_m3_9 = 0.0, R_older = 0.0
    )
    for (day, state) in enumerate(solve_steps(pn, u0, rates; days = 180))
        t = day - 1
        youngest = u0.S_m0_3 * exp(-a * t)
        middle = u0.S_m3_9 * exp(-b * t) + a * u0.S_m0_3 / (b - a) * (exp(-a * t) - exp(-b * t))
        @test state.S_m0_3 ≈ youngest rtol = 1.0e-10
        @test state.S_m3_9 ≈ middle rtol = 1.0e-10
        @test state.S_older ≈ sum(u0) - youngest - middle rtol = 1.0e-10
        @test sum(state) ≈ sum(u0)
        @test minimum(state) >= 0
    end
end

@testitem "Stratum movement graph products: sequential and precomposed factors agree" setup = [StratumTransitionSetup] begin
    for typing in (OnePopulationTyping(), UninfectedInfectedTyping())
        disease = create_model(typing, SIR(); include_reflexives = (:strata,))
        age = AgeStratification(:child, :adult; transitions = age_graph)
        geo = GeographicStratification(:urban, :rural)
        sequential = typed_product(
            typed_product(disease, create_model(typing, age)),
            create_model(typing, geo; include_reflexives = passive(typing, geo)),
        )
        combined = typed_product(disease, create_model(typing, age * geo))
        normalize(s) = Tuple(
            vcat(
                [
                    Symbol.(split(replace(string(p), r"_[UI]$" => ""), 'x')) for p in label_parts(s)
                ]...
            )
        )
        @test incidence_signature(sequential, normalize) == incidence_signature(combined, normalize)
        pn = dom(sequential)
        @test length(movement_indices(sequential, typing)) == 6
        @test length(movement_indices(combined, typing)) == 6
        for t in movement_indices(sequential, typing)
            source, target = label_parts.(unary_arc(pn, t))
            @test source[1] == target[1]  # disease state
            @test source[3] == target[3]  # location
        end
        @test length(unique(flattransitions(pn))) == nt(pn)
        # A static factor that is not passive to movement drops ageing from the sequential product.
        dropped = typed_product(typed_product(disease, create_model(typing, age)), create_model(typing, geo))
        @test isempty(movement_indices(dropped, typing))
    end

    # compose_stratifications multiplies graphs, with identities for a static factor.
    age = AgeStratification(:child, :adult; transitions = age_graph)
    static = GeographicStratification(:urban, :rural)
    moving = GeographicStratification(:urban, :rural; transitions = [:urban => :rural])
    @test isempty((static * GeographicStratification(:a, :b)).transitions)
    @test (age * static).transitions == [:childxurban => :adultxurban, :childxrural => :adultxrural]
    @test (static * age).transitions == [:urbanxchild => :urbanxadult, :ruralxchild => :ruralxadult]
    # One movement type: two moving factors move in the same event, as typed_product does.
    @test (age * moving).transitions == [:childxurban => :adultxrural]
    typing = OnePopulationTyping()
    make(s) = create_model(typing, s)
    normalize_labels(s) = Tuple(Symbol.(split(replace(string(flatten_symbols(s)), "_" => "x"), 'x')))
    @test incidence_signature(typed_product(make(age), make(moving)), normalize_labels) ==
        incidence_signature(make(age * moving), normalize_labels)
    @test ((age * static) * moving).transitions == (age * (static * moving)).transitions
end

@testitem "Stratum movement ODE solutions conserve population and disease state" setup = [StratumTransitionSetup] begin
    typing = OnePopulationTyping()
    model = typed_product(
        create_model(typing, SIR(); include_reflexives = (:strata,)),
        create_model(typing, AgeStratification(:child, :adult; transitions = age_graph)),
    )
    pn = dom(model)
    u0 = LVector(
        S_child = 100.0, I_child = 20.0, R_child = 30.0,
        S_adult = 40.0, I_adult = 5.0, R_adult = 10.0
    )
    aging_rate = 1 / 90
    rates = Dict(name => 0.0 for name in flattransitions(pn))
    for t in movement_indices(model, typing)
        rates[flatten_symbols(tname(pn, t))] = aging_rate
    end
    states = solve_steps(pn, u0, rates)
    for (day, state) in enumerate(states)
        @test sum(state) ≈ sum(u0)
        @test minimum(state) >= 0
        for compartment in (:S, :I, :R)
            child, adult = Symbol(compartment, :_child), Symbol(compartment, :_adult)
            expected = u0[child] * exp(-aging_rate * (day - 1))
            @test state[child] ≈ expected rtol = 1.0e-10
            @test state[adult] ≈ u0[adult] + u0[child] - expected rtol = 1.0e-10
        end
    end
    @test all(diff([s.S_child for s in states]) .< 0)
    @test all(diff([s.S_adult for s in states]) .> 0)

    for name in keys(rates)
        startswith(string(name), "transmission") && (rates[name] = 0.0005)
        startswith(string(name), "I_to_R") && (rates[name] = 0.2)
    end
    infected = solve_steps(pn, u0, rates)
    @test all(s -> all(isfinite, s) && minimum(s) >= 0 && isapprox(sum(s), sum(u0)), infected)
    @test last(infected).R_child + last(infected).R_adult > u0.R_child + u0.R_adult
end

@testitem "ImmuneHistory waning is a movement between immune classes" setup = [StratumTransitionSetup] begin
    typing = UninfectedInfectedTyping()
    @test !ImmuneHistory([:a]).waning
    @test ImmuneHistory(2; waning = true).waning
    @test AEM._waning_edges(ImmuneHistory([:a, :b]; waning = true), Symbol[]) ==
        [((:a,), ()), ((:b,), ()), ((:a, :b), (:b,)), ((:a, :b), (:a,))]
    @test AEM._waning_edges(ImmuneHistory([:a, :b]; mode = LatestInfection(), waning = true), Symbol[]) ==
        [((:a,), ()), ((:b,), ())]
    @test AEM._waning_edges(ImmuneHistory([:a]), [:waning]) == [((), ()), ((:a,), (:a,))]
    @test isempty(AEM._waning_edges(ImmuneHistory([:a]), Symbol[]))

    # The factor implements :waning as edges and carries no :waning reflexives.
    history_spec = ImmuneHistory([:a, :b]; waning = true)
    full = create_model_uwd(typing, history_spec)
    waning_boxes = [b for b in boxes(full) if full[b, :name] == :waning]
    @test length(waning_boxes) == 4
    @test all(length(unique(junction(full, p) for p in ports(full, b))) == 2 for b in waning_boxes)
    @test Set(generate_transition_names(full, history_spec)[waning_boxes]) ==
        Set([:wane_a_a, :wane_b_b, :wane_a_a_b, :wane_b_a_b])
    latest = create_model_uwd(typing, ImmuneHistory([:a, :b]; mode = LatestInfection(), waning = true))
    @test count(==(:waning), latest[:name]) == 2
    # Listing :waning on an implementing factor adds nothing.
    @test nt(dom(create_model(typing, history_spec))) ==
        nt(dom(create_model(typing, history_spec; include_reflexives = (:disease, :waning))))

    # The disease model must be passive to waning for the pullback to keep it.
    history = create_model(typing, history_spec)
    dropped = typed_product(create_model(typing, SIR()), history)
    @test !any(startswith(string(n), "waning") for n in flattransitions(dom(dropped)))
    model = typed_product(create_model(typing, SIR(); include_reflexives = :waning), history)
    pn = dom(model)
    @test Set(filter(n -> startswith(string(n), "waning_S"), flattransitions(pn))) ==
        Set([:waning_S_wane_a_a, :waning_S_wane_b_b, :waning_S_wane_a_a_b, :waning_S_wane_b_a_b])
    @test length(unique(flattransitions(pn))) == nt(pn)
    for t in 1:nt(pn)
        tname(codom(model), model[:T](t)) == :waning || continue
        source, target = label_parts.(unary_arc(pn, t))
        @test source[1] == target[1] == :S
    end

    # Waning alone: U_a_b decays at the sum of its two waning rates; the population is conserved.
    rates = Dict(n => 0.0 for n in flattransitions(pn))
    to_a, to_b = 0.01, 0.03
    rates[:waning_S_wane_b_a_b] = to_a  # forgetting b lands in U_a
    rates[:waning_S_wane_a_a_b] = to_b  # forgetting a lands in U_b
    u0 = LVector(; (n => 0.0 for n in flatnames(pn))...)
    u0.S_U_a_b = 100.0
    for (day, state) in enumerate(solve_steps(pn, u0, rates; days = 40))
        t = day - 1
        decay = exp(-(to_a + to_b) * t)
        @test state.S_U_a_b ≈ 100.0 * decay rtol = 1.0e-10
        @test state.S_U_a ≈ 100.0 * to_a / (to_a + to_b) * (1 - decay) rtol = 1.0e-9 atol = 1.0e-9
        @test state.S_U_b ≈ 100.0 * to_b / (to_a + to_b) * (1 - decay) rtol = 1.0e-9 atol = 1.0e-9
        @test sum(state) ≈ 100.0
        @test minimum(state) >= 0
    end

    # Ageing and waning are independent movements because they have distinct types.
    age = create_model(typing, AgeStratification([:child, :adult]; transitions = age_graph))
    joint = typed_product(
        typed_product(create_model(typing, SIR(); include_reflexives = (:strata, :waning)), age),
        create_model(typing, history_spec; include_reflexives = (:disease, :strata)),
    )
    joint_pn = dom(joint)
    joint_names = flattransitions(joint_pn)
    @test length(unique(joint_names)) == nt(joint_pn)
    @test length(movement_indices(joint, typing)) == 12  # S × 4 classes + (I, R) × 4 infected junctions
    @test count(n -> startswith(string(n), "waning_S"), joint_names) == 8  # 4 edges × 2 ages
    for t in movement_indices(joint, typing)
        source, target = label_parts.(unary_arc(joint_pn, t))
        @test source[1] == target[1]
        @test source[3] == target[3]  # immune history untouched by ageing
    end
    vectorfield_flat(joint_pn)
end
