@testsnippet ImportationSetup begin
    using AlgebraicPetri, Catlab, LabelledArrays
    using AlgebraicPetri.TypedPetri: typed_product
    using SeeToDee: Rk4
    const AEM = AlgebraicEpiMech

    ui_typing = UninfectedInfectedTyping()
    flatnames(pn) = flatten_symbols.(snames(pn))
    flattransitions(pn) = flatten_symbols.(tnames(pn))

    typed_indices(model, type) =
        [t for t in 1:nt(dom(model)) if tname(codom(model), model[:T](t)) == type]
    function arcs(model, type)
        pn = dom(model)
        return [
            (
                    flatten_symbols(tname(pn, t)),
                    flatten_symbols(sname(pn, only(AlgebraicPetri.inputs(pn, t)))),
                    flatten_symbols(sname(pn, only(AlgebraicPetri.outputs(pn, t)))),
                ) for t in typed_indices(model, type)
        ]
    end

    with_importation(typing, spec) = (default_reflexives(typing, spec)..., :importation)

    function history_model(strains; importation = true, disease_importation = importation)
        history = ImmuneHistory(strains)
        reflexives = importation ? with_importation(ui_typing, history) : true
        return typed_product(
            create_model(ui_typing, SEIRS(importation = disease_importation)),
            create_model(ui_typing, history; include_reflexives = reflexives),
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
end

@testitem "Type systems carry importation from uninfected to infected" setup = [ImportationSetup] begin
    one = OnePopulationTyping(population_type = :Person)
    split_named = UninfectedInfectedTyping(uninfected_type = :U, infected_type = :J)
    function signature(system, type)
        t = findfirst(==(type), tnames(system))
        return (
            snames(system)[AlgebraicPetri.inputs(system, t)],
            snames(system)[AlgebraicPetri.outputs(system, t)],
        )
    end
    @test signature(type_system(one), :importation) == ([:Person], [:Person])
    @test signature(type_system(split_named), :importation) == ([:U], [:J])
end

@testitem "importation is declared on compartmental specifications, off by default" setup = [ImportationSetup] begin
    for spec in (SI, SEI, SIR, SEIR, SIS, SEIS, SEIRS)
        @test !spec().demography.importation
        @test spec(importation = true).demography.importation
        @test spec(importation = true) == spec(importation = true)
        @test spec() != spec(importation = true)
    end
    # Stratification factors are passive for importation instead.
    @test_throws ArgumentError AgeStratification([:child, :adult]; importation = true)
    @test_throws MethodError ImmuneHistory([:a, :b]; importation = true)
end

@testitem "add_importation! joins an uninfected junction to an infected one" setup = [ImportationSetup] begin
    uwd = RelationDiagram(Symbol[])
    S = add_junction!(uwd, :Uninfected, variable = :S)
    I = add_junction!(uwd, :Infected, variable = :I)
    box = add_importation!(uwd, S, I, ui_typing)
    @test uwd[box, :name] == :importation
    @test [junction(uwd, p) for p in ports(uwd, box)] == [S, I]
    @test_throws ArgumentError add_importation!(uwd, I, S, ui_typing)
    @test_throws ArgumentError add_importation!(uwd, S, S, ui_typing)
end

@testitem "a compartmental model imports from S into its first infected stage" setup = [ImportationSetup] begin
    for typing in (OnePopulationTyping(), ui_typing)
        cases = (
            SIR(importation = true) => (:importation_S_to_I, :S, :I),
            SEIR(importation = true) => (:importation_S_to_E, :S, :E),
            SEIRS(number_E_stages = 2, importation = true) => (:importation_S_to_E1, :S, :E1),
            SI(number_I_stages = 3, importation = true) => (:importation_S_to_I1, :S, :I1),
        )
        for (spec, arc) in cases
            @test arcs(create_model(typing, spec), :importation) == [arc]
        end
        @test isempty(typed_indices(create_model(typing, SEIRS()), :importation))
    end
    # Passive importation on disease states would pair with a stratum's importation into `S → S`.
    for typing in (OnePopulationTyping(), ui_typing)
        @test_throws ArgumentError create_model(typing, SEIRS(); include_reflexives = :importation)
        @test nt(dom(create_model(typing, SIR(importation = true); include_reflexives = :importation))) ==
            nt(dom(create_model(typing, SIR(importation = true))))
    end
    # A one-population strain factor keeps importation on each strain junction.
    strains = create_model(OnePopulationTyping(), NoCrossImmunity([:x, :y]); include_reflexives = :importation)
    @test count(==(:importation), [tname(codom(strains), strains[:T](t)) for t in 1:nt(dom(strains))]) == 2
end

@testitem "immune history imports one route per susceptible class and strain" setup = [ImportationSetup] begin
    @test Set(arcs(history_model([:a, :b]), :importation)) == Set(
        [
            (:importation_S_to_E_import_a_naive, :S_U_naive, :E_a_from_naive),
            (:importation_S_to_E_import_b_naive, :S_U_naive, :E_b_from_naive),
            (:importation_S_to_E_import_a_b, :S_U_b, :E_a_from_b),
            (:importation_S_to_E_import_b_a, :S_U_a, :E_b_from_a),
        ]
    )
    for mode in (FullHistory(), LatestInfection())
        history = ImmuneHistory([:H1, :H3, :B]; mode = mode)
        model = typed_product(
            create_model(ui_typing, SEIRS(importation = true)),
            create_model(ui_typing, history; include_reflexives = with_importation(ui_typing, history)),
        )
        routes = length(AEM._infected_pairs(mode, [:H1, :H3, :B]))
        @test length(typed_indices(model, :importation)) == routes
    end

    @test isempty(typed_indices(history_model([:a, :b]; disease_importation = false), :importation))
    @test isempty(typed_indices(history_model([:a, :b]; importation = false, disease_importation = true), :importation))
end

@testitem "flu skeleton: 60 importation routes, none into an immune class" setup = [ImportationSetup] begin
    strains = [:H1, :H3, :B]
    ages = [Symbol(:age, k) for k in 1:5]
    age_spec = AgeStratification(ages)
    history = ImmuneHistory(strains; waning = true)
    core = typed_product(
        create_model(ui_typing, SEIRS(importation = true); include_reflexives = :waning),
        create_model(ui_typing, history; include_reflexives = with_importation(ui_typing, history)),
    )
    model = typed_product(
        core, create_model(ui_typing, age_spec; include_reflexives = with_importation(ui_typing, age_spec)),
    )
    pn = dom(model)
    @test ns(pn) == 44 * 5
    @test length(unique(flattransitions(pn))) == nt(pn)

    found = arcs(model, :importation)
    @test length(found) == 60
    for strain in strains
        @test count(a -> startswith(string(a[1]), "importation_S_to_E_import_$(strain)_"), found) == 20
    end
    for t in typed_indices(model, :importation)
        source = sname(pn, only(AlgebraicPetri.inputs(pn, t)))
        target = sname(pn, only(AlgebraicPetri.outputs(pn, t)))
        @test source[1][1] == :S && target[1][1] == :E
        strain, class = split(string(target[1][2]), "_from_")
        @test string(source[1][2]) == "U_" * class
        @test !(strain in split(class, "_"))
        @test chopsuffix(string(source[2]), "_U") == chopsuffix(string(target[2]), "_I")
    end

    # Without the age factor's opt-in the pullback drops importation.
    dropped = typed_product(core, create_model(ui_typing, age_spec))
    @test isempty(typed_indices(dropped, :importation))
    @test nt(dom(dropped)) == nt(pn) - 60
end

@testitem "importation alone: each class is drained at the summed rate of its routes" setup = [ImportationSetup] begin
    pn = dom(history_model([:a, :b]))
    rate_a, rate_b = 0.01, 0.03
    rates = Dict(n => 0.0 for n in flattransitions(pn))
    rates[:importation_S_to_E_import_a_naive] = rate_a
    rates[:importation_S_to_E_import_b_naive] = rate_b
    rates[:importation_S_to_E_import_a_b] = rate_a
    rates[:importation_S_to_E_import_b_a] = rate_b

    u0 = LVector(; (n => 0.0 for n in flatnames(pn))...)
    u0.S_U_naive = 100.0
    u0.S_U_a = 40.0
    u0.S_U_b = 60.0
    for (day, state) in enumerate(solve_steps(pn, u0, rates; days = 60))
        t = day - 1
        @test state.S_U_naive ≈ 100.0 * exp(-(rate_a + rate_b) * t) rtol = 1.0e-10
        @test state.S_U_a ≈ 40.0 * exp(-rate_b * t) rtol = 1.0e-10
        @test state.S_U_b ≈ 60.0 * exp(-rate_a * t) rtol = 1.0e-10
        @test state.E_a_from_naive ≈ 100.0 * rate_a / (rate_a + rate_b) * (1 - exp(-(rate_a + rate_b) * t)) rtol = 1.0e-9 atol = 1.0e-12
        @test state.S_U_a_b == 0.0
        @test sum(state) ≈ 200.0
        @test minimum(state) >= 0
    end
end

@testitem "contact strata are passive for importation when they list it" setup = [ImportationSetup] begin
    for typing in (OnePopulationTyping(), ui_typing)
        spec = AgeStratification([:child, :adult])
        model = typed_product(
            create_model(typing, SIR(importation = true)),
            create_model(typing, spec; include_reflexives = with_importation(typing, spec)),
        )
        one = typing isa OnePopulationTyping
        suffix(stratum) = one ? stratum : Symbol(stratum, "_U")
        target(stratum) = one ? Symbol(:I_, stratum) : Symbol(:I_, stratum, "_I")
        @test Set(arcs(model, :importation)) == Set(
            [(Symbol(:importation_S_to_I_, suffix(s)), Symbol(:S_, suffix(s)), target(s)) for s in (:child, :adult)]
        )
        # The default list leaves the factor's own net unchanged.
        @test isempty(typed_indices(create_model(typing, spec), :importation))
    end
end
