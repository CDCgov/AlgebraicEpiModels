@testsnippet BirthDeathSetup begin
    using AlgebraicPetri, Catlab, LabelledArrays
    using AlgebraicPetri.TypedPetri: typed_product
    using SeeToDee: Rk4
    const AEM = AlgebraicEpiMech

    flatnames(pn) = flatten_symbols.(snames(pn))
    flattransitions(pn) = flatten_symbols.(tnames(pn))
    label_parts(s::Symbol) = [s]
    label_parts(s::Tuple) = vcat(label_parts.(s)...)

    demography() = AgeStratification(
        [:child, :adult]; transitions = [:child => :adult], births = [:child], deaths = [:adult]
    )

    function typed_indices(model, types)
        return [t for t in 1:nt(dom(model)) if tname(codom(model), model[:T](t)) in types]
    end
    births(model) = typed_indices(model, (:birth,))
    deaths(model, typing) = typed_indices(model, AEM.box_names(AEM.Death(), typing))

    function arc(model, name)
        pn = dom(model)
        t = only(findall(==(name), flattransitions(pn)))
        return (
            tname(codom(model), model[:T](t)),
            [flatten_symbols(sname(pn, s)) for s in AlgebraicPetri.inputs(pn, t)],
            [flatten_symbols(sname(pn, s)) for s in AlgebraicPetri.outputs(pn, t)],
        )
    end

    function incidence_signature(model, normalize)
        pn = dom(model)
        return sort!(
            [
                (
                        tname(codom(model), model[:T](t)),
                        sort!(Any[normalize(sname(pn, s)) for s in AlgebraicPetri.inputs(pn, t)]),
                        sort!(Any[normalize(sname(pn, s)) for s in AlgebraicPetri.outputs(pn, t)]),
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

    passive(typing, spec) = (default_reflexives(typing, spec)..., :strata, :death)
end

@testitem "Birth and death declarations on the model specifications" setup = [BirthDeathSetup] begin
    births_of(spec) = spec.demography.births
    deaths_of(spec) = spec.demography.deaths
    @test isempty(births_of(AgeStratification(:child, :adult)))
    @test isempty(deaths_of(AgeStratification(:child, :adult)))
    spec = demography()
    @test births_of(spec) == (:child,)
    @test deaths_of(spec) == (:adult,)
    @test births_of(GeographicStratification([:a, :b]; births = [:b, :a])) == (:b, :a)
    @test deaths_of(ContactStratification([:a, :b], :risk; deaths = [:a])) == (:a,)
    @test births_of(AgeStratification(:child, :adult; births = (:child,))) == (:child,)
    @test_throws ArgumentError AgeStratification([:child, :adult]; births = [:infant])
    @test_throws ArgumentError AgeStratification([:child, :adult]; deaths = [:elderly])
    @test_throws ArgumentError AgeStratification([:child, :adult]; births = [:child, :child])
    @test_throws TypeError AgeStratification([:child, :adult]; deaths = ["adult"])
    @test_throws TypeError AgeStratification([:child, :adult]; deaths = :adult)

    declared = [:child]
    age = AgeStratification([:child, :adult]; births = declared)
    empty!(declared)
    @test births_of(age) == (:child,)

    @test SIR().demography == Demography()
    @test births_of(SIR(births = [:S])) == (:S,)
    @test deaths_of(SIR(deaths = [:I])) == (:I,)
    @test births_of(SEIRS(number_E_stages = 2, births = (:S, :E1))) == (:S, :E1)
    @test SIR() == SIR()
    @test SIR(births = [:S]) == SIR(births = (:S,))
    @test SIR(births = [:S]) != SIR()
    @test hash(SEIR(births = [:S])) == hash(SEIR(births = [:S]))
    for model in (SI, SEI, SIR, SEIR, SIS, SEIS, SEIRS)
        @test births_of(model(births = [:S])) == (:S,)
        @test isempty(births_of(model()))
    end
    @test_throws TypeError SIR(births = ["S"])
    @test_throws TypeError SIR(births = :S)
    @test_throws ArgumentError SIR(births = [:S, :S])
    @test_throws MethodError SIR(birth = [:S])
end

@testitem "Default construction is unchanged by birth and death types" setup = [BirthDeathSetup] begin
    one = OnePopulationTyping()
    split = UninfectedInfectedTyping()
    for typing in (one, split)
        specs = Any[
            SIR(), SEIRS(number_E_stages = 2, number_I_stages = 3),
            AgeStratification([:child, :adult]),
            AgeStratification([:child, :adult]; transitions = [:child => :adult]),
            GeographicStratification([:ca, :ny]),
        ]
        typing isa UninfectedInfectedTyping && push!(specs, ImmuneHistory([:a, :b]))
        for spec in specs
            model = create_model(typing, spec)
            @test isempty(births(model))
            @test isempty(deaths(model, typing))
        end
    end
    @test tnames(dom(create_model(one, SIR(births = ())))) == [:transmission_S_I, :I_to_R]
    @test tnames(dom(create_model(one, AgeStratification([:child, :adult]; births = Symbol[], deaths = Symbol[])))) ==
        tnames(dom(create_model(one, AgeStratification([:child, :adult]))))
    static = create_model(one, AgeStratification([:child, :adult]))
    for sir in (create_model(one, SIR()), create_model(one, SIR(); include_reflexives = (:death,)))
        product = typed_product(sir, static)
        @test nt(dom(product)) == 6
        @test isempty(deaths(product, one))
    end
end

@testitem "Births and deaths exact construction and arcs" setup = [BirthDeathSetup] begin
    typing = OnePopulationTyping()
    sir = create_model(typing, SIR(births = [:S]); include_reflexives = (:strata, :death))
    @test tnames(dom(sir)) == [
        :transmission_S_I, :I_to_R, :birth_S, :S, :I, :R, :death_S, :death_I, :death_R,
    ]
    age = create_model(typing, demography())
    @test tnames(dom(age)) == [
        :child_child, :child_adult, :adult_child, :adult_adult,
        :child, :child, :child, :adult, :adult, :adult,
        :child_to_adult, :child, :adult,
    ]
    model = typed_product(sir, age)
    pn = dom(model)
    @test Set(flatnames(pn)) == Set([:S_child, :I_child, :R_child, :S_adult, :I_adult, :R_adult])
    @test Set(flattransitions(pn)) == Set(
        [
            :transmission_S_I_child_child, :transmission_S_I_child_adult,
            :transmission_S_I_adult_child, :transmission_S_I_adult_adult,
            :I_to_R_child, :I_to_R_adult,
            :S_child_to_adult, :I_child_to_adult, :R_child_to_adult,
            :birth_S_child, :death_S_adult, :death_I_adult, :death_R_adult,
        ]
    )
    @test (ns(pn), nt(pn)) == (6, 13)
    @test length(unique(flattransitions(pn))) == nt(pn)
    @test arc(model, :birth_S_child) == (:birth, Symbol[], [:S_child])
    for compartment in (:S, :I, :R)
        @test arc(model, Symbol("death_", compartment, "_adult")) ==
            (:death, [Symbol(compartment, "_adult")], Symbol[])
    end
    @test length(births(model)) == 1
    @test length(deaths(model, typing)) == 3

    open_sir = create_model(typing, SIR(births = [:S]); include_reflexives = (:death,))
    @test tnames(dom(open_sir)) ==
        [:transmission_S_I, :I_to_R, :birth_S, :death_S, :death_I, :death_R]
    @test arc(open_sir, :birth_S) == (:birth, Symbol[], [:S])
    @test arc(open_sir, :death_I) == (:death, [:I], Symbol[])

    bare_age = create_model(typing, demography(); include_reflexives = false)
    @test tnames(dom(bare_age)) == [
        :child_child, :child_adult, :adult_child, :adult_adult, :child_to_adult, :child, :adult,
    ]

    split = UninfectedInfectedTyping()
    split_sir = create_model(split, SIR(births = [:S]); include_reflexives = (:strata, :death))
    @test tnames(dom(split_sir)) == [
        :transmission_S_I, :I_to_R, :birth_S, :S, :I, :R, :death_S, :death_I, :death_R,
    ]
    split_model = typed_product(split_sir, create_model(split, demography()))
    split_pn = dom(split_model)
    @test (ns(split_pn), nt(split_pn)) == (6, 13)
    @test arc(split_model, :birth_S_child_U) == (:birth, Symbol[], [:S_child_U])
    @test arc(split_model, :death_S_adult_U) == (:death_U, [:S_adult_U], Symbol[])
    @test arc(split_model, :death_I_adult_I) == (:death_I, [:I_adult_I], Symbol[])
    @test arc(split_model, :death_R_adult_I) == (:death_I, [:R_adult_I], Symbol[])
    @test length(births(split_model)) == 1
    @test length(deaths(split_model, split)) == 3
end

@testitem "Birth placement is validated against the compartments" setup = [BirthDeathSetup] begin
    one = OnePopulationTyping()
    split = UninfectedInfectedTyping()
    @test_throws ArgumentError create_model(one, SIR(births = [:X]))
    @test_throws ArgumentError create_model(one, SIR(births = [:E]))
    staged = create_model(one, SEIRS(number_E_stages = 2, births = [:S, :E1]))
    @test :birth_S in tnames(dom(staged))
    @test :birth_E1 in tnames(dom(staged))
    @test_throws ArgumentError create_model(one, SEIRS(number_E_stages = 2, births = [:E]))
    @test :birth_S in tnames(dom(create_model(split, SEIR(births = [:S]))))
    @test_throws ArgumentError create_model(split, SEIR(births = [:E]))
    @test_throws ArgumentError create_model(split, SIR(births = [:R]))
    for model in (SI, SEI, SIR, SEIR, SIS, SEIS, SEIRS), typing in (one, split)
        built = create_model(typing, model(births = [:S]))
        @test arc(built, :birth_S) == (:birth, Symbol[], [:S])
        @test nt(dom(built)) == nt(dom(create_model(typing, model()))) + 1
    end
end

@testitem "Deaths declared on a compartmental model leave only those compartments" setup = [BirthDeathSetup] begin
    one = OnePopulationTyping()
    split = UninfectedInfectedTyping()
    @test tnames(dom(create_model(one, SIR(deaths = [:I])))) == [:transmission_S_I, :I_to_R, :death_I]
    @test arc(create_model(split, SIR(deaths = [:I])), :death_I) == (:death_I, [:I], Symbol[])
    @test_throws ArgumentError create_model(one, SIR(deaths = [:E]))

    # Disease-induced mortality among adults only: the product keeps the one shared death.
    model = typed_product(
        create_model(one, SIR(deaths = [:I]); include_reflexives = :strata),
        create_model(one, demography()),
    )
    @test deaths(model, one) == [only(findall(==(:death_I_adult), flattransitions(dom(model))))]
    @test arc(model, :death_I_adult) == (:death, [:I_adult], Symbol[])
end

@testitem "Death can be passive; birth has to be placed" setup = [BirthDeathSetup] begin
    one = OnePopulationTyping()
    split = UninfectedInfectedTyping()
    static = AgeStratification([:child, :adult])

    @test tnames(dom(create_model(one, SIR(); include_reflexives = (:death,)))) ==
        [:transmission_S_I, :I_to_R, :death_S, :death_I, :death_R]
    @test tnames(dom(create_model(one, static; include_reflexives = (:death,)))) ==
        [:child_child, :child_adult, :adult_child, :adult_adult, :child, :adult]
    split_static = create_model(split, static; include_reflexives = (:death,))
    @test tnames(dom(split_static)) == [
        :child_U_child_I, :child_U_adult_I, :adult_U_child_I, :adult_U_adult_I,
        :child_U, :adult_U, :child_I, :adult_I,
    ]
    @test length(deaths(split_static, split)) == 4
    @test tnames(dom(create_model(split, SIR(); include_reflexives = (:death_I,)))) ==
        [:transmission_S_I, :I_to_R, :death_I, :death_R]
    declared = demography()
    @test nt(dom(create_model(one, declared; include_reflexives = passive(one, declared)))) ==
        nt(dom(create_model(one, declared))) == 13

    # A newborn has to be placed in every factor, so birth cannot be listed.
    @test_throws ArgumentError create_model(one, SIR(); include_reflexives = (:birth,))
    @test_throws ArgumentError create_model(one, static; include_reflexives = (:birth,))
    @test_throws ArgumentError create_model(split, static; include_reflexives = (:strata, :birth))

    age = create_model(one, declared)
    no_births = typed_product(create_model(one, SIR(); include_reflexives = (:strata, :death)), age)
    @test isempty(births(no_births))
    @test length(deaths(no_births, one)) == 3
    no_deaths = typed_product(create_model(one, SIR(births = [:S]); include_reflexives = (:strata,)), age)
    @test length(births(no_deaths)) == 1
    @test isempty(deaths(no_deaths, one))
    closed = typed_product(create_model(one, SIR(); include_reflexives = (:strata,)), age)
    @test isempty(births(closed))
    @test isempty(deaths(closed, one))
    @test nt(dom(closed)) == 9
end

@testitem "Births and deaths compose with other model factors" setup = [BirthDeathSetup] begin
    for typing in (
            OnePopulationTyping(), UninfectedInfectedTyping(
                uninfected_type = :Susceptible, infected_type = :Infectious
            ),
        )
        disease = create_model(
            typing, SEIRS(number_E_stages = 2, number_I_stages = 3, births = [:S]);
            include_reflexives = (:strata, :death)
        )
        age = create_model(typing, demography())
        core = typed_product(disease, age)
        @test length(births(core)) == 1
        @test length(deaths(core, typing)) == 7

        geo_spec = GeographicStratification([:urban, :rural]; births = [:urban, :rural])
        geo = create_model(typing, geo_spec; include_reflexives = passive(typing, geo_spec))
        model = typed_product(core, geo)
        pn = dom(model)
        @test length(births(model)) == 2
        @test length(deaths(model, typing)) == 14
        @test length(unique(flattransitions(pn))) == nt(pn)
        for t in births(model)
            @test isempty(AlgebraicPetri.inputs(pn, t))
            newborn = label_parts(sname(pn, only(AlgebraicPetri.outputs(pn, t))))
            @test newborn[1] == :S
            @test newborn[2] in (:child, :child_U)
        end
        for t in deaths(model, typing)
            @test isempty(AlgebraicPetri.outputs(pn, t))
            @test label_parts(sname(pn, only(AlgebraicPetri.inputs(pn, t))))[2] in
                (:adult, :adult_U, :adult_I)
        end
        regrouped = typed_product(disease, typed_product(age, geo))
        normalize(s) = Tuple(label_parts(s))
        @test incidence_signature(model, normalize) == incidence_signature(regrouped, normalize)

        combined = typed_product(
            disease,
            create_model(typing, demography() * GeographicStratification([:urban, :rural])),
        )
        split_labels(s) = Tuple(
            vcat(
                [
                    Symbol.(split(replace(string(p), r"_[UI]$" => ""), 'x')) for p in label_parts(s)
                ]...
            )
        )
        @test incidence_signature(model, split_labels) == incidence_signature(combined, split_labels)

        silent_spec = GeographicStratification([:urban, :rural])
        silent = create_model(typing, silent_spec; include_reflexives = passive(typing, silent_spec))
        @test isempty(births(typed_product(core, silent)))
        @test length(deaths(typed_product(core, silent), typing)) == 14
        births_only = create_model(
            typing, geo_spec;
            include_reflexives = (default_reflexives(typing, geo_spec)..., :strata)
        )
        @test isempty(deaths(typed_product(core, births_only), typing))
        @test length(births(typed_product(core, births_only))) == 2
    end

    one = OnePopulationTyping()
    two = UninfectedInfectedTyping()
    for (typing, spec) in (
            (one, NoCrossImmunity([:a, :b])), (two, CompleteCrossImmunity([:a, :b])),
            (two, ImmuneHistory([:a, :b])),
        )
        disease = create_model(typing, SIR(births = [:S]); include_reflexives = (:strata, :death))
        core = typed_product(disease, create_model(typing, demography()))
        factor = create_model(typing, spec; include_reflexives = passive(typing, spec))
        model = typed_product(core, factor)
        pn = dom(model)
        @test !isempty(deaths(model, typing))
        @test all(t -> isempty(AlgebraicPetri.outputs(pn, t)), deaths(model, typing))
        @test isempty(births(model))
        @test length(unique(flattransitions(pn))) == nt(pn)
    end
end

@testitem "compose_stratifications multiplies births and deaths" setup = [BirthDeathSetup] begin
    age = demography()
    static = GeographicStratification(:urban, :rural)
    @test (age * static).demography.births == (:childxurban, :childxrural)
    @test (age * static).demography.deaths == (:adultxurban, :adultxrural)
    @test (static * age).demography.births == (:urbanxchild, :ruralxchild)
    @test (static * age).demography.deaths == (:urbanxadult, :ruralxadult)
    partial = GeographicStratification(:urban, :rural; births = [:urban], deaths = [:urban, :rural])
    @test (age * partial).demography.births == (:childxurban,)
    @test (age * partial).demography.deaths == (:adultxurban, :adultxrural)
    closed = static * GeographicStratification(:a, :b)
    @test isempty(closed.demography.births)
    @test isempty(closed.demography.deaths)
    @test ((age * static) * partial).demography.births == (age * (static * partial)).demography.births
    @test ((age * static) * partial).demography.deaths == (age * (static * partial)).demography.deaths
end

@testitem "Births and deaths ODE: total population follows B - mu N" setup = [BirthDeathSetup] begin
    typing = OnePopulationTyping()
    pn = dom(create_model(typing, SIR(births = [:S]); include_reflexives = (:death,)))
    birth_rate, death_rate = 12.0, 1 / 80
    rates = Dict(
        :transmission_S_I => 0.0005, :I_to_R => 0.2, :birth_S => birth_rate,
        :death_S => death_rate, :death_I => death_rate, :death_R => death_rate,
    )
    u0 = LVector(S = 700.0, I = 30.0, R = 70.0)
    carrying = birth_rate / death_rate
    states = solve_steps(pn, u0, rates; days = 120)
    for (day, state) in enumerate(states)
        t = day - 1
        # Mortality is the same in every compartment, so the total is blind to the epidemic.
        @test sum(state) ≈ carrying + (sum(u0) - carrying) * exp(-death_rate * t) rtol = 1.0e-10
        @test minimum(state) >= 0
    end
    @test last(states).R > u0.R
    @test sum(last(states)) > sum(u0)

    balanced = LVector(S = carrying - 40.0, I = 10.0, R = 30.0)
    @test all(s -> isapprox(sum(s), carrying; rtol = 1.0e-12), solve_steps(pn, balanced, rates))

    # Zero-order: with nobody present, births still arrive at `birth_rate`.
    du = LVector(S = 0.0, I = 0.0, R = 0.0)
    vectorfield_flat(pn)(du, LVector(S = 0.0, I = 0.0, R = 0.0), rates, 0.0)
    @test du == LVector(S = birth_rate, I = 0.0, R = 0.0)
end

@testitem "Births and deaths ODE: an ageing chain with entry and exit" setup = [BirthDeathSetup] begin
    typing = OnePopulationTyping()
    model = typed_product(
        create_model(typing, SIR(births = [:S]); include_reflexives = (:strata, :death)),
        create_model(typing, demography()),
    )
    pn = dom(model)
    births_per_day, aging, mortality = 9.0, 1 / 90, 1 / 200
    rates = Dict(name => 0.0 for name in flattransitions(pn))
    rates[:birth_S_child] = births_per_day
    for compartment in (:S, :I, :R)
        rates[Symbol(compartment, "_child_to_adult")] = aging
        rates[Symbol("death_", compartment, "_adult")] = mortality
    end
    for name in keys(rates)
        startswith(string(name), "transmission") && (rates[name] = 0.0005)
        startswith(string(name), "I_to_R") && (rates[name] = 0.2)
    end
    u0 = LVector(
        S_child = 100.0, I_child = 20.0, R_child = 30.0,
        S_adult = 400.0, I_adult = 5.0, R_adult = 10.0
    )
    children(u) = u.S_child + u.I_child + u.R_child
    adults(u) = u.S_adult + u.I_adult + u.R_adult
    # child' = B - a child;  adult' = a child - mu adult, whatever the epidemic does.
    child_eq, adult_eq = births_per_day / aging, births_per_day / mortality
    k1 = aging * (children(u0) - child_eq) / (mortality - aging)
    k2 = adults(u0) - adult_eq - k1
    states = solve_steps(pn, u0, rates; days = 180)
    for (day, state) in enumerate(states)
        t = day - 1
        @test children(state) ≈ child_eq + (children(u0) - child_eq) * exp(-aging * t) rtol = 1.0e-10
        @test adults(state) ≈ adult_eq + k1 * exp(-aging * t) + k2 * exp(-mortality * t) rtol = 1.0e-10
        @test minimum(state) >= 0
    end
    @test last(states).R_child + last(states).R_adult > u0.R_child + u0.R_adult

    equilibrium = LVector(
        S_child = child_eq, I_child = 0.0, R_child = 0.0,
        S_adult = adult_eq, I_adult = 0.0, R_adult = 0.0
    )
    for state in solve_steps(pn, equilibrium, rates; days = 60)
        @test state ≈ equilibrium rtol = 1.0e-12
    end
end
