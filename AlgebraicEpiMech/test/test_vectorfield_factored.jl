@testsnippet FactoredSetup begin
    using AlgebraicPetri, LabelledArrays, Catlab
    using AlgebraicPetri: snames, tnames
    using AlgebraicPetri.TypedPetri: typed_product

    # Deterministic values, distinct and non-round (the test env has no RNG).
    strided(n, seed) = [mod(seed + k * 0.6180339887, 1.0) for k in 1:n]

    flat_names(names) = [flatten_symbols(n) for n in names]
    labelled(names, values) = LVector(NamedTuple{Tuple(names)}(Tuple(values)))

    # Largest difference between the factored field and `vectorfield_flat`, relative to the
    # largest flat component, with the flat rates expanded as `weight * p[key]`.
    function factored_error(pn; rate_class = nothing, fixed = (;), seed = 0.1)
        species, transitions = flat_names(snames(pn)), flat_names(tnames(pn))
        structure = rate_class === nothing ? kinetic_structure(pn; fixed) :
            kinetic_structure(pn; rate_class, fixed)
        classes = rate_class === nothing ? [(t, 1.0) for t in transitions] :
            [(flatten_symbols(first(c)), last(c)) for c in rate_class.(tnames(pn))]
        keyed = Dict(zip(structure.rate_keys, strided(length(structure.rate_keys), 3 * seed)))
        value(key) = haskey(keyed, key) ? keyed[key] : Float64(fixed[key])
        u = labelled(species, 100 .* strided(length(species), seed))
        p_flat = NamedTuple{Tuple(transitions)}(Tuple(w * value(k) for (k, w) in classes))
        p_keyed = NamedTuple{Tuple(structure.rate_keys)}(
            Tuple(keyed[k] for k in structure.rate_keys)
        )
        du_flat = labelled(species, zeros(length(species)))
        du_factored = labelled(species, fill(NaN, length(species)))
        vectorfield_flat(pn)(du_flat, u, p_flat, 0.0)
        vectorfield_factored(structure)(du_factored, u, p_keyed, 0.0)
        return maximum(abs.(du_flat .- du_factored)) / maximum(abs.(du_flat))
    end

    single_population() = attach_observation(
        dom(create_model(OnePopulationTyping(), SEIRS())), AtEvent(:transmission); n_stages = 2,
    )

    age_stratified() = attach_observation(
        dom(
            typed_product(
                create_model(
                    OnePopulationTyping(), SEIRS(number_I_stages = 2, births = [:S]);
                    include_reflexives = (:strata, :death),
                ),
                create_model(
                    OnePopulationTyping(),
                    AgeStratification(
                        [:child, :adult, :elder];
                        transitions = [:child => :adult, :adult => :elder],
                        births = [:child], deaths = [:elder],
                    ),
                ),
            ),
        ),
        AtEvent(:transmission); n_stages = 2,
    )

    # The flu net: SEIRS x full immune history over three subtypes x five ages, with waning and
    # importation, observed by one chain per (age, subtype).
    function flu()
        typing = UninfectedInfectedTyping()
        ages = AgeStratification([:a0, :a1, :a2, :a3, :a4])
        history = ImmuneHistory([:H1, :H3, :B]; waning = true)
        importing(spec) = (default_reflexives(typing, spec)..., :importation)
        composed = typed_product(
            typed_product(
                create_model(typing, SEIRS(importation = true); include_reflexives = :waning),
                create_model(typing, history; include_reflexives = importing(history)),
            ),
            create_model(typing, ages; include_reflexives = importing(ages)),
        )
        chain_key(((compartment, junction), age)) =
            ((compartment, Symbol(first(split(string(junction), "_from_")))), age)
        infections = AtEvent((:transmission, :importation); source = :infection, group = chain_key)
        return attach_observation(dom(composed), infections; n_stages = 2)
    end

    # One key per (subtype, age pair), weighted by the size of the infectee's immune history.
    function flu_rate_class(name)
        (disease, immune), age = name
        disease == :transmission_S_I || return (name, 1.0)
        parts = match(r"^infect_([^_]+)_(.+)_by_(.+)$", string(immune))
        history = parts[2] == "naive" ? 0 : count(==('_'), parts[2]) + 1
        return ((:transmission, Symbol(parts[1]), age), 0.7^history)
    end
end

@testitem "vectorfield_factored reproduces vectorfield_flat with the default classes" setup = [FactoredSetup] begin
    dimer = LabelledPetriNet([:A, :B], :dimerise => ((:A, :A) => :B), :decay => (:B => :A))
    autocatalysis = LabelledPetriNet([:A, :B], :grow => ((:A, :A) => (:A, :B)))
    cubic = LabelledPetriNet([:A, :B, :C], :join => ((:A, :A, :B) => :C), :split => (:C => (:A, :B)))
    open_chain = LabelledPetriNet(
        [:A, :B], :source => (() => :A), :move => (:A => :B), :sink => (:B => ())
    )
    tap = LabelledPetriNet([:S, :I, :O], :tap => ((:S, :I) => (:S, :I, :O)), :clear => (:O => ()))
    for pn in (single_population(), age_stratified(), flu(), dimer, autocatalysis, cubic, open_chain, tap)
        for seed in (0.1, 0.2)
            @test factored_error(pn; seed) < 1.0e-13
        end
    end

    report = kinetic_structure(cubic).report
    @test (report.first_order, report.residual, report.catalysed) == (1, 1, 0)
    @test kinetic_structure(dimer).report.residual == 1
    @test kinetic_structure(autocatalysis).report.catalysed == 1
    @test kinetic_structure(open_chain).report.zero_order == 1
end

@testitem "vectorfield_factored reproduces vectorfield_flat with declared classes" setup = [FactoredSetup] begin
    pn = single_population()
    shared(name) = startswith(string(flatten_symbols(name)), "transmission") ? (:beta, 1.0) : (name, 1.0)
    @test factored_error(pn; rate_class = shared) < 1.0e-13

    # One contact rate per age pair, shared by the two infectious stages.
    pn = age_stratified()
    transmits(name) = startswith(string(name[1]), "transmission")
    by_pair(name) = transmits(name) ? ((:contact, name[2]) => 0.5) : (name => 1.0)
    report = kinetic_structure(pn; rate_class = by_pair).report
    @test report.catalysed == 18
    @test (report.classes, report.left_factors, report.right_factors) == (3, 3, 3)
    @test report.kernel_entries == 9
    @test factored_error(pn; rate_class = by_pair) < 1.0e-13

    pn = flu()
    @test factored_error(pn; rate_class = flu_rate_class) < 1.0e-13
    # Fixing every first-order rate leaves only the transmission keys in `p`.
    transmission(name) = name[1][1] == :transmission_S_I
    first_order = flat_names(filter(!transmission, tnames(pn)))
    fixed = NamedTuple{Tuple(first_order)}(Tuple(strided(length(first_order), 0.4)))
    @test factored_error(pn; rate_class = flu_rate_class, fixed) < 1.0e-13
    structure = kinetic_structure(pn; rate_class = flu_rate_class, fixed)
    @test structure.report.rate_keys == 3 * 25
    @test all(startswith("transmission"), string.(structure.rate_keys))
end

@testitem "kinetic_structure finds the cells of the flu net" setup = [FactoredSetup] begin
    pn = flu()
    default = kinetic_structure(pn).report
    @test default.catalysed == 1200
    @test default.classes == 60
    @test default.catalysts_per_class == (20, 20)
    @test default.kernel_entries == 1200

    declared = kinetic_structure(pn; rate_class = flu_rate_class).report
    @test declared.classes == 60
    @test (declared.left_factors, declared.right_factors) == (15, 15)
    @test declared.kernel_entries == 75
    @test declared.keys_not_rank_one == 0
    @test declared.max_weight_error < 1.0e-12
    @test declared.factored_operations < default.factored_operations < default.flat_operations

    @test occursin("60 classes", sprint(show, MIME"text/plain"(), kinetic_structure(pn)))
end

@testitem "a rate that depends on the catalyst beyond its cell keeps its rows" setup = [FactoredSetup] begin
    # Infectiousness depends on the infector's stage differently for each infectee, so the
    # shared key's weight matrix is not rank one.
    pn = LabelledPetriNet(
        [:S1, :S2, :I1, :I2, :E],
        :a => ((:S1, :I1) => (:E, :I1)), :b => ((:S1, :I2) => (:E, :I2)),
        :c => ((:S2, :I1) => (:E, :I1)), :d => ((:S2, :I2) => (:E, :I2)),
    )
    weights = (a = 1.0, b = 2.0, c = 3.0, d = 5.0)
    not_separable(name) = (:beta, weights[name])
    report = kinetic_structure(pn; rate_class = not_separable).report
    @test report.keys_not_rank_one == 1
    @test report.max_row_directions == 2
    @test factored_error(pn; rate_class = not_separable) < 1.0e-13

    separable(name) = (:beta, (a = 1.0, b = 2.0, c = 3.0, d = 6.0)[name])
    report = kinetic_structure(pn; rate_class = separable).report
    @test report.keys_not_rank_one == 0
    @test (report.left_factors, report.right_factors, report.kernel_entries) == (1, 1, 1)
    @test factored_error(pn; rate_class = separable) < 1.0e-13
end

@testitem "kinetic_structure rejects weights too far apart to share a factor" setup = [FactoredSetup] begin
    two_catalysts = LabelledPetriNet(
        [:S, :I1, :I2, :E], :a => ((:S, :I1) => (:E, :I1)), :b => ((:S, :I2) => (:E, :I2)),
    )
    two_substrates = LabelledPetriNet(
        [:S1, :S2, :I, :E], :a => ((:S1, :I) => (:E, :I)), :b => ((:S2, :I) => (:E, :I)),
    )
    spread(wa, wb) = name -> (:beta, name == :a ? wa : wb)
    # One class's row (the right factor), overflowing and then underflowing.
    @test_throws "rate key `beta`" kinetic_structure(two_catalysts; rate_class = spread(1.0e-200, 1.0e200))
    @test_throws ArgumentError kinetic_structure(two_catalysts; rate_class = spread(1.0e200, 1.0e-200))
    # Two classes with one row direction (the left factor).
    @test_throws ArgumentError kinetic_structure(two_substrates; rate_class = spread(1.0e-200, 1.0e200))
    @test factored_error(two_catalysts; rate_class = spread(1.0e-150, 1.0e150)) < 1.0e-13
end

@testitem "vectorfield_factored - arguments" setup = [FactoredSetup] begin
    pn = LabelledPetriNet(
        [:S, :I, :R], :infection => ((:S, :I) => (:I, :I)), :recovery => (:I => :R)
    )
    u = LVector(S = 10.0, I = 2.0, R = 0.0)
    du = LVector(S = 0.0, I = 0.0, R = 0.0)

    f = vectorfield_factored(pn)
    f(du, u, (recovery = 0.5, infection = 0.01), 0.0)
    @test du ≈ LVector(S = -0.2, I = -0.8, R = 1.0)
    @test @allocated(f(du, u, (recovery = 0.5, infection = 0.01), 0.0)) == 0

    du_dict = Dict(:S => 0.0, :I => 0.0, :R => 0.0)
    f(du_dict, Dict(:S => 10.0, :I => 2.0, :R => 0.0), Dict(:infection => 0.01, :recovery => 0.5), 0.0)
    @test all(du_dict[k] ≈ du[k] for k in (:S, :I, :R))

    f(du, u, (infection = (u, t) -> 0.001 * u[:S], recovery = 0.5), 0.0)
    @test du ≈ LVector(S = -0.2, I = -0.8, R = 1.0)

    @test_throws ArgumentError f(du, u, (infection = 0.01,), 0.0)

    fixed = vectorfield_factored(pn; fixed = (recovery = 0.5,))
    fixed(du, u, (infection = 0.01,), 0.0)
    @test du ≈ LVector(S = -0.2, I = -0.8, R = 1.0)
    # A fixed rate cannot also arrive in `p`, where it would be ignored.
    @test_throws ArgumentError fixed(du, u, (infection = 0.01, recovery = 0.1), 0.0)
    @test_throws ArgumentError fixed(du, u, Dict(:infection => 0.01, :recovery => 0.1), 0.0)
    @test_throws ArgumentError kinetic_structure(pn; fixed = (nonsense = 1.0,))

    # A zero rate removes its transitions from the structure.
    @test kinetic_structure(pn; fixed = (infection = 0.0,)).report.inactive == 1

    @test_throws ArgumentError kinetic_structure(pn; rate_class = name -> name)
    @test_throws ArgumentError kinetic_structure(pn; rate_class = name -> (name, Inf))

    TupleNamedPetriNet = AlgebraicPetri.LabelledPetriNetUntyped{Tuple{Symbol, Symbol}}
    collision = TupleNamedPetriNet()
    add_species!(collision, 1; sname = [(:state, :one)])
    add_transitions!(collision, 2; tname = [(:a_b, :c), (:a, :b_c)])
    @test_throws ArgumentError kinetic_structure(collision)
end
