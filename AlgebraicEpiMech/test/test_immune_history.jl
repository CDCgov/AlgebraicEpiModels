@testsnippet ImmuneHistorySetup begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri: typed_product
    using Catlab
    using LabelledArrays

    typing = UninfectedInfectedTyping()

    # The immune-history factor composed onto a generic SEIRS base.
    composed(m) = dom(typed_product(create_model(typing, SEIRS()), create_model(typing, m)))

    fnames(xs) = Set(Symbol.(AlgebraicEpiMech.flatten_symbols.(xs)))
    transmissions(pn) =
        filter(
        t -> startswith(string(t), "transmission_"),
        string.(AlgebraicEpiMech.flatten_symbols.(tnames(pn)))
    )
end

# ----------------------------------------------------------------------------
# Immune-class logic (internal helpers)
# ----------------------------------------------------------------------------

@testitem "immune classes: FullHistory = 2^n, LatestInfection = n+1" setup = [ImmuneHistorySetup] begin
    strains = [:current, :invader]
    @test length(AlgebraicEpiMech._immune_classes(FullHistory(), strains)) == 4
    @test length(AlgebraicEpiMech._immune_classes(LatestInfection(), strains)) == 3
    @test length(AlgebraicEpiMech._immune_classes(FullHistory(), [:a, :b, :c])) == 8
    @test length(AlgebraicEpiMech._immune_classes(LatestInfection(), [:a, :b, :c])) == 4
end

@testitem "susceptibility + reversion target" setup = [ImmuneHistorySetup] begin
    strains = [:current, :invader]
    @test AlgebraicEpiMech._susceptible_strains((:current,), strains) == [:invader]   # escape
    @test AlgebraicEpiMech._susceptible_strains((), strains) == [:current, :invader]
    @test AlgebraicEpiMech._recover_to(FullHistory(), (:current,), :invader) == (:current, :invader)
    @test AlgebraicEpiMech._recover_to(LatestInfection(), (:current,), :invader) == (:invader,)
end

# ----------------------------------------------------------------------------
# Composed-model structure — S indexed by {h}, E/I/R by {(h,i)}
# ----------------------------------------------------------------------------

@testitem "FullHistory composed with SEIRS (2 strains)" setup = [ImmuneHistorySetup] begin
    pn = composed(ImmuneHistory([:current, :invader]))
    sp = fnames(snames(pn))
    @test length(sp) == 16
    # S indexed by history h
    @test Set([:S_U_naive, :S_U_current, :S_U_invader, :S_U_current_invader]) ⊆ sp
    # E/I/R indexed by (h,i)
    @test Set([:E_invader_from_current, :I_invader_from_current, :R_invader_from_current]) ⊆ sp
    @test length(transmissions(pn)) == 8
end

@testitem "LatestInfection composed with SEIRS (2 strains)" setup = [ImmuneHistorySetup] begin
    pn = composed(ImmuneHistory([:current, :invader]; mode = LatestInfection()))
    sp = fnames(snames(pn))
    @test length(sp) == 15
    @test Set([:S_U_naive, :S_U_current, :S_U_invader]) ⊆ sp
    @test !(:S_U_current_invader in sp)   # no multi-immunity class under overwrite
end

@testitem "factor composes with any disease base (SEIR, multi-stage)" setup = [ImmuneHistorySetup] begin
    history = create_model(typing, ImmuneHistory([:current, :invader]; mode = LatestInfection()))
    # Same factor, different bases — the point of the compositional design.
    for base in (SEIR(), SEIRS(number_I_stages = 2))
        pn = dom(typed_product(create_model(typing, base), history))
        @test ns(pn) > 0 && nt(pn) > 0
        @test any(t -> occursin("infect_invader_current", t), transmissions(pn))  # escape survives
    end
end

# ----------------------------------------------------------------------------
# The escape property — falls out of the pullback
# ----------------------------------------------------------------------------

@testitem "escape: invader infects the incumbent-immune S, incumbent does not" setup = [ImmuneHistorySetup] begin
    for mode in (FullHistory(), LatestInfection())
        tx = transmissions(composed(ImmuneHistory([:current, :invader]; mode = mode)))
        @test any(t -> occursin("infect_invader_current", t), tx)   # invader → current-immune S
        @test any(t -> occursin("infect_current_naive", t), tx)     # naive infectable by both
        @test any(t -> occursin("infect_invader_naive", t), tx)
        @test !any(t -> occursin("infect_current_current", t), tx)  # never re-infected by own strain
        @test !any(t -> occursin("infect_invader_invader", t), tx)
    end
end

# ----------------------------------------------------------------------------
# Guards + validation
# ----------------------------------------------------------------------------

@testitem "OnePopulationTyping is rejected" setup = [ImmuneHistorySetup] begin
    @test_throws ErrorException create_model_uwd(OnePopulationTyping(), ImmuneHistory([:a, :b]))
end

@testitem "constructor validation" setup = [ImmuneHistorySetup] begin
    @test_throws ArgumentError ImmuneHistory(Symbol[])
    @test_throws ArgumentError ImmuneHistory([:a, :a])
    @test ImmuneHistory([:a, :b]) isa ImmuneHistory{FullHistory}
    @test ImmuneHistory([:a, :b]; mode = LatestInfection()) isa ImmuneHistory{LatestInfection}
end

# ----------------------------------------------------------------------------
# Behavioral: escape flux, and the history-incrementing reversion
# ----------------------------------------------------------------------------

@testitem "escape flux: invader infects the current-immune pool" setup = [ImmuneHistorySetup] begin
    pn = composed(ImmuneHistory([:current, :invader]; mode = LatestInfection()))
    states = [flatten_symbols(s) for s in snames(pn)]
    transitions = [flatten_symbols(t) for t in tnames(pn)]
    @test :transmission_S_I_infect_invader_current_by_naive in transitions

    u = LVector(;
        (
            s => (s === :S_U_current ? 10.0 : s === :I_invader_from_naive ? 5.0 : 0.0)
                for s in states
        )...
    )
    du = LVector(; (s => 0.0 for s in states)...)
    p = LVector(;
        (
            t => (t === :transmission_S_I_infect_invader_current_by_naive ? 1.0 : 0.0)
                for t in transitions
        )...
    )
    vectorfield_flat(pn)(du, u, p, 0.0)

    @test du[:E_invader_from_current] > 0.0   # invader infects the current-immune pool
    @test du[:S_U_current] < 0.0
end

@testitem "reversion increments immune history" setup = [ImmuneHistorySetup] begin
    # Recovering from the invader while immune to `current` should land in the class the
    # mode dictates: `current_invader` (Full) or `invader` (Latest).
    for (mode, target) in ((FullHistory(), :S_U_current_invader), (LatestInfection(), :S_U_invader))
        pn = composed(ImmuneHistory([:current, :invader]; mode = mode))
        states = [flatten_symbols(s) for s in snames(pn)]
        transitions = [flatten_symbols(t) for t in tnames(pn)]
        @test :R_to_S_invader_from_current in transitions

        u = LVector(; (s => (s === :R_invader_from_current ? 10.0 : 0.0) for s in states)...)
        du = LVector(; (s => 0.0 for s in states)...)
        p = LVector(; (t => (t === :R_to_S_invader_from_current ? 1.0 : 0.0) for t in transitions)...)
        vectorfield_flat(pn)(du, u, p, 0.0)

        @test du[target] > 0.0                     # lands in the incremented class
        @test du[:R_invader_from_current] < 0.0
    end
end


# ----------------------------------------------------------------------------
# Waning between immune classes: U_h → U_{h∖{i}}, one strain forgotten at a time
# ----------------------------------------------------------------------------

@testsnippet ImmuneWaningSetup begin
    waning_typing = UninfectedInfectedTyping()

    # SEIRS passive to waning, composed with an immune-history factor that implements it.
    function waning_model(strains; mode = FullHistory())
        return typed_product(
            create_model(waning_typing, SEIRS(); include_reflexives = :waning),
            create_model(waning_typing, ImmuneHistory(strains; mode = mode, waning = true)),
        )
    end

    waning_indices(model) =
        [t for t in 1:nt(dom(model)) if tname(codom(model), model[:T](t)) == :waning]

    # Flattened (name, source, target) of every :waning transition.
    function waning_arcs(model)
        pn = dom(model)
        return [
            (flatten_symbols(tname(pn, t)), flatten_symbols.(unary_arc(pn, t))...)
                for t in waning_indices(model)
        ]
    end

    # One rate per strain, assigned to every edge that forgets it by the name prefix.
    function waning_rates(model, rate_of)
        rates = Dict(n => 0.0 for n in flattransitions(dom(model)))
        for (name, _, _) in waning_arcs(model), (strain, rate) in rate_of
            startswith(string(name), "waning_S_wane_$(strain)_") && (rates[name] = rate)
        end
        return rates
    end
end

@testitem "waning edges are named per strain (2 and 3 strains, both modes)" setup = [StratumTransitionSetup, ImmuneWaningSetup] begin
    # FullHistory: wane_<strain>_<class it leaves>, landing in the class without that strain.
    @test Set(waning_arcs(waning_model([:a, :b]))) == Set(
        [
            (:waning_S_wane_a_a, :S_U_a, :S_U_naive),
            (:waning_S_wane_b_b, :S_U_b, :S_U_naive),
            (:waning_S_wane_a_a_b, :S_U_a_b, :S_U_b),
            (:waning_S_wane_b_a_b, :S_U_a_b, :S_U_a),
        ]
    )
    @test Set(waning_arcs(waning_model([:H1, :H3, :B]))) == Set(
        [
            (:waning_S_wane_H1_H1, :S_U_H1, :S_U_naive),
            (:waning_S_wane_H3_H3, :S_U_H3, :S_U_naive),
            (:waning_S_wane_B_B, :S_U_B, :S_U_naive),
            (:waning_S_wane_H1_H1_H3, :S_U_H1_H3, :S_U_H3),
            (:waning_S_wane_H3_H1_H3, :S_U_H1_H3, :S_U_H1),
            (:waning_S_wane_B_B_H1, :S_U_B_H1, :S_U_H1),
            (:waning_S_wane_H1_B_H1, :S_U_B_H1, :S_U_B),
            (:waning_S_wane_B_B_H3, :S_U_B_H3, :S_U_H3),
            (:waning_S_wane_H3_B_H3, :S_U_B_H3, :S_U_B),
            (:waning_S_wane_B_B_H1_H3, :S_U_B_H1_H3, :S_U_H1_H3),
            (:waning_S_wane_H1_B_H1_H3, :S_U_B_H1_H3, :S_U_B_H3),
            (:waning_S_wane_H3_B_H1_H3, :S_U_B_H1_H3, :S_U_B_H1),
        ]
    )
    # LatestInfection: the only immune strain is lost, so every class returns to naive.
    @test Set(waning_arcs(waning_model([:a, :b]; mode = LatestInfection()))) == Set(
        [(:waning_S_wane_a_a, :S_U_a, :S_U_naive), (:waning_S_wane_b_b, :S_U_b, :S_U_naive)]
    )
    @test Set(waning_arcs(waning_model([:H1, :H3, :B]; mode = LatestInfection()))) == Set(
        [
            (:waning_S_wane_H1_H1, :S_U_H1, :S_U_naive),
            (:waning_S_wane_H3_H3, :S_U_H3, :S_U_naive),
            (:waning_S_wane_B_B, :S_U_B, :S_U_naive),
        ]
    )

    for strains in ([:a, :b], [:H1, :H3, :B]), mode in (FullHistory(), LatestInfection())
        model = waning_model(strains; mode = mode)
        names = flattransitions(dom(model))
        per_strain = mode isa FullHistory ? 2^(length(strains) - 1) : 1
        @test length(unique(names)) == nt(dom(model))
        @test length(waning_indices(model)) == per_strain * length(strains)
        # Keyable per strain: each strain owns its edges under one prefix.
        for strain in strains
            @test count(n -> startswith(string(n), "waning_S_wane_$(strain)_"), names) == per_strain
        end
        # The base's own R → S reversion is untouched by the waning edges.
        @test any(n -> startswith(string(n), "R_to_S_"), names)
    end
end

@testitem "waning alone: each strain's immunity decays independently (3 strains)" setup = [StratumTransitionSetup, ImmuneWaningSetup] begin
    # DynODE flu_2026 durations: H1 and H3 wane after 900 days, B after 1800.
    strains = [:H1, :H3, :B]
    rate_of = Dict(:H1 => 1 / 900, :H3 => 1 / 900, :B => 1 / 1800)
    model = waning_model(strains)
    pn = dom(model)
    rates = waning_rates(model, rate_of)
    @test count(!iszero, values(rates)) == 12

    classes = AEM._immune_classes(FullHistory(), strains)
    species(h) = Symbol(:S_, AEM._uninfected_name(h))
    immune_to(state, strain) = sum(state[species(h)] for h in classes if strain in h)

    u0 = LVector(; (n => 0.0 for n in flatnames(pn))...)
    u0.S_U_B_H1_H3 = 100.0
    for (day, state) in enumerate(solve_steps(pn, u0, rates; days = 120))
        t = day - 1
        # The fully immune class empties at the sum of the three rates ...
        @test state.S_U_B_H1_H3 ≈ 100.0 * exp(-sum(values(rate_of)) * t) rtol = 1.0e-10
        # ... and immunity to each strain is lost at that strain's own rate.
        for strain in strains
            @test immune_to(state, strain) ≈ 100.0 * exp(-rate_of[strain] * t) rtol = 1.0e-10
        end
        @test sum(state[species(h)] for h in classes) ≈ 100.0   # nobody leaves S
        @test sum(state) ≈ 100.0
        @test minimum(state) >= 0
    end
end

@testitem "waning alone: LatestInfection returns each class to naive" setup = [StratumTransitionSetup, ImmuneWaningSetup] begin
    model = waning_model([:a, :b]; mode = LatestInfection())
    pn = dom(model)
    rate_a, rate_b = 0.02, 0.05
    rates = waning_rates(model, Dict(:a => rate_a, :b => rate_b))
    @test count(!iszero, values(rates)) == 2

    u0 = LVector(; (n => 0.0 for n in flatnames(pn))...)
    u0.S_U_a = 60.0
    u0.S_U_b = 40.0
    for (day, state) in enumerate(solve_steps(pn, u0, rates; days = 60))
        t = day - 1
        left_a = 60.0 * exp(-rate_a * t)
        left_b = 40.0 * exp(-rate_b * t)
        @test state.S_U_a ≈ left_a rtol = 1.0e-9
        @test state.S_U_b ≈ left_b rtol = 1.0e-9
        @test state.S_U_naive ≈ 100.0 - left_a - left_b rtol = 1.0e-9 atol = 1.0e-9
        @test sum(state) ≈ 100.0
        @test minimum(state) >= 0
    end
end

@testitem "waning adds transitions only: names and observation layout survive" setup = [StratumTransitionSetup, ImmuneWaningSetup] begin
    strains = [:H1, :H3, :B]
    waning = dom(waning_model(strains))
    plain = dom(
        typed_product(
            create_model(waning_typing, SEIRS()),
            create_model(waning_typing, ImmuneHistory(strains)),
        ),
    )
    # Without the two opt-ins the product carries no waning at all, reflexive or otherwise.
    @test !any(n -> startswith(string(n), "waning"), flattransitions(plain))
    # With them: the same species, the same transitions, plus the twelve waning edges.
    @test flatnames(waning) == flatnames(plain)
    @test issubset(flattransitions(plain), flattransitions(waning))
    added = setdiff(flattransitions(waning), flattransitions(plain))
    @test length(added) == 12
    @test all(n -> startswith(string(n), "waning_S_wane_"), added)

    observed = attach_observation(waning, AtCompartment(:I); n_stages = 2)
    observed_plain = attach_observation(plain, AtCompartment(:I); n_stages = 2)
    @test length(unique(flatnames(observed))) == ns(observed)
    @test length(unique(flattransitions(observed))) == nt(observed)
    layout = observation_layout(observed)
    reference = observation_layout(observed_plain)
    @test length(layout.chains) == 12   # one per infected (history, strain) pair
    @test layout.obs_names == reference.obs_names
    @test layout.cumulative_names == reference.cumulative_names
    @test [chain.source_name for chain in layout.chains] ==
        [chain.source_name for chain in reference.chains]
end

@testitem "flu skeleton composes with waning: strains × ages × vaccination" setup = [StratumTransitionSetup, ImmuneWaningSetup] begin
    strains = [:H1, :H3, :B]
    ages = [Symbol(:age, k) for k in 1:5]
    skeleton = typed_product(
        typed_product(
            waning_model(strains),
            create_model(waning_typing, AgeStratification(ages)),
        ),
        create_model(waning_typing, ContactStratification([:v0, :v1], :vacc)),
    )
    pn = dom(skeleton)
    names = flattransitions(pn)
    @test ns(pn) == 44 * 5 * 2
    @test length(unique(names)) == nt(pn)

    # 12 edges per (age, vaccination) stratum; the strata are passive to waning by default.
    waning = waning_indices(skeleton)
    @test length(waning) == 12 * 5 * 2
    for strain in strains
        @test count(n -> startswith(string(n), "waning_S_wane_$(strain)_"), names[waning]) == 40
    end
    for t in waning
        source, target = label_parts.(unary_arc(pn, t))
        @test source[1] == target[1] == :S
        @test source[2] != target[2]            # immune class changes
        @test source[3:end] == target[3:end]    # age and vaccination status do not
    end
    vectorfield_flat(pn)
end
