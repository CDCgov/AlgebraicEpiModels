@testsnippet ImmuneHistorySetup begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri: typed_product
    using Catlab
    using LabelledArrays
    using AlgebraicEpiMech

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
