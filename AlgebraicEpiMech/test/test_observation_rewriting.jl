# Observation attached by pushout, after composition.
#
# The two observation processes are the SAME colimit applied to different rules, so these tests
# assert the two things that distinguish them and would silently be wrong otherwise:
#
#   incidence  — one observation per event, at the event's own rate. Checked as
#                `du[accumulator] == -du[S]` on a net where susceptibles drain only to infection.
#                No rate matching is assumed, because there is only one rate.
#   prevalence — rate x occupancy, source not consumed.
#
# and the property that motivated the whole approach: attaching observation LEAVES COMPOSITION
# ALONE, so stratification factors never learn that observation exists.

@testsnippet RewriteSetup begin
    using AlgebraicPetri
    using AlgebraicPetri: snames, tnames, sname, tname
    using AlgebraicPetri.TypedPetri: typed_product
    using Catlab

    flat(x) = AlgebraicEpiMech.flatten_symbols(x)
    names(pn) = [flat(s) for s in snames(pn)]
    tnames_flat(pn) = [flat(t) for t in tnames(pn)]

    "Evaluate the vectorfield with `rates` (by flattened transition name) and `state`."
    function derivs(pn; rates, state)
        u = Dict(s => get(state, s, 0.0) for s in names(pn))
        p = Dict(t => get(rates, t, 0.0) for t in tnames_flat(pn))
        du = Dict(s => 0.0 for s in names(pn))
        vectorfield_flat(pn)(du, u, p, 0.0)
        return du
    end

    "All rates whose name starts with `pre` set to `v`."
    matching(pn, pre, v) =
        Dict(t => v for t in tnames_flat(pn) if startswith(string(t), string(pre)))

    geo_seirs(locs) = dom(
        typed_product(
            create_model(OnePopulationTyping(), SEIRS()),
            create_model(OnePopulationTyping(), GeographicStratification(locs)),
        ),
    )
end

@testitem "incidence: an infection tap counts one per infection, at the event's own rate" setup = [RewriteSetup] begin
    # SEIR, not SEIRS: with no waning, S drains only to infection, so `-du[:S]` IS incidence.
    base = dom(create_model(OnePopulationTyping(), SEIR()))
    obs = attach_observation(base, AtEvent(:transmission); n_stages = 1)

    # The event was AUGMENTED, not shadowed: no new transition.
    @test nt(obs) == nt(base)
    @test ns(obs) == ns(base) + 1
    @test :O_transmission_1 in names(obs)

    du = derivs(
        obs;
        rates = merge(matching(obs, :transmission, 0.002), matching(obs, :E_to_I, 0.4)),
        state = Dict(:S => 900.0, :E => 40.0, :I => 60.0),
    )
    @test du[:O_transmission_1] ≈ -du[:S]                 # exactly one observation per infection
    @test du[:O_transmission_1] ≈ 0.002 * 900.0 * 60.0

    # Independent of dwell times — the point of tapping the event rather than a compartment.
    for (sigma, gamma) in ((0.05, 0.05), (3.0, 2.0))
        alt = derivs(
            obs;
            rates = merge(
                matching(obs, :transmission, 0.002),
                matching(obs, :E_to_I, sigma), matching(obs, :I_to_R, gamma),
            ),
            state = Dict(:S => 900.0, :E => 40.0, :I => 60.0),
        )
        @test alt[:O_transmission_1] ≈ du[:O_transmission_1]
        @test alt[:O_transmission_1] ≈ -alt[:S]
    end
end

@testitem "prevalence: a compartment tap is rate x occupancy and consumes nobody" setup = [RewriteSetup] begin
    base = dom(create_model(OnePopulationTyping(), SEIRS()))
    obs = attach_observation(base, AtCompartment(:I); n_stages = 1)

    @test nt(obs) == nt(base) + 1            # a NEW mechanism, unlike the incidence rule
    @test :O_I_1 in names(obs)               # named for the compartment it samples

    du = derivs(
        obs;
        rates = matching(obs, :obs_inflow_I, 0.25),
        state = Dict(:S => 900.0, :I => 60.0),
    )
    @test du[:O_I_1] ≈ 0.25 * 60.0           # occupancy, not flux
    @test du[:I] ≈ 0.0                       # catalytic
end

@testitem "chains add Erlang stages and only the last accumulates" setup = [RewriteSetup] begin
    base = dom(create_model(OnePopulationTyping(), SEIR()))
    obs = attach_observation(base, AtEvent(:transmission); n_stages = 3)

    @test ns(obs) == ns(base) + 3
    @test nt(obs) == nt(base) + 2            # two delay transitions, event still augmented
    for s in (:O_transmission_1, :O_transmission_2, :O_transmission_3)
        @test s in names(obs)
    end

    # The terminal stage has no outflow: it is the accumulator the observation model reads.
    terminal = only([s for s in 1:ns(obs) if flat(sname(obs, s)) === :O_transmission_3])
    @test isempty([t for t in 1:nt(obs) if terminal in AlgebraicPetri.inputs(obs, t)])

    # Flow reaches the end of the chain.
    du = derivs(
        obs;
        rates = merge(
            matching(obs, :transmission, 0.002), matching(obs, :O_transmission_1_to_O_transmission_2, 0.5),
        ),
        state = Dict(:S => 900.0, :I => 60.0, :O_transmission_1 => 10.0),
    )
    @test du[:O_transmission_2] > 0.0
end

@testitem "composition is untouched: factors never learn observation exists" setup = [RewriteSetup] begin
    # The composed net is built with no knowledge of observation whatsoever. If attaching
    # observation required factor support, this net would be missing structure BEFORE we start —
    # which is the failure mode the pushout approach removes entirely.
    base = geo_seirs([:a, :b])
    @test ns(base) == 8                       # S,E,I,R x 2 — no accumulators
    @test nt(base) == 10                      # 4 routes + 3 progressions x 2

    obs = attach_observation(base, AtEvent(:transmission); n_stages = 1)
    @test nt(obs) == nt(base)                 # still no new transitions
    @test ns(obs) == ns(base) + 2             # one accumulator per location, NOT per route

    # Every route into a location records into that location's accumulator.
    du = derivs(
        obs;
        rates = matching(obs, :transmission, 0.002),
        state = Dict(:S_a => 900.0, :S_b => 500.0, :I_a => 60.0, :I_b => 40.0),
    )
    @test du[:O_transmission_1_a] ≈ -du[:S_a]
    @test du[:O_transmission_1_b] ≈ -du[:S_b]
    # Both routes into `a` count into the same accumulator — the pushout GLUED rather than added.
    @test du[:O_transmission_1_a] ≈ 0.002 * 900.0 * (60.0 + 40.0)
end

@testitem "escape-selective infection is preserved without any mirroring" setup = [RewriteSetup] begin
    # Under ImmuneHistory a class immune to `h` simply has no transmission for `i in h`. Because
    # we only rewrite transitions that EXIST, selectivity survives automatically — there is no
    # second structure to keep in sync and therefore nothing to get silently wrong.
    ui = UninfectedInfectedTyping()
    base = dom(
        typed_product(
            create_model(ui, SEIRS()),
            create_model(ui, ImmuneHistory([:current, :invader])),
        ),
    )
    routes = count(t -> startswith(string(flat(tname(base, t))), "transmission"), 1:nt(base))
    @test routes == 8

    obs = attach_observation(base, AtEvent(:transmission); n_stages = 1)
    @test nt(obs) == nt(base)
    @test ns(obs) == ns(base) + 4             # one chain per (h,i) junction, not per route

    accs = sort([s for s in names(obs) if startswith(string(s), "O_transmission_1_")])
    @test length(accs) == 4

    du = derivs(
        obs;
        rates = matching(obs, :transmission, 0.002),
        state = Dict(
            s => (
                    startswith(string(s), "S_U") ? 500.0 :
                    startswith(string(s), "I_") ? 50.0 : 0.0
                ) for s in names(obs)
        ),
    )
    observed = sum(du[a] for a in accs)
    infected = -sum(du[s] for s in names(obs) if startswith(string(s), "S_U"))
    @test observed ≈ infected                 # mass 1 across every escape-permitted route
end

@testitem "attaching observation is idempotent and order-independent" setup = [RewriteSetup] begin
    base = geo_seirs([:a, :b])

    once = attach_observation(base, AtEvent(:transmission); n_stages = 2)
    twice = attach_observation(once, AtEvent(:transmission); n_stages = 2)
    @test ns(twice) == ns(once)               # already recorded, nothing added
    @test nt(twice) == nt(once)

    # Both kinds can coexist: incidence off the event, prevalence off a compartment.
    both = attach_observation(
        attach_observation(base, AtEvent(:transmission); n_stages = 1),
        AtCompartment(:I); n_stages = 1, prefix = :P,
    )
    @test :O_transmission_1_a in names(both)               # incidence chain
    @test :P_I_1_a in names(both)               # prevalence chain, distinct prefix
end

@testitem "gluing a shared chain identifies every stage, not just the first" setup = [RewriteSetup] begin
    # Regression: `L` originally carried only the FIRST stage when gluing, so each additional
    # route into a stratum re-created the later stages and delay transitions. The duplicates
    # shared a label, so nothing downstream could tell them apart. Needs BOTH a shared chain
    # (stratified) and more than one stage to show up — the original tests had one or the other.
    base = geo_seirs([:a, :b])
    obs = attach_observation(base, AtEvent(:transmission); n_stages = 3)

    for loc in (:a, :b), stage in 1:3
        label = Symbol("O_transmission_", stage, "_", loc)
        @test count(==(label), names(obs)) == 1
    end
    for loc in (:a, :b), stage in 1:2
        label = Symbol(
            "O_transmission_", stage, "_to_O_transmission_", stage + 1, "_", loc
        )
        @test count(==(label), tnames_flat(obs)) == 1
    end

    # 4 routes into 2 strata: 3 stages each, and 2 delay transitions each.
    @test ns(obs) == ns(base) + 6
    @test nt(obs) == nt(base) + 4
end

@testitem "each stage of a multi-stage compartment gets its own chain" setup = [RewriteSetup] begin
    # Regression: chain labels that omitted the source collapsed `I1`, `I2`, `I3` onto one label,
    # and every match after the first was skipped as already-tapped.
    base = dom(create_model(OnePopulationTyping(), SEIR(number_I_stages = 3)))
    obs = attach_observation(base, AtCompartment(:I); n_stages = 1)

    @test nt(obs) == nt(base) + 3
    for stage in ("I1", "I2", "I3")
        @test Symbol("O_", stage, "_1") in names(obs)
        @test Symbol("obs_inflow_", stage) in tnames_flat(obs)
    end
end

@testitem "rewritten chains are recognised by observation_layout" setup = [RewriteSetup] begin
    # The naming contract is not cosmetic: `observation_layout` parses a
    # `<prefix>_<source>_<stage>` leaf, and `ConfigurableEpi`'s `StateLayout` builds on it. Labels
    # that do not parse leave the accumulators looking like core model state.
    obs = attach_observation(
        dom(create_model(OnePopulationTyping(), SEIR())),
        AtEvent(:transmission); n_stages = 2,
    )
    layout = observation_layout(obs)
    @test Set(layout.obs_names) == Set((:O_transmission_1, :O_transmission_2))
    @test length(layout.chains) == 1
    @test only(layout.chains).obs_names == (:O_transmission_1, :O_transmission_2)
    @test only(layout.cumulative_names) == :O_transmission_2

    # And on a stratified net, one chain per stratum.
    geo = attach_observation(geo_seirs([:a, :b]), AtEvent(:transmission); n_stages = 2)
    @test length(observation_layout(geo).chains) == 2
end

@testitem "direct attachment: structure, input-order stability, idempotency" setup = [RewriteSetup] begin
    # The observed net is the base net plus, per matched route, exactly one output arc into the
    # infectee location's chain head; base parts keep their order and positions; re-application
    # is a no-op. Together with the rate identities above this is the whole contract.
    using Catlab: nparts, subpart

    base = geo_seirs([:a, :b, :c])
    obs = attach_observation(base, AtEvent(:transmission); n_stages = 2)
    routes = [
        t for t in 1:AlgebraicPetri.nt(base)
            if startswith(string(tnames_flat(base)[t]), "transmission")
    ]
    @test length(routes) == 9
    @test names(obs)[1:AlgebraicPetri.ns(base)] == names(base)
    @test tnames_flat(obs)[1:AlgebraicPetri.nt(base)] == tnames_flat(base)
    # 3 chains of 2 stages, each with one delay transition
    @test AlgebraicPetri.ns(obs) == AlgebraicPetri.ns(base) + 6
    @test AlgebraicPetri.nt(obs) == AlgebraicPetri.nt(base) + 3
    @test nparts(obs, :O) == nparts(base, :O) + length(routes) + 3
    @test nparts(obs, :I) == nparts(base, :I) + 3
    for t in routes
        heads = [
            flat(sname(obs, subpart(obs, o, :os)))
                for o in 1:nparts(obs, :O) if subpart(obs, o, :ot) == t
        ]
        @test count(s -> startswith(string(s), "O_transmission_1"), heads) == 1
    end
    @test attach_observation(obs, AtEvent(:transmission); n_stages = 2) == obs

    # Prevalence: one catalytic tap transition plus a chain per matched species.
    seir = dom(create_model(OnePopulationTyping(), SEIR()))
    tap = attach_observation(seir, AtCompartment(:I); n_stages = 2)
    @test names(tap)[1:AlgebraicPetri.ns(seir)] == names(seir)
    @test AlgebraicPetri.ns(tap) == AlgebraicPetri.ns(seir) + 2
    @test AlgebraicPetri.nt(tap) == AlgebraicPetri.nt(seir) + 2
    @test attach_observation(tap, AtCompartment(:I); n_stages = 2) == tap
end
