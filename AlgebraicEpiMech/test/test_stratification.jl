@testitem "Stratification composition" begin
    using AlgebraicEpiMech

    @testset "ContactStratification composition via compose_stratifications" begin
        age = ContactStratification([:child, :adult], :age)
        geo = ContactStratification([:urban, :rural], :geography)

        # Compose using function
        age_geo = compose_stratifications(age, geo)

        # Check stratum names are Cartesian product
        @test age_geo.stratum_names ==
            [:childxurban, :childxrural, :adultxurban, :adultxrural]

        # Check label is combined
        @test age_geo.label == :age_x_geography

        # Check it's still a valid ContactStratification
        @test age_geo isa ContactStratification
        @test length(age_geo.stratum_names) == 4
    end

    @testset "ContactStratification composition via * operator" begin
        age = ContactStratification([:child, :adult], :age)
        geo = ContactStratification([:urban, :rural], :geography)

        # Compose using * operator
        age_geo = age * geo

        # Should give same results as function
        @test age_geo.stratum_names ==
            [:childxurban, :childxrural, :adultxurban, :adultxrural]
        @test age_geo.label == :age_x_geography
    end

    @testset "Composition with convenience constructors" begin
        age = AgeStratification([:child, :adult, :elderly])
        geo = GeographicStratification([:urban, :suburban, :rural])

        age_geo = age * geo

        # Should create 3×3 = 9 strata
        @test length(age_geo.stratum_names) == 9

        # Check ordering (child × all locations first, then adult × all, etc.)
        expected = [
            :childxurban, :childxsuburban, :childxrural,
            :adultxurban, :adultxsuburban, :adultxrural,
            :elderlyxurban, :elderlyxsuburban, :elderlyxrural,
        ]
        @test age_geo.stratum_names == expected

        # Check label
        @test age_geo.label == :age_x_geography
    end

    @testset "Triple composition (associativity)" begin
        age = ContactStratification([:child, :adult], :age)
        geo = ContactStratification([:urban, :rural], :geography)
        risk = ContactStratification([:low, :high], :risk)

        # Compose in different orders
        result1 = (age * geo) * risk
        result2 = age * (geo * risk)

        # Both should have 2×2×2 = 8 strata
        @test length(result1.stratum_names) == 8
        @test length(result2.stratum_names) == 8

        # Check some expected names
        @test :childxurbanxlow in result1.stratum_names
        @test :adultxruralxhigh in result1.stratum_names

        # Note: labels will differ due to grouping
        @test result1.label == :age_x_geography_x_risk
        @test result2.label == :age_x_geography_x_risk  # Should still combine nicely
    end

    @testset "Composition preserves uniqueness constraint" begin
        strat1 = ContactStratification([:a, :b], :test1)
        strat2 = ContactStratification([:x, :y], :test2)

        composed = strat1 * strat2

        # All names should be unique
        @test length(composed.stratum_names) == length(unique(composed.stratum_names))
    end

    @testset "Edge case: single stratum compositions" begin
        single = ContactStratification([:only], :single)
        multi = ContactStratification([:a, :b, :c], :multi)

        result = single * multi

        @test result.stratum_names == [:onlyxa, :onlyxb, :onlyxc]
        @test length(result.stratum_names) == 3
    end
end

@testitem "ContactStratification model integration" begin
    using AlgebraicEpiMech
    using AlgebraicPetri, AlgebraicPetri.TypedPetri
    using Catlab

    @testset "Composed stratification creates valid model" begin
        schema = OnePopulationSchema()

        age = AgeStratification([:child, :adult])
        geo = GeographicStratification([:urban, :rural])

        # Create product stratification
        age_geo = age * geo

        # Should be able to create model from composed stratification
        age_geo_model = create_model(schema, age_geo)

        # Extract Petri net
        pn = dom(age_geo_model)

        # Should have 4 strata (junctions)
        @test ns(pn) == 4

        # Should have n² + 3n = 28 transitions (16 transmission + 12 reflexive)
        @test nt(pn) == 28
    end

    @testset "Composed stratification equivalent to sequential composition" begin
        schema = OnePopulationSchema()
        sir = create_model(schema, SIR())
        age = AgeStratification([:child, :adult])
        geo = GeographicStratification([:urban, :rural])

        # Approach 1: Sequential typed_product
        age_model = create_model(schema, age)
        geo_model = create_model(schema, geo)
        sir_age = typed_product(sir, age_model)
        sir_age_geo_sequential = typed_product(sir_age, geo_model)

        # Approach 2: Pre-composed stratification
        age_geo = age * geo
        age_geo_model = create_model(schema, age_geo)
        sir_age_geo_composed = typed_product(sir, age_geo_model)

        # Both should create same structure
        pn_seq = dom(sir_age_geo_sequential)
        pn_comp = dom(sir_age_geo_composed)

        # Same number of states (S, I, R) × 4 strata
        @test ns(pn_seq) == ns(pn_comp)
        @test ns(pn_comp) == 12  # 3 compartments × 4 strata

        # Same number of transitions
        @test nt(pn_seq) == nt(pn_comp)
    end

    @testset "Transmission transitions preserve infectee stratum (regression test)" begin
        # This test catches the bug where S_child + I_adult → I_adult + I_adult
        # instead of the correct S_child + I_adult → I_child + I_adult

        schema = OnePopulationSchema()
        sir = create_model(schema, SIR())
        age = AgeStratification([:child, :adult])

        age_model = create_model(schema, age)
        age_sir = typed_product(sir, age_model)
        pn = dom(age_sir)

        # Find the transmission transition where child is infected by adult
        # Transition names format: (:transmission_S_I, :child_adult)
        # This means "child infected by adult"
        child_by_adult_idx = findfirst(
            t -> t == (:transmission_S_I, :child_adult),
            tnames(pn)
        )

        @test child_by_adult_idx !== nothing

        # Get the inputs and outputs of this transition
        inputs = [pn[edge, :is] for edge in incident(pn, child_by_adult_idx, :it)]
        outputs = [pn[edge, :os] for edge in incident(pn, child_by_adult_idx, :ot)]

        input_states = [snames(pn)[i] for i in inputs]
        output_states = [snames(pn)[i] for i in outputs]

        # Expected: S_child + I_adult → I_child + I_adult
        # The infectee (child) should become I_child, not I_adult
        @test Set(input_states) == Set([(:S, :child), (:I, :adult)])
        @test Set(output_states) == Set([(:I, :child), (:I, :adult)])

        # Critical check: infectee (child) stays in child stratum when infected
        @test (:I, :child) in output_states
        @test !any(
            s -> s == (:I, :adult) && count(==(s), output_states) == 2, [
                (
                    :I, :adult,
                ),
            ]
        )
    end

    @testset "Multi-stratification transmission preserves infectee stratum" begin
        # Test with composed stratifications (age × geography)
        schema = OnePopulationSchema()
        sir = create_model(schema, SIR())

        age = AgeStratification([:child, :adult])
        geo = GeographicStratification([:urban, :rural])
        age_geo = age * geo

        age_geo_model = create_model(schema, age_geo)
        age_geo_sir = typed_product(sir, age_geo_model)
        pn = dom(age_geo_sir)

        # Test critical case: urban child infected by rural adult
        # Transition name format: infectee_infector (with 'x' within each stratum name)
        childxurban_by_adultxrural_idx = findfirst(
            t -> t == (:transmission_S_I, :childxurban_adultxrural),
            tnames(pn)
        )

        @test childxurban_by_adultxrural_idx !== nothing

        inputs = [
            pn[edge, :is]
                for edge in incident(pn, childxurban_by_adultxrural_idx, :it)
        ]
        outputs = [
            pn[edge, :os]
                for edge in incident(pn, childxurban_by_adultxrural_idx, :ot)
        ]

        input_states = [snames(pn)[i] for i in inputs]
        output_states = [snames(pn)[i] for i in outputs]

        # Expected: S_childxurban + I_adultxrural → I_childxurban + I_adultxrural
        # The infectee (childxurban) must become I_childxurban, NOT I_adultxrural
        @test Set(input_states) == Set([(:S, :childxurban), (:I, :adultxrural)])
        @test Set(output_states) == Set([(:I, :childxurban), (:I, :adultxrural)])

        # Verify infectee stays in their stratum
        @test (:I, :childxurban) in output_states

        # Test several other cross-stratum transmissions
        test_cases = [
            (
                (:transmission_S_I, :childxrural_adultxurban),
                [(:S, :childxrural), (:I, :adultxurban)],
                [(:I, :childxrural), (:I, :adultxurban)],
            ),
            (
                (:transmission_S_I, :adultxurban_childxrural),
                [(:S, :adultxurban), (:I, :childxrural)],
                [(:I, :adultxurban), (:I, :childxrural)],
            ),
        ]

        for (tname, expected_inputs, expected_outputs) in test_cases
            tidx = findfirst(t -> t == tname, tnames(pn))
            @test tidx !== nothing

            inputs = [pn[edge, :is] for edge in incident(pn, tidx, :it)]
            outputs = [pn[edge, :os] for edge in incident(pn, tidx, :ot)]

            input_states = [snames(pn)[i] for i in inputs]
            output_states = [snames(pn)[i] for i in outputs]

            @test Set(input_states) == Set(expected_inputs)
            @test Set(output_states) == Set(expected_outputs)
        end
    end
end
