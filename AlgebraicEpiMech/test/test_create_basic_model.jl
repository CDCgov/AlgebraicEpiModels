@testitem "create_model returns ACSetTransformation" setup = [ModelSetup] begin
    model = SEIR()
    typed_model = create_model(one_pop_typing, model)

    # Should return an ACSetTransformation (typed Petri net)
    @test typed_model isa ACSetTransformation
end

@testitem "create_model typed model has dom as LabelledPetriNet" setup = [ModelSetup] begin
    model = SIR()
    typed_model = create_model(one_pop_typing, model)

    # The domain should be a LabelledPetriNet (usable for ODEs)
    pn = dom(typed_model)
    @test pn isa LabelledPetriNet
end

@testitem "create_model type system accessible via codom" setup = [ModelSetup] begin
    model = SEIR()
    typed_model = create_model(one_pop_typing, model)

    # The type system should be accessible via codom
    schema_pn = codom(typed_model)
    @test schema_pn isa LabelledPetriNet
end

@testitem "create_model(SI) creates valid typed model" setup = [ModelSetup] begin
    model = SI()
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 2  # S, I
    @test length(AlgebraicPetri.tnames(pn)) == 1  # transmission
end

@testitem "create_model(SI) with multi-stage I" setup = [ModelSetup] begin
    model = SI(number_I_stages = 3)
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, I1, I2, I3
end

@testitem "create_model(SEI) creates valid typed model" setup = [ModelSetup] begin
    model = SEI()
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 3  # S, E, I
    @test length(AlgebraicPetri.tnames(pn)) == 2  # transmission, density
end

@testitem "create_model(SEI) with multi-stage E and I" setup = [ModelSetup] begin
    model = SEI(number_E_stages = 2, number_I_stages = 3)
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 6  # S, E1, E2, I1, I2, I3
end

@testitem "create_model(SIR) creates valid typed model" setup = [ModelSetup] begin
    model = SIR()
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 3  # S, I, R
    @test length(AlgebraicPetri.tnames(pn)) == 2  # transmission, density
end

@testitem "create_model(SEIR) creates valid typed model" setup = [ModelSetup] begin
    model = SEIR()
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, E, I, R
    @test length(AlgebraicPetri.tnames(pn)) == 3  # transmission, density, density
end

@testitem "create_model(SEIR) with multi-stage E and I" setup = [ModelSetup] begin
    model = SEIR(number_E_stages = 2, number_I_stages = 3)
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 7  # S, E1, E2, I1, I2, I3, R
end

@testitem "create_model(SIS) creates valid typed model" setup = [ModelSetup] begin
    model = SIS()
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 2  # S, I
    @test length(AlgebraicPetri.tnames(pn)) == 2  # transmission, reversion
end

@testitem "create_model(SEIS) creates valid typed model" setup = [ModelSetup] begin
    model = SEIS()
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 3  # S, E, I
    @test length(AlgebraicPetri.tnames(pn)) == 3  # transmission, density, reversion
end

@testitem "create_model(SEIRS) creates valid typed model" setup = [ModelSetup] begin
    model = SEIRS()
    typed_model = create_model(one_pop_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, E, I, R
    @test length(AlgebraicPetri.tnames(pn)) == 4  # transmission, density, density, reversion
end

@testitem "create_model works with UninfectedInfectedTyping" setup = [ModelSetup] begin
    model = SEIR()
    typed_model = create_model(ui_typing, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, E, I, R
    @test typed_model isa ACSetTransformation
end

@testitem "create_model output can be used for ODE simulation" setup = [ModelSetup] begin
    model = SIR()
    typed_model = create_model(one_pop_typing, model)
    pn = dom(typed_model)

    # Create vectorfield, wrapped into the out-of-place `(x, u, p, t) -> dx` form SeeToDee
    # discretises (`u` is the control input, unused here)
    f! = vectorfield_flat(pn)
    f(x, u, p, t) = f!(similar(x), x, p, t)

    # Set up initial conditions and parameters using auto-generated transition names
    u0 = LVector(S = 990.0, I = 10.0, R = 0.0)
    p = LVector(transmission_S_I = 0.0005, I_to_R = 0.25)

    # Discretise to daily RK4 steps and integrate over 40 days
    step = Rk4(f, 1.0; supersample = 4)
    x = foldl((x, t) -> step(x, nothing, p, t), 0.0:1.0:39.0; init = u0)

    # Verify solution is valid
    @test all(isfinite, x)
    @test sum(x) ≈ sum(u0)  # SIR conserves population
    @test x[:R] > 0.0  # Some recovery should occur
end

@testitem "create_model output can be composed with typed_product" setup = [ModelSetup] begin
    # Create two simple models
    si_model = create_model(one_pop_typing, SI())
    sir_model = create_model(one_pop_typing, SIR())

    # Should be able to compose them (this tests the typing is compatible)
    # typed_product requires both to be ACSetTransformations
    @test si_model isa ACSetTransformation
    @test sir_model isa ACSetTransformation

    # Verify codomain (type system) is the same type
    @test codom(si_model) isa LabelledPetriNet
    @test codom(sir_model) isa LabelledPetriNet

    # Verify both codomains are equivalent (same type-system structure)
    @test codom(si_model) == codom(sir_model)
end

@testitem "create_model codom matches type_system" setup = [ModelSetup] begin
    model = SEIR()

    # Create typed model
    typed_model = create_model(one_pop_typing, model)

    # Get type system via codom
    codomain_from_typed_model = codom(typed_model)

    # Create the type system directly
    direct_type_system = type_system(one_pop_typing)

    # Both should be LabelledPetriNets
    @test codomain_from_typed_model isa LabelledPetriNet
    @test direct_type_system isa LabelledPetriNet

    # They should be equal in structure
    @test codomain_from_typed_model == direct_type_system
end

@testitem "create_model forwards extra transitions to type_system" setup = [ModelSetup] begin
    strata = :strata => (:Individual => :Individual)
    birth = :birth => (:Individual => (:Individual, :Individual))

    typed_model = create_model(one_pop_typing, SIR(); population_transitions = strata)
    @test codom(typed_model) == type_system(one_pop_typing, strata)
    @test :strata in tnames(codom(typed_model))
    @test dom(typed_model) == dom(create_model(one_pop_typing, SIR()))

    two_extra = create_model(one_pop_typing, SIR(); population_transitions = [strata, birth])
    @test codom(two_extra) == type_system(one_pop_typing, strata, birth)

    @test codom(create_model(one_pop_typing, SIR(); population_transitions = nothing)) ==
        type_system(one_pop_typing)
end

@testitem "models over an extended type system compose" setup = [ModelSetup] begin
    strata = :strata => (:Individual => :Individual)

    sir = create_model(one_pop_typing, SIR(); population_transitions = strata)
    age = create_model(
        one_pop_typing, AgeStratification([:child, :adult]); population_transitions = strata
    )
    extended = dom(typed_product(sir, age))

    standard = dom(
        typed_product(
            create_model(one_pop_typing, SIR()),
            create_model(one_pop_typing, AgeStratification([:child, :adult])),
        )
    )
    @test (ns(extended), nt(extended)) == (ns(standard), nt(standard))
end
