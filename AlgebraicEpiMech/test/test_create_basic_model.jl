# Unit tests for create_basic_model.jl
# Tests the create_model function that produces typed Petri nets

@testsnippet BasicModelSetup begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri
    using Catlab
    using AlgebraicEpiMech
    using LabelledArrays
    using SeeToDee: Rk4

    # Create schema instances for testing
    one_pop_schema = OnePopulationSchema(population_type = :Individual)
    ui_schema = UninfectedInfectedSchema(
        uninfected_type = :Susceptible,
        infected_type = :Infectious
    )
end

# Test that create_model returns typed Petri net
@testitem "create_model returns ACSetTransformation" setup = [BasicModelSetup] begin
    model = SEIR()
    typed_model = create_model(one_pop_schema, model)

    # Should return an ACSetTransformation (typed Petri net)
    @test typed_model isa ACSetTransformation
end

@testitem "create_model typed model has dom as LabelledPetriNet" setup = [BasicModelSetup] begin
    model = SIR()
    typed_model = create_model(one_pop_schema, model)

    # The domain should be a LabelledPetriNet (usable for ODEs)
    pn = dom(typed_model)
    @test pn isa LabelledPetriNet
end

@testitem "create_model schema accessible via codom" setup = [BasicModelSetup] begin
    model = SEIR()
    typed_model = create_model(one_pop_schema, model)

    # The schema should be accessible via codom
    schema_pn = codom(typed_model)
    @test schema_pn isa LabelledPetriNet
end

# Test SI model
@testitem "create_model(SI) creates valid typed model" setup = [BasicModelSetup] begin
    model = SI()
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 2  # S, I
    @test length(AlgebraicPetri.tnames(pn)) == 1  # transmission
end

@testitem "create_model(SI) with multi-stage I" setup = [BasicModelSetup] begin
    model = SI(number_I_stages = 3)
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, I1, I2, I3
end

# Test SEI model
@testitem "create_model(SEI) creates valid typed model" setup = [BasicModelSetup] begin
    model = SEI()
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 3  # S, E, I
    @test length(AlgebraicPetri.tnames(pn)) == 2  # transmission, density
end

@testitem "create_model(SEI) with multi-stage E and I" setup = [BasicModelSetup] begin
    model = SEI(number_E_stages = 2, number_I_stages = 3)
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 6  # S, E1, E2, I1, I2, I3
end

# Test SIR model
@testitem "create_model(SIR) creates valid typed model" setup = [BasicModelSetup] begin
    model = SIR()
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 3  # S, I, R
    @test length(AlgebraicPetri.tnames(pn)) == 2  # transmission, density
end

# Test SEIR model
@testitem "create_model(SEIR) creates valid typed model" setup = [BasicModelSetup] begin
    model = SEIR()
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, E, I, R
    @test length(AlgebraicPetri.tnames(pn)) == 3  # transmission, density, density
end

@testitem "create_model(SEIR) with multi-stage E and I" setup = [BasicModelSetup] begin
    model = SEIR(number_E_stages = 2, number_I_stages = 3)
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 7  # S, E1, E2, I1, I2, I3, R
end

# Test SIS model
@testitem "create_model(SIS) creates valid typed model" setup = [BasicModelSetup] begin
    model = SIS()
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 2  # S, I
    @test length(AlgebraicPetri.tnames(pn)) == 2  # transmission, reversion
end

# Test SEIS model
@testitem "create_model(SEIS) creates valid typed model" setup = [BasicModelSetup] begin
    model = SEIS()
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 3  # S, E, I
    @test length(AlgebraicPetri.tnames(pn)) == 3  # transmission, density, reversion
end

# Test SEIRS model
@testitem "create_model(SEIRS) creates valid typed model" setup = [BasicModelSetup] begin
    model = SEIRS()
    typed_model = create_model(one_pop_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, E, I, R
    @test length(AlgebraicPetri.tnames(pn)) == 4  # transmission, density, density, reversion
end

# Test with UninfectedInfectedSchema
@testitem "create_model works with UninfectedInfectedSchema" setup = [BasicModelSetup] begin
    model = SEIR()
    typed_model = create_model(ui_schema, model)

    pn = dom(typed_model)
    @test length(AlgebraicPetri.snames(pn)) == 4  # S, E, I, R
    @test typed_model isa ACSetTransformation
end

# Test that model can be used for ODE simulation
@testitem "create_model output can be used for ODE simulation" setup = [BasicModelSetup] begin
    using LabelledArrays

    model = SIR()
    typed_model = create_model(one_pop_schema, model)
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

# Test model composition capability
@testitem "create_model output can be composed with typed_product" setup = [BasicModelSetup] begin
    # Create two simple models
    si_model = create_model(one_pop_schema, SI())
    sir_model = create_model(one_pop_schema, SIR())

    # Should be able to compose them (this tests the typing is compatible)
    # typed_product requires both to be ACSetTransformations
    @test si_model isa ACSetTransformation
    @test sir_model isa ACSetTransformation

    # Verify codomain (schema) is the same type
    @test codom(si_model) isa LabelledPetriNet
    @test codom(sir_model) isa LabelledPetriNet

    # Verify both codomains are equivalent (same schema structure)
    @test codom(si_model) == codom(sir_model)
end

# Test that codom matches direct create_schema call
@testitem "create_model codom matches create_schema" setup = [BasicModelSetup] begin
    model = SEIR()

    # Create typed model
    typed_model = create_model(one_pop_schema, model)

    # Get schema via codom
    schema_from_typed_model = codom(typed_model)

    # Create schema directly
    schema_direct = create_schema(one_pop_schema)

    # Both should be LabelledPetriNets
    @test schema_from_typed_model isa LabelledPetriNet
    @test schema_direct isa LabelledPetriNet

    # They should be equal in structure
    @test schema_from_typed_model == schema_direct
end
