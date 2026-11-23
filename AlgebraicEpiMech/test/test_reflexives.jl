@testitem "default_reflexives per model type" setup = [ModelSetup] begin
    one, split = OnePopulationTyping(), UninfectedInfectedTyping()
    @test default_reflexives(one, SIR()) == Symbol[]
    @test default_reflexives(one, NoCrossImmunity([:x, :y])) == Symbol[]
    @test default_reflexives(one, AgeStratification([:child, :adult])) == [:disease, :reversion, :waning]
    @test default_reflexives(split, ImmuneHistory([:a])) == [:disease, :waning]
end

@testitem "compartmental models add passive reflexives only on request" setup = [ModelSetup] begin
    one, split = OnePopulationTyping(), UninfectedInfectedTyping()
    names(typing, include) = tnames(dom(create_model(typing, SIR(); include_reflexives = include)))

    @test tnames(dom(create_model(one, SIR()))) == names(one, true) == names(one, false) ==
        [:transmission_S_I, :I_to_R]
    @test names(one, (:waning,)) == names(one, :waning) == names(one, [:waning, :waning]) ==
        [:transmission_S_I, :I_to_R, :waning_S, :waning_I, :waning_R]
    # Under the split typing :waning is Uninfected → Uninfected, so only S carries one.
    @test names(split, (:waning,)) == [:transmission_S_I, :I_to_R, :waning_S]
    @test names(one, (:strata,)) == names(one, :strata) == names(one, [:strata, :strata]) ==
        [:transmission_S_I, :I_to_R, :S, :I, :R]
    # :strata expands to both movement types under the split typing; other types are prefixed.
    split_sir = create_model(split, SIR(); include_reflexives = (:strata, :waning))
    @test tnames(dom(split_sir)) == [:transmission_S_I, :I_to_R, :S, :I, :R, :waning_S]
    @test [tname(codom(split_sir), split_sir[:T](t)) for t in 1:6] ==
        [:transmission, :disease, :strata_U, :strata_I, :strata_I, :waning]
    # Types the factor implements are skipped.
    @test names(one, (:disease, :transmission)) == [:transmission_S_I, :I_to_R]
    @test length(names(one, (:reversion,))) == 5

    @test_throws ArgumentError names(one, (:nope,))
    @test_throws ArgumentError names(one, ("waning",))
    # :reversion is Infected → Uninfected under the split typing, so it cannot be reflexive.
    @test_throws ArgumentError names(split, (:reversion,))
end

@testitem "stratification factors select their reflexive boxes" setup = [ModelSetup] begin
    one, split = OnePopulationTyping(), UninfectedInfectedTyping()
    age = AgeStratification([:child, :adult])
    names(typing, include) = tnames(dom(create_model(typing, age; include_reflexives = include)))

    @test nt(dom(create_model(one, age))) == length(names(one, true)) == 10
    @test names(one, false) == [:child_child, :child_adult, :adult_child, :adult_adult]
    @test names(one, (:disease,)) == [:child_child, :child_adult, :adult_child, :adult_adult, :child, :adult]
    @test names(split, (:waning,)) ==
        [:child_U_child_I, :child_U_adult_I, :adult_U_child_I, :adult_U_adult_I, :child_U, :adult_U]
    @test create_model_uwd(one, age; include_reflexives = false) |> boxes |> length == 4
    # A static factor is passive for movement on request, one box per population type.
    @test names(one, (:strata,)) == [:child_child, :child_adult, :adult_child, :adult_adult, :child, :adult]
    @test names(split, (:strata,)) == [
        :child_U_child_I, :child_U_adult_I, :adult_U_child_I, :adult_U_adult_I,
        :child_U, :adult_U, :child_I, :adult_I,
    ]
    # A moving factor implements movement, so listing :strata adds nothing.
    aging = AgeStratification([:child, :adult]; transitions = [:child => :adult])
    @test nt(dom(create_model(one, aging; include_reflexives = (default_reflexives(one, aging)..., :strata)))) ==
        nt(dom(create_model(one, aging))) == 11
end

@testitem "immune history adds disease and waning reflexives when listed" setup = [ModelSetup] begin
    split = UninfectedInfectedTyping()
    history = ImmuneHistory([:a, :b])
    count(include) = nt(dom(create_model(split, history; include_reflexives = include)))
    # FullHistory over two strains: four (class, strain) pairs and four classes.
    @test count(true) == count(false) + 8
    @test count((:disease,)) == count((:waning,)) == count(false) + 4
    @test nt(dom(create_model(split, history))) == count(true)
end
