using Test
using LabelledArrays
using AlgebraicPetri: LabelledPetriNet
using ConfigurableEpi
import AlgebraicEpiMech

@testset "Core Dynamics" begin
    @testset "make_lvector_constructor" begin
        to_lvec = make_lvector_constructor((:S, :I, :R))
        x = to_lvec([990.0, 10.0, 0.0])

        @test x.S ≈ 990.0
        @test x[:I] ≈ 10.0
        @test length(x) == 3
    end

    @testset "build_unified_vf" begin
        function mock_petri_vf!(du, u, p, t)
            infection = p[:beta] * u[:S] * u[:I]
            recovery = p[:gamma] * u[:I]
            obs_flow = p[:gamma] * u[:I] - p[:delta] * u[:O_I_1]

            du[:S] = -infection
            du[:I] = infection - recovery
            du[:R] = recovery
            du[:O_I_1] = obs_flow
            du[:O_I_2] = p[:delta] * u[:O_I_1]

            return nothing
        end

        layout = StateLayout((:S, :I, :R), (:O_I_1, :O_I_2), ())
        vf = build_unified_vf(mock_petri_vf!, layout)

        @test vf isa Function

        x = [990.0, 10.0, 0.0, 20.0, 50.0]
        p = LVector(beta = 0.001, gamma = 0.1, delta = 0.5)

        dx = vf(x, nothing, p, 0.0)

        @test length(dx) == 5
        @test dx[1] ≈ -9.9
        @test dx[2] ≈ 8.9
        @test dx[3] ≈ 1.0
        @test dx[4] ≈ 0.1 * 10.0 - 0.5 * 20.0
        @test dx[5] ≈ 0.5 * 20.0
        @test eltype(dx) == Float64
    end

    @testset "build_unified_vf handles zero states" begin
        function mock_petri_vf!(du, u, p, t)
            du[:S] = -p[:beta] * u[:S] * u[:I]
            du[:I] = p[:beta] * u[:S] * u[:I]
            du[:O_I_1] = p[:gamma] * u[:I]
            return nothing
        end

        layout = StateLayout((:S, :I), (:O_I_1,), ())
        vf = build_unified_vf(mock_petri_vf!, layout)

        x = [1000.0, 0.0, 0.0]
        p = LVector(beta = 0.001, gamma = 0.1)

        dx = vf(x, nothing, p, 0.0)

        @test all(dx .≈ 0.0)
    end

    @testset "build_petri_vf defaults and override" begin
        pn = LabelledPetriNet(
            [:S, :I, :R],
            :infection => ((:S, :I) => (:I, :I)),
            :recovery => (:I => :R),
        )
        empty_rates(latent, hyper, t) = (;)

        @testset "defaults supply fixed transition rates" begin
            # rates() is empty; both transitions come from the defaults.
            vf! = build_petri_vf(pn, empty_rates; defaults = (infection = 0.0, recovery = 0.5))
            du = LVector(S = 0.0, I = 0.0, R = 0.0)
            u = LVector(S = 990.0, I = 10.0, R = 0.0)
            vf!(du, u, ((;), (;)), 0.0)

            @test du.S ≈ 0.0          # infection rate 0
            @test du.I ≈ -5.0         # recovery 0.5 * 10 out of I
            @test du.R ≈ 5.0          # recovery into R
        end

        @testset "rates() overrides a default" begin
            override(latent, hyper, t) = (recovery = hyper.gamma,)
            vf! = build_petri_vf(pn, override; defaults = (infection = 0.0, recovery = 0.5))
            du = LVector(S = 0.0, I = 0.0, R = 0.0)
            u = LVector(S = 990.0, I = 10.0, R = 0.0)
            vf!(du, u, ((gamma = 0.1,), (;)), 0.0)

            @test du.R ≈ 1.0          # 0.1 * 10 from rates(), not the 0.5 default
        end

        @testset "unknown default key throws" begin
            @test_throws ArgumentError build_petri_vf(
                pn, empty_rates; defaults = (infection = 0.1, nonsense = 0.2)
            )
        end
    end

    @testset "build_petri_vf agrees with the flat field" begin
        # `build_petri_vf` as it was before the factored field, kept as the reference.
        function flat_petri_vf(pn, rates; defaults = (;))
            vf! = AlgebraicEpiMech.vectorfield_flat(pn)
            return (du, u, p, t) -> vf!(du, u, merge(defaults, rates(p[2], p[1], t)), t)
        end
        pn = LabelledPetriNet(
            [:S, :I1, :I2, :R],
            :infection1 => ((:S, :I1) => (:I1, :I1)),
            :infection2 => ((:S, :I2) => (:I1, :I2)),
            :progression => (:I1 => :I2),
            :recovery => (:I2 => :R),
        )
        defaults = (progression = 0.4, recovery = 0.5)
        per_transition(latent, hyper, t) = (infection1 = hyper.beta * latent.Rt, infection2 = 0.5 * hyper.beta * latent.Rt)
        per_key(latent, hyper, t) = (beta = hyper.beta * latent.Rt,)
        rate_class(name) = name == :infection1 ? (:beta, 1.0) : name == :infection2 ? (:beta, 0.5) : (name, 1.0)

        u = LVector(S = 990.0, I1 = 7.0, I2 = 3.0, R = 0.0)
        p = ((beta = 0.001,), (Rt = 1.2,))
        du_reference = LVector(S = 0.0, I1 = 0.0, I2 = 0.0, R = 0.0)
        flat_petri_vf(pn, per_transition; defaults)(du_reference, u, p, 0.0)
        function evaluate(vf!, u, p)
            out = LVector(S = NaN, I1 = NaN, I2 = NaN, R = NaN)
            vf!(out, u, p, 0.0)
            return out, @allocated(vf!(out, u, p, 0.0))
        end

        du, allocations = evaluate(build_petri_vf(pn, per_transition; defaults), u, p)
        @test du ≈ du_reference rtol = 1.0e-13
        @test allocations == 0

        du, allocations = evaluate(build_petri_vf(pn, per_key; rate_class, defaults), u, p)
        @test du ≈ du_reference rtol = 1.0e-13
        @test allocations == 0

        # With rate classes the defaults are keyed by rate key, not by transition.
        @test_throws ArgumentError build_petri_vf(
            pn, per_key; rate_class, defaults = (infection1 = 0.2,)
        )
    end
end
