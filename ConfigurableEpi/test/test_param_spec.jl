using Test
using ConfigurableEpi

@testset "ParamSpec Types" begin
    @testset "FixedParam" begin
        fp = FixedParam(:gamma, 0.1)
        @test fp isa ParamSpec
        @test fp isa FixedParam{Float64}
        @test fp.name == :gamma
        @test fp.value == 0.1

        # Integer value
        fp_int = FixedParam(:count, 5)
        @test fp_int isa FixedParam{Int}
        @test fp_int.value == 5
    end

    @testset "HyperParam" begin
        hp = HyperParam(unconstrained_gaussian(:R0, 1.0, 0.5))
        @test hp isa ParamSpec
        @test hp isa HyperParam
        @test hp.name == :R0
        @test hp.prior isa ParameterDistribution
    end

    @testset "AR1ParamSpec - All Fixed" begin
        ar1_fixed = AR1ParamSpec(:Rt_mod; init = unconstrained_gaussian(:Rt_mod, 0.0, 1.0), mu = 0.0, tau = 9.491221581029905, sigma = 0.22941573387056183)
        @test ar1_fixed isa ParamSpec
        @test ar1_fixed isa AR1ParamSpec
        @test ar1_fixed.name == :Rt_mod
        @test ar1_fixed.init isa ParameterDistribution

        # Child params are all FixedParam
        @test ar1_fixed.mu isa FixedParam
        @test ar1_fixed.tau isa FixedParam
        @test ar1_fixed.sigma isa FixedParam

        # Check child names
        @test ar1_fixed.mu.name == :Rt_mod_mu
        @test ar1_fixed.tau.name == :Rt_mod_tau
        @test ar1_fixed.sigma.name == :Rt_mod_sigma_stat

        # Check values
        @test ar1_fixed.mu.value == 0.0
        @test ar1_fixed.tau.value ≈ -1 / log(0.9)   # tau, not rho
        @test ar1_fixed.sigma.value ≈ 0.1 / sqrt(1 - 0.9^2)   # STATIONARY sd, not innovation
    end

    @testset "AR1ParamSpec - All Hyper" begin
        ar1_hyper = AR1ParamSpec(:Rt_mod; init = unconstrained_gaussian(:Rt_mod, 0.0, 1.0), mu = unconstrained_gaussian(:Rt_mod_mu, 0.0, 1.0), tau = positive_gaussian(:Rt_mod_tau, 9.5, 2.0), sigma = positive_gaussian(:Rt_mod_sigma_stat, 0.23, 0.05))
        @test ar1_hyper isa AR1ParamSpec

        # Child params are HyperParam
        @test ar1_hyper.mu isa HyperParam
        @test ar1_hyper.tau isa HyperParam
        @test ar1_hyper.sigma isa HyperParam

        # Check child names
        @test ar1_hyper.mu.name == :Rt_mod_mu
        @test ar1_hyper.tau.name == :Rt_mod_tau
        @test ar1_hyper.sigma.name == :Rt_mod_sigma_stat
    end

    @testset "AR1ParamSpec - Mixed: Fixed mu" begin
        ar1_mixed = AR1ParamSpec(:Rt_mod; init = unconstrained_gaussian(:Rt_mod, 0.0, 1.0), mu = 0.0, tau = positive_gaussian(:Rt_mod_tau, 9.5, 2.0), sigma = positive_gaussian(:Rt_mod_sigma_stat, 0.23, 0.05))
        @test ar1_mixed isa AR1ParamSpec

        @test ar1_mixed.mu isa FixedParam
        @test ar1_mixed.tau isa HyperParam
        @test ar1_mixed.sigma isa HyperParam

        @test ar1_mixed.mu.value == 0.0
    end

    @testset "AR1ParamSpec - Mixed: Fixed sigma" begin
        ar1_mixed = AR1ParamSpec(:Rt_mod; init = unconstrained_gaussian(:Rt_mod, 0.0, 1.0), mu = unconstrained_gaussian(:Rt_mod_mu, 0.0, 1.0), tau = positive_gaussian(:Rt_mod_tau, 9.5, 2.0), sigma = 0.1)
        @test ar1_mixed isa AR1ParamSpec

        @test ar1_mixed.mu isa HyperParam
        @test ar1_mixed.tau isa HyperParam
        @test ar1_mixed.sigma isa FixedParam

        @test ar1_mixed.sigma.value == 0.1
    end

    @testset "AR1ParamSpec with explicit init prior" begin
        init_prior = unit_interval_gaussian(:rate, 0.5, 0.1)
        ar1_ui = AR1ParamSpec(:rate; init = init_prior, mu = 0.5, tau = 9.491221581029905, sigma = 0.22941573387056183)
        @test ar1_ui.init === init_prior
    end

    @testset "RWParamSpec - Fixed sigma" begin
        rw_fixed = RWParamSpec(:trend; init = unconstrained_gaussian(:trend, 0.0, 1.0), sigma_rate = 0.05)
        @test rw_fixed isa ParamSpec
        @test rw_fixed isa RWParamSpec
        @test rw_fixed.name == :trend
        @test rw_fixed.init isa ParameterDistribution

        @test rw_fixed.sigma isa FixedParam
        @test rw_fixed.sigma.name == :trend_sigma_rate
        @test rw_fixed.sigma.value == 0.05
    end

    @testset "RWParamSpec - Hyper sigma" begin
        rw_hyper = RWParamSpec(:trend; init = unconstrained_gaussian(:trend, 0.0, 1.0), sigma_rate = positive_gaussian(:trend_sigma, 0.1, 0.05))
        @test rw_hyper isa RWParamSpec

        @test rw_hyper.sigma isa HyperParam
        @test rw_hyper.sigma.name == :trend_sigma
    end

    @testset "RWParamSpec with explicit init prior" begin
        init_prior = positive_gaussian(:rate, 1.0, 0.25)
        rw_positive = RWParamSpec(:rate; init = init_prior, sigma_rate = 0.05)
        @test rw_positive.init === init_prior
    end

    @testset "DerivedParam" begin
        dp = DerivedParam(:beta, θ -> θ[:R0] * θ[:gamma])
        @test dp isa ParamSpec
        @test dp isa DerivedParam
        @test dp.name == :beta
        @test dp.formula isa Function

        # Test the formula works
        mock_θ = (R0 = 2.5, gamma = 0.1)
        @test dp.formula(mock_θ) ≈ 0.25
    end

    @testset "Type Hierarchy" begin
        # All types are ParamSpec
        @test FixedParam <: ParamSpec
        @test HyperParam <: ParamSpec
        @test AR1ParamSpec <: ParamSpec
        @test RWParamSpec <: ParamSpec
        @test DerivedParam <: ParamSpec
    end

    @testset "AR1ParamSpec Type Parameters" begin
        # Type parameters encode the inference structure
        ar1_all_fixed = AR1ParamSpec(:x; init = unconstrained_gaussian(:x, 0.0, 1.0), mu = 0.0, tau = 9.491221581029905, sigma = 0.22941573387056183)
        ar1_all_hyper = AR1ParamSpec(:x; init = unconstrained_gaussian(:x, 0.0, 1.0), mu = unconstrained_gaussian(:x_mu, 0.0, 1.0), tau = positive_gaussian(:x_tau, 9.5, 2.0), sigma = positive_gaussian(:x_sigma_stat, 0.23, 0.05))

        # Different type parameters
        @test typeof(ar1_all_fixed) != typeof(ar1_all_hyper)

        # Both are AR1ParamSpec
        @test ar1_all_fixed isa AR1ParamSpec
        @test ar1_all_hyper isa AR1ParamSpec
    end

end

@testset "RK4 stability recommendation advances an exact-boundary configuration" begin
    durations = Durations(latent = 1.0, infectious = 1.0)
    err = try
        assert_integration_stable(
            durations, 1, 1, RK4_STABILITY_LIMIT, 1; R_eff_max = 0.0
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("at least 2 (currently 1)", sprint(showerror, err))
    @test assert_integration_stable(
        durations, 1, 1, RK4_STABILITY_LIMIT, 2; R_eff_max = 0.0
    ) < RK4_STABILITY_LIMIT
end
