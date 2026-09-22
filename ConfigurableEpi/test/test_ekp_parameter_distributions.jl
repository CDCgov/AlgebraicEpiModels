using Test
using ConfigurableEpi
using Distributions: logpdf

@testset "EKP Parameter Distribution Helpers" begin
    @testset "prior_name" begin
        prior = positive_gaussian(:sigma, 0.1, 0.05)
        @test ConfigurableEpi.prior_name(prior) == :sigma

        named_with_string = unconstrained_gaussian("beta", 0.3, 0.1)
        @test ConfigurableEpi.prior_name(named_with_string) == :beta

        bundle = ParameterPriorBundle(
            positive_gaussian(:sigma, 0.1, 0.05),
            unit_interval_gaussian(:rho, 0.9, 0.05)
        )
        @test_throws ArgumentError ConfigurableEpi.prior_name(bundle.prior)
    end

    @testset "scalar prior constructors" begin
        positive = positive_gaussian(:sigma, 0.1, 0.05)
        unit = unit_interval_gaussian(:rho, 0.9, 0.05)
        unc = unconstrained_gaussian(:beta, 0.0, 1.0)

        @test positive isa ParameterDistribution
        @test unit isa ParameterDistribution
        @test unc isa ParameterDistribution

        @test ConfigurableEpi.prior_name(positive) == :sigma
        @test ConfigurableEpi.prior_name(unit) == :rho
        @test ConfigurableEpi.prior_name(unc) == :beta
    end

    @testset "ParameterPriorBundle constructors" begin
        sigma = positive_gaussian(:sigma, 0.1, 0.05)
        rho = unit_interval_gaussian(:rho, 0.9, 0.05)

        bundle = ParameterPriorBundle(sigma, rho)
        @test bundle.names == (:sigma, :rho)
        @test bundle.prior isa ParameterDistribution

        bundle_from_nt = ParameterPriorBundle((; sigma, rho))
        @test bundle_from_nt.names == (:sigma, :rho)
    end

    @testset "roundtrip constrained/unconstrained values" begin
        bundle = ParameterPriorBundle(
            positive_gaussian(:sigma, 0.1, 0.05),
            unit_interval_gaussian(:rho, 0.9, 0.05),
            unconstrained_gaussian(:beta, 0.0, 1.0)
        )

        constrained = (sigma = 0.12, rho = 0.85, beta = -0.4)
        unconstrained = unconstrained_values(bundle, constrained)
        roundtrip = constrained_values(bundle, unconstrained)

        @test length(unconstrained) == 3
        @test roundtrip.sigma ≈ constrained.sigma
        @test roundtrip.rho ≈ constrained.rho
        @test roundtrip.beta ≈ constrained.beta
    end

    @testset "boundary rejection" begin
        bundle = ParameterPriorBundle(unit_interval_gaussian(:rho, 0.5, 0.1))
        @test_throws ArgumentError unconstrained_values(bundle, (rho = 0.0,))
        @test_throws ArgumentError unconstrained_values(bundle, (rho = 1.0,))
    end

    @testset "prior_logpdf" begin
        bundle = ParameterPriorBundle(
            positive_gaussian(:sigma, 0.1, 0.05),
            unit_interval_gaussian(:rho, 0.9, 0.05)
        )

        unconstrained = unconstrained_values(bundle, (sigma = 0.12, rho = 0.85))
        expected = let values = logpdf(bundle.prior, collect(unconstrained))
            values isa Real ? values : sum(values)
        end

        @test prior_logpdf(bundle, unconstrained) ≈ expected
        @test isfinite(prior_logpdf(bundle, unconstrained))
    end
end
