using Test, ConfigurableEpi, LinearAlgebra
using ForwardDiff

@testset "UKF covariance factor preserves state units" begin
    # Compartment counts and log-chart latent coefficients have very different units.
    # A valid covariance must survive factorization in every coordinate, including
    # a pinned level variance and its small cross-covariances.
    scale = Diagonal([1.0e7, 0.01, 1.0])
    correlation = [1.0 0.2 0.03; 0.2 1.0 0.15; 0.03 0.15 1.0]
    covariance = Matrix(scale * correlation * scale)
    original = copy(covariance)
    factor = ConfigurableEpi.positive_cholesky!(covariance)
    reconstructed = Matrix(factor.L * factor.L')
    @test covariance == original
    for index in CartesianIndices(original)
        @test reconstructed[index] ≈ original[index] rtol = 1.0e-12
    end

    # The same helper supports differentiable UKF likelihoods.
    for initial_variance in (1.0e-4, 1.0e-24)
        derivative = ForwardDiff.derivative(initial_variance) do variance
            matrix = [oftype(variance, 1.0e14) zero(variance); zero(variance) variance]
            root = ConfigurableEpi.positive_cholesky!(matrix)
            return (root.L * root.L')[2, 2]
        end
        @test derivative ≈ 1.0 rtol = 1.0e-12
    end

    # A positive variance is not a zero pivot, even below the compartment floor.
    tiny = ConfigurableEpi.positive_cholesky!(Diagonal([1.0e14, 1.0e-24]))
    @test (tiny.L * tiny.L')[2, 2] ≈ 1.0e-24 rtol = 1.0e-12
    zero_variance = ConfigurableEpi.positive_cholesky!(Diagonal([1.0e14, 0.0]))
    @test (zero_variance.L * zero_variance.L')[2, 2] ≈ ConfigurableEpi.MIN_MODEL_VARIANCE

    # Repairing an indefinite covariance must also be independent of state units.
    indefinite = [1.0 1.1; 1.1 1.0]
    units = Diagonal([1.0e6, 0.01])
    repaired = ConfigurableEpi.positive_cholesky!(indefinite)
    scaled_repair = ConfigurableEpi.positive_cholesky!(units * indefinite * units)
    restored = units \ Matrix(scaled_repair.L * scaled_repair.L') / units
    @test isposdef(Matrix(scaled_repair.L * scaled_repair.L'))
    @test restored ≈ Matrix(repaired.L * repaired.L')
end
