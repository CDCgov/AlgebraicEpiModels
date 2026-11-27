# The EnKF engine's filter: LowLevelParticleFilters' `EnsembleKalmanFilter{true, true}` with the
# engine's measurement wrapper. On a linear-Gaussian system the augmented form
#
#     x⁺ = A x + G w,   w ~ N(0, R1)
#     y  = C x + H v,   v ~ N(0, R2)
#
# is the additive system with covariances `G R1 Gᵀ` and `H R2 Hᵀ`, so the filter must reproduce
# the exact Kalman filter on it within sampling error. A double-counted observation noise would
# show as a too-low log-likelihood and an over-wide filtered covariance. `G` is 3×2, so `nw != nx`.

using ConfigurableEpi
using LowLevelParticleFilters: KalmanFilter, EnsembleKalmanFilter, forward_trajectory
using Distributions: MvNormal
using LinearAlgebra: I, norm
using Random
using Test

const CE = ConfigurableEpi
const _AE_A = [0.9 0.1 0.0; 0.0 0.85 0.05; 0.0 0.0 0.8]
const _AE_G = [1.0 0.0; 0.0 1.0; 0.5 0.5]
const _AE_C = [1.0 0.0 0.0; 0.0 1.0 1.0]
const _AE_H = [0.7 0.0; 0.0 0.4]

_ae_dynamics(x, u, p, t, w) = _AE_A * x .+ _AE_G * w
_ae_measurement(x, u, p, t, v) = _AE_C * x .+ _AE_H * v
_ae_d0() = MvNormal(zeros(3), Matrix(1.0I, 3, 3))

function _ae_simulate(T::Int; seed = 20260803)
    rng = MersenneTwister(seed)
    x = rand(rng, _ae_d0())
    us = [Float64[] for _ in 1:T]
    ys, xs = Vector{Vector{Float64}}(undef, T), Vector{Vector{Float64}}(undef, T)
    for k in 1:T
        xs[k] = copy(x)
        ys[k] = _ae_measurement(x, us[k], nothing, float(k - 1), randn(rng, 2))
        x = _ae_dynamics(x, us[k], nothing, float(k - 1), randn(rng, 2))
    end
    return us, ys, xs
end

_ae_enkf(N; seed = 4242) = EnsembleKalmanFilter{true, true}(
    _ae_dynamics, CE._ObservedComponents(_ae_measurement), Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2), _ae_d0(), N;
    nu = 0, ny = 2, rng = MersenneTwister(seed),
)

_ae_sse(means, truth) = sum(norm(m .- x)^2 for (m, x) in zip(means, truth))

@testset "EnKF" begin
    @testset "EnKF config" begin
        @test EnKF(n_ensemble = 200).inflation == 1.0
        @test !EnKF(n_ensemble = 200).threads
        @test_throws ArgumentError CE._validate(EnKF(n_ensemble = 1))
        @test_throws ArgumentError CE._validate(EnKF(n_ensemble = 10, inflation = 0.99))
    end

    @testset "agrees with the exact Kalman filter (nw != nx)" begin
        us, ys, xs = _ae_simulate(60)
        exact = KalmanFilter(_AE_A, zeros(3, 0), _AE_C, zeros(2, 0), _AE_G * _AE_G', _AE_H * _AE_H', _ae_d0(); check = false)
        sol_kf = forward_trajectory(exact, us, ys)
        sol_en = forward_trajectory(_ae_enkf(4000), us, ys)
        @test sol_en.ll ≈ sol_kf.ll atol = 5.0
        @test _ae_sse(sol_en.xt, xs) < 1.15 * _ae_sse(sol_kf.xt, xs)
        @test norm(sol_en.Rt[end] - sol_kf.Rt[end]) / norm(sol_kf.Rt[end]) < 0.15
        @test norm(Matrix(sol_en.S[end]) - Matrix(sol_kf.S[end])) / norm(Matrix(sol_kf.S[end])) < 0.15
    end

    @testset "marginal_loglik is the filter's log-likelihood" begin
        us, ys, _ = _ae_simulate(30)
        @test marginal_loglik(_ae_enkf(500), ys, nothing) ≈ forward_trajectory(_ae_enkf(500), us, ys).ll
    end

    @testset "re-seeding repeats a run, which common random numbers rely on" begin
        _, ys, _ = _ae_simulate(20)
        @test marginal_loglik(_ae_enkf(300; seed = 7), ys, nothing) == marginal_loglik(_ae_enkf(300; seed = 7), ys, nothing)
        @test marginal_loglik(_ae_enkf(300; seed = 7), ys, nothing) != marginal_loglik(_ae_enkf(300; seed = 8), ys, nothing)
        # `reset!` draws from the filter's rng, so only re-seeding repeats a filter's run.
        f = _ae_enkf(300; seed = 7)
        @test marginal_loglik(f, ys, nothing) != marginal_loglik(f, ys, nothing)
        Random.seed!(f.rng, 123)
        first_run = marginal_loglik(f, ys, nothing)
        Random.seed!(f.rng, 123)
        @test marginal_loglik(f, ys, nothing) == first_run
    end
end
