# ============================================================================
# Augmented-noise ensemble Kalman filter (src/inference/augmented_enkf.jl)
# ============================================================================
#
# The reference is the exact Kalman filter. On a linear-Gaussian system the augmented form
#
#     x⁺ = A x + G w,   w ~ N(0, R1)
#     y  = C x + H v,   v ~ N(0, R2)
#
# is the additive system with effective covariances `G R1 Gᵀ` and `H R2 Hᵀ`, so the ensemble
# filter must reproduce `LowLevelParticleFilters.KalmanFilter` on that system within sampling
# error. That is the only test that can distinguish a correct augmented EnKF from one that
# double-counts the observation noise: adding `R2` to the innovation covariance *and* drawing a
# perturbation would inflate `S`, which shows up as a too-low log-likelihood and an over-wide
# filtered covariance rather than as an outright failure.
#
# `G` is deliberately 3×2 and `H` deliberately non-square, so `nw != nx` and `nv != ny` are
# exercised by the main agreement test rather than only by a shape smoke.

using ConfigurableEpi
using LowLevelParticleFilters: KalmanFilter, EnsembleKalmanFilter, KalmanFilteringSolution,
    forward_trajectory, predict!, correct!, reset!, state, covariance, num_particles,
    particles, particletype
using Distributions: MvNormal
using LinearAlgebra: I, norm, diag
using Random
using Statistics: mean
using Test

# --- the shared linear-Gaussian test system ----------------------------------------
const _AE_A = [0.9 0.1 0.0; 0.0 0.85 0.05; 0.0 0.0 0.8]
const _AE_G = [1.0 0.0; 0.0 1.0; 0.5 0.5]      # 3x2 => nw = 2 != nx = 3
const _AE_C = [1.0 0.0 0.0; 0.0 1.0 1.0]       # 2x3
const _AE_H = [0.7 0.0; 0.0 0.4]                 # 2x2 => nv = 2 == ny
const _AE_NX, _AE_NW, _AE_NY, _AE_NV = 3, 2, 2, 2

_ae_dynamics(x, u, p, t, w) = _AE_A * x .+ _AE_G * w
_ae_measurement(x, u, p, t, v) = _AE_C * x .+ _AE_H * v

_ae_d0() = MvNormal(zeros(_AE_NX), Matrix(1.0I, _AE_NX, _AE_NX))

"""Simulate `T` steps of the augmented system; return `(us, ys, xs)`."""
function _ae_simulate(T::Int; seed = 20260803)
    rng = MersenneTwister(seed)
    x = rand(rng, _ae_d0())
    us = [Float64[] for _ in 1:T]
    ys = Vector{Vector{Float64}}(undef, T)
    xs = Vector{Vector{Float64}}(undef, T)
    for k in 1:T
        xs[k] = copy(x)
        ys[k] = _ae_measurement(x, us[k], nothing, float(k - 1), randn(rng, _AE_NV))
        x = _ae_dynamics(x, us[k], nothing, float(k - 1), randn(rng, _AE_NW))
    end
    return us, ys, xs
end

_ae_enkf(
    N; seed = 4242, threads = false, inflation = 1.0, R1 = Matrix(1.0I, _AE_NW, _AE_NW),
    R2 = Matrix(1.0I, _AE_NV, _AE_NV), measurement = _ae_measurement, ny = _AE_NY
) =
    AugmentedEnsembleKalmanFilter(
    _ae_dynamics, measurement, R1, R2, _ae_d0(), N;
    nu = 0, ny = ny, rng = MersenneTwister(seed), threads = threads, inflation = inflation,
)

"""The exact filter for the equivalent additive system."""
_ae_exact_kf(; H = _AE_H, R2 = Matrix(1.0I, _AE_NV, _AE_NV)) = KalmanFilter(
    _AE_A, zeros(_AE_NX, 0), _AE_C, zeros(_AE_NY, 0),
    _AE_G * Matrix(1.0I, _AE_NW, _AE_NW) * _AE_G', H * R2 * H', _ae_d0();
    check = false,
)

_ae_sse(means, truth) = sum(norm(m .- x)^2 for (m, x) in zip(means, truth))

@testset "AugmentedEnsembleKalmanFilter" begin

    @testset "construction and accessors" begin
        f = _ae_enkf(64)
        @test f isa AugmentedEnsembleKalmanFilter
        @test num_particles(f) == 64
        @test length(particles(f)) == 64
        @test (f.nx, f.nw, f.ny, f.nv) == (_AE_NX, _AE_NW, _AE_NY, _AE_NV)
        @test f.nu == 0
        @test length(state(f)) == _AE_NX
        @test size(covariance(f)) == (_AE_NX, _AE_NX)
        @test particletype(f) <: AbstractVector
        # Field names `forward_trajectory` reaches for directly.
        @test f.Ts == 1.0
        @test size(f.R1) == (_AE_NW, _AE_NW)
        @test size(f.R2) == (_AE_NV, _AE_NV)
        @test f.names.name == "AugmentedEnKF"

        R1, R2 = Matrix(1.0I, _AE_NW, _AE_NW), Matrix(1.0I, _AE_NV, _AE_NV)
        @test_throws ArgumentError AugmentedEnsembleKalmanFilter(
            _ae_dynamics, _ae_measurement, R1, R2, _ae_d0(), 1; nu = 0, ny = _AE_NY,
        )
        @test_throws ArgumentError AugmentedEnsembleKalmanFilter(
            _ae_dynamics, _ae_measurement, zeros(2, 3), R2, _ae_d0(), 8;
            nu = 0, ny = _AE_NY,
        )
        # Below 1.0 shrinks the ensemble every step, which is filter divergence by design.
        @test_throws ArgumentError AugmentedEnsembleKalmanFilter(
            _ae_dynamics, _ae_measurement, R1, R2, _ae_d0(), 8;
            nu = 0, ny = _AE_NY, inflation = 0.9,
        )
    end

    @testset "EnKF config" begin
        @test EnKF(n_ensemble = 200).inflation == 1.0
        @test !EnKF(n_ensemble = 200).threads
        @test_throws ArgumentError ConfigurableEpi._validate(EnKF(n_ensemble = 1))
        @test_throws ArgumentError ConfigurableEpi._validate(EnKF(n_ensemble = 10, inflation = 0.99))
    end

    @testset "agrees with the exact Kalman filter (nw != nx)" begin
        T = 60
        us, ys, xs = _ae_simulate(T)

        sol_kf = forward_trajectory(_ae_exact_kf(), us, ys)
        sol_en = forward_trajectory(_ae_enkf(4000), us, ys)

        @test isfinite(sol_en.ll)
        @test sol_en.ll ≈ sol_kf.ll atol = 5.0
        # Filtered means track the truth as well as the exact filter does.
        @test _ae_sse(sol_en.xt, xs) < 1.15 * _ae_sse(sol_kf.xt, xs)
        # Steady-state covariance matches. This is what a double-counted R2 would break:
        # inflating S widens the posterior and shrinks the gain.
        @test norm(sol_en.Rt[end] - sol_kf.Rt[end]) / norm(sol_kf.Rt[end]) < 0.15
        # The innovation covariance is the noisy-predicted-observation covariance, i.e. it
        # already contains H R2 Hᵀ exactly once.
        S_en = Matrix(sol_en.S[end])
        S_kf = Matrix(sol_kf.S[end])
        @test norm(S_en - S_kf) / norm(S_kf) < 0.15
    end

    @testset "nv != ny (rank-deficient measurement noise)" begin
        H = reshape([0.7, 0.35], 2, 1)         # 2x1 => nv = 1 != ny = 2
        measurement = (x, u, p, t, v) -> _AE_C * x .+ H * v
        T = 40
        us, ys, xs = _ae_simulate(T)

        f = _ae_enkf(
            3000; R2 = Matrix(1.0I, 1, 1), measurement = measurement, ny = _AE_NY,
        )
        @test f.nv == 1
        @test f.ny == 2
        sol_en = forward_trajectory(f, us, ys)
        sol_kf = forward_trajectory(
            _ae_exact_kf(; H = H, R2 = Matrix(1.0I, 1, 1)), us, ys,
        )

        @test isfinite(sol_en.ll)
        @test all(all(isfinite, x) for x in sol_en.xt)
        @test sol_en.ll ≈ sol_kf.ll atol = 5.0
        @test _ae_sse(sol_en.xt, xs) < 1.15 * _ae_sse(sol_kf.xt, xs)
    end

    @testset "forward_trajectory returns a standard KalmanFilteringSolution" begin
        T = 12
        us, ys, _ = _ae_simulate(T)
        sol = forward_trajectory(_ae_enkf(200), us, ys)

        @test sol isa KalmanFilteringSolution
        @test length(sol.x) == T
        @test length(sol.xt) == T
        @test length(sol.R) == T
        @test length(sol.Rt) == T
        @test length(sol.e) == T
        @test length(sol.K) == T
        @test length(sol.S) == T
        @test length(sol.t) == T
        @test all(length(x) == _AE_NX for x in sol.x)
        @test all(size(R) == (_AE_NX, _AE_NX) for R in sol.Rt)
        @test all(size(K) == (_AE_NX, _AE_NY) for K in sol.K)
        @test isfinite(sol.ll)
        # `Base.show` reads `sol.f.names.name`, so the SignalNames field must be populated.
        @test occursin("AugmentedEnKF", sprint(show, sol))
    end

    @testset "marginal_loglik agrees with forward_trajectory(...).ll" begin
        T = 30
        us, ys, _ = _ae_simulate(T)
        # Both filters are constructed with the same seed and both reset! internally, so they
        # consume the same RNG stream. This is the invariant the EKP objective relies on.
        ll_direct = forward_trajectory(_ae_enkf(500), us, ys).ll
        ll_marg = marginal_loglik(_ae_enkf(500), ys, nothing)
        @test ll_marg ≈ ll_direct
    end

    @testset "seeded runs are reproducible; re-seeding is what makes them so" begin
        T = 20
        us, ys, _ = _ae_simulate(T)

        # Same seed, fresh filter => identical.
        @test forward_trajectory(_ae_enkf(300; seed = 7), us, ys).ll ==
            forward_trajectory(_ae_enkf(300; seed = 7), us, ys).ll
        # Different seed => different, so the equality above is not vacuous.
        @test forward_trajectory(_ae_enkf(300; seed = 7), us, ys).ll !=
            forward_trajectory(_ae_enkf(300; seed = 8), us, ys).ll

        # reset! CONSUMES the rng, so re-running the same filter object does NOT repeat
        # itself. Re-seeding first does. This is the invariant the EKP objective depends on:
        # every candidate parameter vector must see the same inner noise stream (common random
        # numbers), which means re-seeding the filter's rng, not merely calling reset!.
        f = _ae_enkf(300; seed = 7)
        @test forward_trajectory(f, us, ys).ll != forward_trajectory(f, us, ys).ll

        Random.seed!(f.rng, 123)
        reseeded_a = forward_trajectory(f, us, ys).ll
        Random.seed!(f.rng, 123)
        reseeded_b = forward_trajectory(f, us, ys).ll
        @test reseeded_a == reseeded_b
    end

    @testset "post_correct_cb captures the CORRECTED ensemble, not the predicted one" begin
        # `src/inference/ensemble.jl` forecasts from the corrected ensemble at the origin, captured
        # through this callback. The loop's last act is `predict!`, and both `predict!` and
        # `correct!` mutate `f.ensemble` in place, so a REFERENCE read after `forward_trajectory`
        # returns is the predicted ensemble — one step past the origin. This pins the difference,
        # because the two are numerically close enough that the mistake would not look like a bug.
        T = 15
        us, ys, _ = _ae_simulate(T)
        f = _ae_enkf(400; seed = 31)

        calls = Ref(0)
        snapshot = Ref{Any}(nothing)
        reference = Ref{Any}(nothing)
        sol = forward_trajectory(
            f, us, ys;
            post_correct_cb = function (kf, _p, _ret)
                calls[] += 1
                reference[] = kf.ensemble               # aliased — keeps mutating
                calls[] == T && (snapshot[] = deepcopy(kf.ensemble))
                return nothing
            end,
        )

        # Fires exactly once per correction, which is what makes the counter a safe trigger.
        @test calls[] == T
        # The snapshot is the corrected mean at the origin: `xt` is post-`correct!`.
        @test mean(snapshot[]) ≈ sol.xt[end]
        # The aliased reference is NOT — it has been advanced by the final `predict!`.
        @test !isapprox(mean(reference[]), sol.xt[end])
        # And it is the predicted state instead, i.e. exactly one step too far.
        @test mean(reference[]) ≈ state(f)
    end

    @testset "inflation widens the ensemble" begin
        T = 20
        us, ys, _ = _ae_simulate(T)
        tr_plain = sum(diag(forward_trajectory(_ae_enkf(600; seed = 5), us, ys).Rt[end]))
        tr_infl = sum(
            diag(
                forward_trajectory(
                    _ae_enkf(600; seed = 5, inflation = 1.1), us, ys,
                ).Rt[end],
            )
        )
        @test tr_infl > tr_plain
    end

    @testset "reset! rewinds the step counter and the ensemble" begin
        f = _ae_enkf(128)
        us, ys, _ = _ae_simulate(4)
        for k in 1:4
            correct!(f, us[k], ys[k], nothing, float(k - 1))
            predict!(f, us[k], nothing, float(k - 1))
        end
        @test f.t == 4
        reset!(f)
        @test f.t == 0
        @test num_particles(f) == 128
        reset!(f; x0 = fill(5.0, _AE_NX))
        @test all(abs.(state(f) .- 5.0) .< 1.0)
    end

    @testset "stock additive EnsembleKalmanFilter is unchanged" begin
        # Regression guard: every method added for the augmented filter dispatches on our own
        # type, so LLPF's additive ensemble filter must behave exactly as it did before.
        add_dynamics(x, u, p, t) = _AE_A * x
        add_measurement(x, u, p, t) = _AE_C * x
        T = 20
        us, ys, xs = _ae_simulate(T)
        # Note the additive form cannot express this system's process noise at all: the
        # effective covariance `G Gᵀ` is 3x3 of rank 2, and the stock filter wraps R1 in
        # `PDMats.PDMat`, which throws on a singular matrix. `nw < nx` is exactly what the
        # augmented form handles natively, so the stock filter gets a full-rank R1 here.
        stock = EnsembleKalmanFilter(
            add_dynamics, add_measurement,
            Matrix(0.5I, _AE_NX, _AE_NX), _AE_H * _AE_H', _ae_d0(), 500;
            nu = 0, ny = _AE_NY, rng = MersenneTwister(11),
        )
        sol = forward_trajectory(stock, us, ys)
        @test sol isa KalmanFilteringSolution
        @test isfinite(sol.ll)
        @test length(sol.xt) == T
        @test all(all(isfinite, x) for x in sol.xt)
        # It still tracks the state — the process noise is misspecified relative to the
        # data-generating `G Gᵀ`, so this is a "runs and converges" check, not an accuracy one.
        @test _ae_sse(sol.xt, xs) < _ae_sse([zeros(_AE_NX) for _ in 1:T], xs)
    end
end
