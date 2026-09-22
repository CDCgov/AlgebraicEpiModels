using Test
using ConfigurableEpi
using StaticArrays: SVector
using LinearAlgebra: Diagonal
using Distributions: MvNormal
using LowLevelParticleFilters: AdvancedParticleFilter, forward_trajectory, simulate,
    mean_trajectory, num_particles
import Random

# The PF builders adapt the UKF predict closure / measurement model to the
# AdvancedParticleFilter convention (x, u, p, t, noise=false). We reuse the same
# mock Petri vectorfield as test_full_dynamics.jl.
@testset "PF builders" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))

    function mock_petri_vf!(du, u, p, t)
        hyper, _latent = p   # full_dynamics threads (hyperparams, latent)
        beta = 0.3
        du[:S] = -beta * u[:S] * u[:I] / 1000.0
        du[:I] = beta * u[:S] * u[:I] / 1000.0 - 0.1 * u[:I]
        du[:O_I_1] = hyper.obs_scale * u[:I]
        return nothing
    end

    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.05))
    latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
    hyperparams = (sigma_Rt = 0.05, obs_scale = 0.2)
    nw = size(build_R1(layout), 1)

    @testset "deterministic dynamics (noise=false) reuses the zero-noise UKF step" begin
        rng = Random.MersenneTwister(1)
        ukf_dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout; supersample = 2)
        pf_dynamics = build_pf_dynamics(ukf_dynamics, layout; rng = rng)

        x = [990.0, 10.0, 0.0, 0.0]
        a = pf_dynamics(x, nothing, hyperparams, 0.0, false)
        b = pf_dynamics(x, nothing, hyperparams, 0.0)            # 4-arg → default false
        c = ukf_dynamics(x, nothing, hyperparams, 0.0, zeros(nw))
        @test a == b == c
        @test a[4] == x[4]   # latent unchanged with zero process noise
    end

    @testset "noisy dynamics (noise=true) draw process noise; seed → reproducible" begin
        x = [990.0, 10.0, 0.0, 0.0]
        dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout)
        d1 = build_pf_dynamics(dynamics, layout; rng = Random.MersenneTwister(7))
        d2 = build_pf_dynamics(dynamics, layout; rng = Random.MersenneTwister(7))

        y1 = d1(x, nothing, hyperparams, 0.0, true)
        y2 = d2(x, nothing, hyperparams, 0.0, true)
        @test y1 == y2                              # same seed → identical
        @test y1[4] != x[4]                         # latent moved under process noise
        @test y1 != d1(x, nothing, hyperparams, 0.0, false)
    end

    @testset "threaded propagation: per-thread RNG pool, seeded-reproducible" begin
        x = [990.0, 10.0, 0.0, 0.0]
        dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout)
        d1 = build_pf_dynamics(
            dynamics, layout; rng = Random.MersenneTwister(7), threads = true
        )
        d2 = build_pf_dynamics(
            dynamics, layout; rng = Random.MersenneTwister(7), threads = true
        )
        y1 = d1(x, nothing, hyperparams, 0.0, true)
        @test y1 == d2(x, nothing, hyperparams, 0.0, true)   # same seed → identical streams
        @test y1[4] != x[4]                                  # noise actually drawn
        # The deterministic path bypasses the pool entirely and matches the serial builder.
        d_serial = build_pf_dynamics(dynamics, layout; rng = Random.MersenneTwister(7))
        @test d1(x, nothing, hyperparams, 0.0, false) ==
            d_serial(x, nothing, hyperparams, 0.0, false)
    end

    @testset "threaded AdvancedParticleFilter reproduces under a fixed thread count" begin
        obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 100.0); mean_modifier = 1.0),)
        dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout; supersample = 2)
        measure, _ny, nv = build_measurement_model(layout, obs_specs, latent_dynamics)
        g = build_measurement_logpdf(layout, obs_specs, latent_dynamics)
        x0 = [900.0, 50.0, 0.0, latent_dynamics.to_unconstrained((Rt = 1.0,))[1]]
        d0 = MvNormal(x0, Matrix(Diagonal([1.0, 1.0, 1.0, 0.04])))

        function make_threaded_pf(seed)
            pf_dyn = build_pf_dynamics(
                dynamics, layout; rng = Random.MersenneTwister(seed), threads = true
            )
            pf_meas = build_pf_measurement(
                layout, obs_specs, latent_dynamics;
                rng = Random.MersenneTwister(seed),
            )
            return AdvancedParticleFilter(
                200, pf_dyn, pf_meas, g, nothing, d0;
                p = hyperparams, ny = 1, nu = 0,
                rng = Random.MersenneTwister(seed), threads = true,
            )
        end

        T = 10
        us = fill(Float64[], T)
        y_data = [[5.0 + 2.0 * t] for t in 1:T]
        # LLPF's `ResampleSystematic` draws its offset from the GLOBAL rng (`rand()`, not
        # `pf.rng`), so the reproducibility contract — which run_model.jl satisfies via
        # `Random.seed!(seed)` — covers the global stream as well as the filter rng.
        Random.seed!(31)
        sol1 = forward_trajectory(make_threaded_pf(11), us, y_data)
        Random.seed!(31)
        sol2 = forward_trajectory(make_threaded_pf(11), us, y_data)
        @test isfinite(sol1.ll)
        @test sol1.ll == sol2.ll
        @test sol1.x == sol2.x
        @test sol1.we == sol2.we
    end

    @testset "measurement: deterministic mean vs true-distribution sample" begin
        obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 100.0); mean_modifier = 1.0),)
        measure, _ny, nv = build_measurement_model(layout, obs_specs, latent_dynamics)
        pf_measure = build_pf_measurement(
            layout, obs_specs, latent_dynamics;
            rng = Random.MersenneTwister(3),
        )

        x = [900.0, 50.0, 30.0, 0.0]   # accumulator (index 3) = 30
        mean_obs = pf_measure(x, nothing, hyperparams, 0.0, false)
        @test mean_obs isa SVector{1}
        @test mean_obs[1] ≈ 30.0 atol = 1.0e-8     # deterministic mean = accumulator

        sampled = pf_measure(x, nothing, hyperparams, 0.0, true)
        @test sampled isa SVector{1}
        @test sampled[1] >= 0.0
        @test sampled[1] == round(sampled[1])      # NB draw is an integer count
    end

    @testset "AdvancedParticleFilter end-to-end" begin
        Random.seed!(20260617)
        obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 100.0); mean_modifier = 1.0),)
        dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout; supersample = 2)
        measure, _ny, nv = build_measurement_model(layout, obs_specs, latent_dynamics)
        pf_dynamics = build_pf_dynamics(dynamics, layout)
        pf_measure = build_pf_measurement(
            layout, obs_specs, latent_dynamics
        )
        g = build_measurement_logpdf(layout, obs_specs, latent_dynamics)

        x0 = [900.0, 50.0, 0.0, latent_dynamics.to_unconstrained((Rt = 1.0,))[1]]
        P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.04]))
        d0 = MvNormal(x0, P0)

        N_particles = 300
        pf = AdvancedParticleFilter(
            N_particles, pf_dynamics, pf_measure, g, nothing, d0;
            p = hyperparams, ny = 1, nu = 0,
        )
        @test pf isa AdvancedParticleFilter
        @test pf.dynamics_density === nothing
        @test num_particles(pf) == N_particles

        T = 20
        x_true, _, y_true = simulate(pf, fill(Float64[], T), hyperparams)
        y_data = [[round(max(yt[1], 0.0))] for yt in y_true]
        @test length(y_data) == T

        sol = forward_trajectory(pf, fill(Float64[], T), y_data)
        @test isfinite(sol.ll)
        @test size(sol.x) == (N_particles, T)
        @test all(t -> isapprox(sum(sol.we[:, t]), 1.0; atol = 1.0e-6), 1:T)

        xm = mean_trajectory(sol)
        @test size(xm) == (T, layout.total_dim)
    end
end

# The noisy PF draw must carry the additive `baseline` exactly as the likelihood it is scored
# against does (`build_measurement_logpdf` goes through `observation_mean`); it used to apply the
# multiplicative modifier alone.
@testset "PF noisy draw carries the additive baseline" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.05))
    latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
    hyperparams = (sigma_Rt = 0.05, obs_scale = 0.2)
    obs_specs = (
        SignalObservationSpec(
            1, LogNormalNoise(1.0e-12); mean_modifier = 0.5, baseline = 7.0
        ),
    )
    measure, _ny, nv = build_measurement_model(layout, obs_specs, latent_dynamics)
    pf_measure = build_pf_measurement(
        layout, obs_specs, latent_dynamics; rng = Random.MersenneTwister(5)
    )
    x = [900.0, 50.0, 30.0, 0.0]   # accumulator (index 3) = 30
    @test pf_measure(x, nothing, hyperparams, 0.0, false)[1] ≈ 30.0 * 0.5 + 7.0
    @test pf_measure(x, nothing, hyperparams, 0.0, true)[1] ≈ 30.0 * 0.5 + 7.0 rtol = 1.0e-8
end
