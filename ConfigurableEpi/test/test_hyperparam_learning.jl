using Test
using ConfigurableEpi
using StaticArrays: SVector
using LinearAlgebra: Diagonal
using Distributions: MvNormal
using Statistics: mean, var
using LowLevelParticleFilters: AdvancedParticleFilter, simulate, reset!, correct!,
    predict!, particles, expweights, state
import Random

# Mock SI model whose transmission scales with the learned hyperparameter R0_baseline.
function mock_petri_vf!(du, u, p, t)
    hyper, latent = p
    beta = hyper.R0_baseline * latent.Rt * 0.3 / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

@testset "Hyperparameter learning (Liu-West)" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.02))
    latent_dynamics = build_stochastic_update(layout, (Rt_spec,))

    @testset "build_learned_hyperparams: layout + round-trip" begin
        learned = build_learned_hyperparams(positive_gaussian(:R0_baseline, 1.5, 0.5), layout)
        @test n_learned(learned) == 1
        @test learned.offset == layout.total_dim
        @test hyper_range(learned) == (layout.total_dim + 1):(layout.total_dim + 1)
        @test ConfigurableEpi.hyper_slots(learned) == (layout.total_dim + 1,)
        θunc = learned.to_unconstrained((R0_baseline = 2.3,))
        x = vcat(zeros(layout.total_dim), collect(θunc))
        @test learned.extract(x).R0_baseline ≈ 2.3 atol = 1.0e-6
    end

    @testset "one block owns several priors" begin
        r0_prior = positive_gaussian(:R0_baseline, 1.5, 0.5)
        phi_prior = positive_gaussian(:phi, 20.0, 5.0)
        learned = build_learned_hyperparams((R0_baseline = r0_prior, phi = phi_prior), layout)
        @test learned.priors == (r0_prior, phi_prior)
        @test learned.names == (:R0_baseline, :phi)
        @test ConfigurableEpi.hyper_slots(learned) == (layout.total_dim + 1, layout.total_dim + 2)
        @test build_learned_hyperparams((r0_prior, phi_prior), layout).names == (:R0_baseline, :phi)
        constrained = (R0_baseline = 2.3, phi = 35.0)
        roundtrip = learned.extract(vcat(zeros(layout.total_dim), collect(learned.to_unconstrained(constrained))))
        @test roundtrip.R0_baseline ≈ 2.3 atol = 1.0e-6
        @test roundtrip.phi ≈ 35.0 atol = 1.0e-6

        @test_throws ArgumentError build_learned_hyperparams((;), layout)
        @test_throws ArgumentError build_learned_hyperparams((r0_prior, r0_prior), layout)
        @test_throws ArgumentError build_learned_hyperparams((wrong_name = r0_prior,), layout)
        @test_throws ArgumentError build_hyperparam_updater(learned; discount = 0.0)
        @test_throws ArgumentError build_hyperparam_updater(learned; jitter_floor_fraction = -1.0)
    end

    @testset "augmented dynamics: θ carried static, drives the rates per-particle" begin
        learned = build_learned_hyperparams(positive_gaussian(:R0_baseline, 1.5, 0.5), layout)
        dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout; supersample = 2)
        pf_dyn = build_pf_dynamics(dynamics, layout; learned)
        p = (R0_baseline = 2.0, obs_scale = 0.2)
        Rt_unc = latent_dynamics.to_unconstrained((Rt = 1.0,))[1]
        x_lo = [990.0, 50.0, 0.0, Rt_unc, learned.to_unconstrained((R0_baseline = 1.0,))[1]]
        x_hi = [990.0, 50.0, 0.0, Rt_unc, learned.to_unconstrained((R0_baseline = 3.0,))[1]]
        nxt_lo = pf_dyn(x_lo, nothing, p, 0.0, false)
        nxt_hi = pf_dyn(x_hi, nothing, p, 0.0, false)
        @test length(nxt_lo) == layout.total_dim + 1
        @test nxt_lo[end] == x_lo[end]
        @test nxt_hi[end] == x_hi[end]
        @test nxt_hi[2] > nxt_lo[2]
    end

    @testset "Liu-West updater preserves weighted mean & variance" begin
        learned = build_learned_hyperparams(positive_gaussian(:R0_baseline, 1.5, 0.5), layout)
        update! = build_hyperparam_updater(learned; discount = 0.95, rng = Random.MersenneTwister(42))
        rng0 = Random.MersenneTwister(1)
        Nn = 4000
        θsamp = [0.3 * randn(rng0) + 0.5 for _ in 1:Nn]
        parts = [SVector{5}(990.0, 50.0, 0.0, 0.0, θsamp[i]) for i in 1:Nn]
        w = fill(1.0 / Nn, Nn)
        pre_mean, pre_var = mean(θsamp), var(θsamp)
        update!(parts, w)
        postθ = [p[5] for p in parts]
        @test mean(postθ) ≈ pre_mean atol = 0.02
        @test var(postθ) ≈ pre_var rtol = 0.12
        @test all(p -> p[1] == 990.0 && p[2] == 50.0 && p[3] == 0.0 && p[4] == 0.0, parts)
    end

    @testset "Kulhavý forgetting pulls a named parameter toward its prior" begin
        priors = (positive_gaussian(:R0_baseline, 1.5, 0.5), positive_gaussian(:phi, 20.0, 5.0))
        learned = build_learned_hyperparams(priors, layout)
        m0 = collect(prior_unconstrained_mean.(priors))
        v0 = collect(prior_unconstrained_variance.(priors))
        Nn = 20_000
        w = fill(1.0 / Nn, Nn)
        # No jitter floor, so the only thing beyond plain Liu-West is the forgetting.
        forgetful(memory; dt = 1.0, seed = 3) = build_hyperparam_updater(
            learned; discount = 0.95, jitter_floor_fraction = 0.0,
            forgetting_memory_days = (R0_baseline = memory,), rng = Random.MersenneTwister(seed), dt,
        )
        function cloud(sd1; seed = 4)
            rng0 = Random.MersenneTwister(seed)
            θ1 = (m0[1] + 0.5) .+ sd1 .* randn(rng0, Nn)
            θ2 = (m0[2] + 0.5) .+ 0.05 .* randn(rng0, Nn)
            return [SVector{6}(990.0, 50.0, 0.0, 0.0, θ1[i], θ2[i]) for i in 1:Nn], θ1, θ2
        end
        # The Gaussian geometric mean of N(m, v) and the prior N(m0, v0), with λ = exp(-dt/memory).
        function expected(m, v, memory, dt)
            λ = exp(-dt / memory)
            denom = λ * v0[1] + (1 - λ) * v
            return m + (1 - λ) * v / denom * (m0[1] - m), v * v0[1] / denom
        end

        parts, θ1, θ2 = cloud(0.05)
        m, v = mean(θ1), var(θ1; corrected = false)
        forgetful(2.0)(parts, w)
        post1 = [p[5] for p in parts]
        post2 = [p[6] for p in parts]
        m_new, v_new = expected(m, v, 2.0, 1.0)
        @test v_new > 1.5 * v
        @test abs(m_new - m0[1]) < abs(m - m0[1])
        @test mean(post1) ≈ m_new atol = 5.0e-4
        @test var(post1; corrected = false) ≈ v_new rtol = 0.05
        @test length(unique(post1)) == Nn
        @test mean(post2) ≈ mean(θ2) atol = 5.0e-4
        @test var(post2) ≈ var(θ2) rtol = 0.05
        @test all(p -> p[1] == 990.0 && p[2] == 50.0 && p[3] == 0.0 && p[4] == 0.0, parts)

        # The memory is in days: a 7-day step forgets as much as seven 1-day steps would.
        parts7, θ1_7, _ = cloud(0.05)
        forgetful(14.0; dt = 7.0)(parts7, w)
        m7, _ = expected(mean(θ1_7), var(θ1_7; corrected = false), 14.0, 7.0)
        @test mean(p[5] for p in parts7) ≈ m7 atol = 5.0e-4

        # A cloud wider than its prior is narrowed toward it.
        wide, θ1w, _ = cloud(2 * sqrt(v0[1]))
        vw = var(θ1w; corrected = false)
        forgetful(2.0)(wide, w)
        _, vw_new = expected(mean(θ1w), vw, 2.0, 1.0)
        @test vw_new < vw
        @test var([p[5] for p in wide]; corrected = false) ≈ vw_new rtol = 0.05

        # The prior is the fixed point: with no information the cloud relaxes to it.
        relaxing, _, _ = cloud(0.05)
        update! = forgetful(5.0)
        for _ in 1:80
            update!(relaxing, w)
        end
        relaxed = [p[5] for p in relaxing]
        @test mean(relaxed) ≈ m0[1] atol = 0.03
        @test var(relaxed) ≈ v0[1] rtol = 0.1
        @test mean([p[6] for p in relaxing]) ≈ m0[2] + 0.5 atol = 0.03
    end

    @testset "forgetting: off is bit-identical, and the request is validated" begin
        priors = (positive_gaussian(:R0_baseline, 1.5, 0.5), positive_gaussian(:phi, 20.0, 5.0))
        learned = build_learned_hyperparams(priors, layout)
        function run_once(; kwargs...)
            update! = build_hyperparam_updater(learned; discount = 0.95, rng = Random.MersenneTwister(21), kwargs...)
            rng0 = Random.MersenneTwister(22)
            parts = [SVector{6}(990.0, 50.0, 0.0, 0.0, 0.4 + 0.1 * randn(rng0), 3.0 + 0.1 * randn(rng0)) for _ in 1:500]
            update!(parts, fill(1.0 / 500, 500))
            return parts
        end
        plain = run_once()
        @test run_once(; forgetting_memory_days = (;)) == plain
        @test run_once(; forgetting_memory_days = (R0_baseline = Inf,)) == plain
        @test run_once(; forgetting_memory_days = Dict("R0_baseline" => 30.0)) != plain
        for bad in ((nope = 30.0,), (phi = 0.0,), (phi = -5.0,), (phi = NaN,))
            @test_throws ArgumentError build_hyperparam_updater(learned; forgetting_memory_days = bad)
        end
        @test isempty(LiuWest().forgetting_memory_days)
    end

    @testset "end-to-end: learns R0_baseline toward the truth" begin
        Random.seed!(20260618)
        obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 80.0); mean_modifier = 1.0),)
        truth_hyper = (R0_baseline = 2.0, obs_scale = 0.25)
        x0 = vcat([900.0, 50.0, 0.0], collect(latent_dynamics.to_unconstrained((Rt = 1.0,))))
        P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01]))
        truth_dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout; supersample = 2)
        pf_truth = AdvancedParticleFilter(
            500, build_pf_dynamics(truth_dynamics, layout), build_pf_measurement(layout, obs_specs, latent_dynamics),
            build_measurement_logpdf(layout, obs_specs, latent_dynamics), nothing, MvNormal(x0, P0);
            p = truth_hyper, ny = 1, nu = 0,
        )
        T = 40
        _, _, y_true = simulate(pf_truth, fill(Float64[], T), truth_hyper)
        y_data = [yt[1:1] for yt in y_true]

        learned = build_learned_hyperparams(positive_gaussian(:R0_baseline, 1.2, 0.6), layout)
        updater! = build_hyperparam_updater(learned; discount = 0.97)
        x0L = vcat(x0, collect(learned.to_unconstrained((R0_baseline = 1.2,))))
        P0L = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01, 0.4]))
        pfL = AdvancedParticleFilter(
            1500, build_pf_dynamics(truth_dynamics, layout; learned),
            build_pf_measurement(layout, obs_specs, latent_dynamics; learned),
            build_measurement_logpdf(layout, obs_specs, latent_dynamics; learned),
            nothing, MvNormal(x0L, P0L); p = truth_hyper, ny = 1, nu = 0,
        )
        u = fill(Float64[], T)
        reset!(pfL)
        final_R0 = 0.0
        for t in 1:T
            correct!(pfL, u[t], y_data[t], truth_hyper)
            parts, we = particles(pfL), expweights(pfL)
            R0s = [learned.extract(parts[i]).R0_baseline for i in eachindex(parts)]
            final_R0 = sum(we .* R0s) / sum(we)
            updater!(state(pfL).xprev, expweights(pfL))
            predict!(pfL, u[t], truth_hyper)
        end
        @test isfinite(final_R0)
        @test final_R0 > 1.35
        @test abs(final_R0 - 2.0) < abs(1.2 - 2.0)
    end
end

# With the transmission level known and the latent held tight, the only thing that can explain
# counts falling faster than incidence is the ascertainment decline rate, so the cloud must move.
@testset "Liu-West learns the ascertainment decline rate toward the truth" begin
    Random.seed!(20260914)
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.05), sigma_rate = FixedParam(:sigma_Rt, 0.005))
    latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
    path = AscertainmentPath(0.2, 0.0)
    obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 80.0); mean_modifier = path),)
    r_true = 6.0
    truth_hyper = (R0_baseline = 1.2, obs_scale = 1.0, ascertainment = 1.0, ascertainment_decline_rate = r_true)
    x0 = vcat([900.0, 50.0, 0.0], collect(latent_dynamics.to_unconstrained((Rt = 1.0,))))
    P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01]))
    dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout; supersample = 2)
    pf_truth = AdvancedParticleFilter(
        500, build_pf_dynamics(dynamics, layout), build_pf_measurement(layout, obs_specs, latent_dynamics),
        build_measurement_logpdf(layout, obs_specs, latent_dynamics), nothing, MvNormal(x0, P0);
        p = truth_hyper, ny = 1, nu = 0,
    )
    T = 60
    _, _, y_true = simulate(pf_truth, fill(Float64[], T), truth_hyper)
    y_data = [yt[1:1] for yt in y_true]

    learned = build_learned_hyperparams(unconstrained_gaussian(:ascertainment_decline_rate, 0.0, 4.0), layout)
    updater! = build_hyperparam_updater(learned; discount = 0.97)
    start = merge(truth_hyper, (ascertainment_decline_rate = 0.0,))
    x0L = vcat(x0, collect(learned.to_unconstrained((ascertainment_decline_rate = 0.0,))))
    P0L = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01, 16.0]))
    pfL = AdvancedParticleFilter(
        1500, build_pf_dynamics(dynamics, layout; learned),
        build_pf_measurement(layout, obs_specs, latent_dynamics; learned),
        build_measurement_logpdf(layout, obs_specs, latent_dynamics; learned),
        nothing, MvNormal(x0L, P0L); p = start, ny = 1, nu = 0,
    )
    u = fill(Float64[], T)
    reset!(pfL)
    final_r = 0.0
    for t in 1:T
        correct!(pfL, u[t], y_data[t], start)
        parts, we = particles(pfL), expweights(pfL)
        rs = [learned.extract(parts[i]).ascertainment_decline_rate for i in eachindex(parts)]
        final_r = sum(we .* rs) / sum(we)
        updater!(state(pfL).xprev, expweights(pfL))
        predict!(pfL, u[t], start)
    end
    @test isfinite(final_r)
    @test final_r > 0.5 * r_true
    @test abs(final_r - r_true) < abs(0.0 - r_true)
end
