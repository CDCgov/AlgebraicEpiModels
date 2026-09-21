using Test
using ConfigurableEpi
using StaticArrays: SVector
using LinearAlgebra: Diagonal
using Distributions: MvNormal
using Statistics: mean, var
using LowLevelParticleFilters: AdvancedParticleFilter, simulate, reset!, correct!,
    predict!, particles, expweights, state
import Random

# Mock SI model whose transmission scales with the learned hyperparameter
# R0_baseline, so learning it has an observable effect.
function mock_petri_vf!(du, u, p, t)
    hyper, latent = p   # full_dynamics threads (hyperparams, latent)
    beta = hyper.R0_baseline * latent.Rt * 0.3 / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

@testset "Hyperparameter learning (Liu-West)" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    # Tight Rt so R0_baseline carries the transmission level (identifiable).
    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.02))
    latent_dynamics = build_stochastic_update(layout, (Rt_spec,))

    @testset "build_learned_hyperparams: layout + round-trip" begin
        learned = build_learned_hyperparams(
            LiuWest(positive_gaussian(:R0_baseline, 1.5, 0.5)), layout
        )
        @test n_learned(learned) == 1
        @test learned.offset == layout.total_dim          # θ block is the tail
        @test hyper_range(learned) == (layout.total_dim + 1):(layout.total_dim + 1)
        @test learned.value_slots == (layout.total_dim + 1,)  # LiuWest: 1 value slot/param
        @test learned.total_slots == 1

        θunc = learned.to_unconstrained((R0_baseline = 2.3,))
        x = vcat(zeros(layout.total_dim), collect(θunc))
        @test learned.extract(x).R0_baseline ≈ 2.3 atol = 1.0e-6   # unconstrained round-trip
    end

    @testset "one LiuWest method owns a joint prior block" begin
        r0_prior = positive_gaussian(:R0_baseline, 1.5, 0.5)
        phi_prior = positive_gaussian(:phi, 20.0, 5.0)
        method = LiuWest((R0_baseline = r0_prior, phi = phi_prior); discount = 0.97)
        learned = build_learned_hyperparams(method, layout)

        @test method.priors == (r0_prior, phi_prior)
        @test method.discount == 0.97
        @test learned.method === method
        @test learned.names == (:R0_baseline, :phi)
        @test learned.value_slots == (layout.total_dim + 1, layout.total_dim + 2)
        @test learned.total_slots == 2

        constrained = (R0_baseline = 2.3, phi = 35.0)
        x = vcat(zeros(layout.total_dim), collect(learned.to_unconstrained(constrained)))
        roundtrip = learned.extract(x)
        @test roundtrip.R0_baseline ≈ constrained.R0_baseline atol = 1.0e-6
        @test roundtrip.phi ≈ constrained.phi atol = 1.0e-6

        @test_throws ArgumentError LiuWest(())
        @test_throws ArgumentError LiuWest((r0_prior, r0_prior))
        @test_throws ArgumentError LiuWest((wrong_name = r0_prior,))
        @test_throws ArgumentError LiuWest((r0_prior,); discount = 0.0)
    end

    @testset "augmented dynamics: θ carried static, drives the rates per-particle" begin
        learned = build_learned_hyperparams(
            LiuWest(positive_gaussian(:R0_baseline, 1.5, 0.5)), layout
        )
        dynamics = build_full_dynamics(
            mock_petri_vf!, latent_dynamics, layout; supersample = 2
        )
        pf_dyn = build_pf_dynamics(
            dynamics, layout; learned = learned
        )
        p = (R0_baseline = 2.0, obs_scale = 0.2)  # overridden per-particle

        Rt_unc = latent_dynamics.to_unconstrained((Rt = 1.0,))[1]
        x_lo = [990.0, 50.0, 0.0, Rt_unc, learned.to_unconstrained((R0_baseline = 1.0,))[1]]
        x_hi = [990.0, 50.0, 0.0, Rt_unc, learned.to_unconstrained((R0_baseline = 3.0,))[1]]

        nxt_lo = pf_dyn(x_lo, nothing, p, 0.0, false)
        nxt_hi = pf_dyn(x_hi, nothing, p, 0.0, false)

        @test length(nxt_lo) == layout.total_dim + 1     # augmented output
        @test nxt_lo[end] == x_lo[end]                    # θ carried unchanged (static)
        @test nxt_hi[end] == x_hi[end]
        @test nxt_hi[2] > nxt_lo[2]                        # higher R0 → faster I growth
    end

    @testset "Liu-West updater preserves weighted mean & variance" begin
        learned = build_learned_hyperparams(
            LiuWest(positive_gaussian(:R0_baseline, 1.5, 0.5); discount = 0.95), layout
        )
        update! = build_hyperparam_updater(learned; rng = Random.MersenneTwister(42))

        rng0 = Random.MersenneTwister(1)
        Nn = 4000
        θsamp = [0.3 * randn(rng0) + 0.5 for _ in 1:Nn]    # unconstrained θ
        parts = [SVector{5}(990.0, 50.0, 0.0, 0.0, θsamp[i]) for i in 1:Nn]
        w = fill(1.0 / Nn, Nn)

        pre_mean, pre_var = mean(θsamp), var(θsamp)
        update!(parts, w)
        postθ = [p[5] for p in parts]

        @test mean(postθ) ≈ pre_mean atol = 0.02          # shrinkage preserves the mean
        @test var(postθ) ≈ pre_var rtol = 0.12            # a² + (1−a²) = 1 preserves the variance
        @test all(p -> p[1] == 990.0 && p[2] == 50.0 && p[3] == 0.0 && p[4] == 0.0, parts)  # state untouched
    end

    @testset "Kulhavý forgetting pulls a named parameter toward its prior" begin
        priors = (positive_gaussian(:R0_baseline, 1.5, 0.5), positive_gaussian(:phi, 20.0, 5.0))
        m0 = collect(prior_unconstrained_mean.(priors))
        v0 = collect(prior_unconstrained_variance.(priors))
        Nn = 20_000
        w = fill(1.0 / Nn, Nn)
        # No jitter floor, so the only thing acting beyond plain Liu-West is the forgetting.
        forgetful(memory; dt = 1.0, seed = 3) = build_hyperparam_updater(
            build_learned_hyperparams(
                LiuWest(
                    priors; discount = 0.95, jitter_floor_fraction = 0.0,
                    forgetting_memory_days = (R0_baseline = memory,),
                ),
                layout,
            );
            rng = Random.MersenneTwister(seed), dt,
        )
        function cloud(sd1; seed = 4)
            rng0 = Random.MersenneTwister(seed)
            θ1 = (m0[1] + 0.5) .+ sd1 .* randn(rng0, Nn)     # offset from the prior mean
            θ2 = (m0[2] + 0.5) .+ 0.05 .* randn(rng0, Nn)
            return [SVector{6}(990.0, 50.0, 0.0, 0.0, θ1[i], θ2[i]) for i in 1:Nn], θ1, θ2
        end
        # The Gaussian geometric mean of N(m, v) and the prior N(m0, v0), with λ = exp(-dt/memory).
        function expected(m, v, memory, dt)
            λ = exp(-dt / memory)
            denom = λ * v0[1] + (1 - λ) * v
            return m + (1 - λ) * v / denom * (m0[1] - m), v * v0[1] / denom
        end

        # One step with a SHORT memory so the move is large against Monte-Carlo noise.
        parts, θ1, θ2 = cloud(0.05)
        m, v = mean(θ1), var(θ1; corrected = false)
        forgetful(2.0)(parts, w)
        post1 = [p[5] for p in parts]
        post2 = [p[6] for p in parts]
        m_new, v_new = expected(m, v, 2.0, 1.0)
        @test v_new > 1.5 * v                                   # a tight cloud is re-widened …
        @test abs(m_new - m0[1]) < abs(m - m0[1])               # … and pulled toward the prior
        @test mean(post1) ≈ m_new atol = 5.0e-4
        @test var(post1; corrected = false) ≈ v_new rtol = 0.05
        @test length(unique(post1)) == Nn
        # The parameter with no memory configured is plain Liu-West: mean and variance preserved.
        @test mean(post2) ≈ mean(θ2) atol = 5.0e-4
        @test var(post2) ≈ var(θ2) rtol = 0.05
        @test all(p -> p[1] == 990.0 && p[2] == 50.0 && p[3] == 0.0 && p[4] == 0.0, parts)

        # The memory is in DAYS: a 7-day step forgets as much as seven 1-day steps would.
        parts7, θ1_7, _ = cloud(0.05)
        forgetful(14.0; dt = 7.0)(parts7, w)
        m7, _ = expected(mean(θ1_7), var(θ1_7; corrected = false), 14.0, 7.0)
        @test mean(p[5] for p in parts7) ≈ m7 atol = 5.0e-4

        # A cloud WIDER than its prior is narrowed toward it (the rescaling branch).
        wide, θ1w, _ = cloud(2 * sqrt(v0[1]))
        vw = var(θ1w; corrected = false)
        forgetful(2.0)(wide, w)
        _, vw_new = expected(mean(θ1w), vw, 2.0, 1.0)
        @test vw_new < vw
        @test var([p[5] for p in wide]; corrected = false) ≈ vw_new rtol = 0.05

        # The prior is the FIXED POINT: with no information in the weights the cloud relaxes to
        # the prior's mean and variance, which plain Liu-West never does.
        relaxing, _, _ = cloud(0.05)
        update! = forgetful(5.0)
        for _ in 1:80
            update!(relaxing, w)
        end
        relaxed = [p[5] for p in relaxing]
        @test mean(relaxed) ≈ m0[1] atol = 0.03
        @test var(relaxed) ≈ v0[1] rtol = 0.1
        untouched = [p[6] for p in relaxing]
        @test mean(untouched) ≈ m0[2] + 0.5 atol = 0.03         # no memory, no pull
    end

    @testset "forgetting: off is bit-identical, and the request is validated" begin
        priors = (positive_gaussian(:R0_baseline, 1.5, 0.5), positive_gaussian(:phi, 20.0, 5.0))
        function run_once(; kwargs...)
            update! = build_hyperparam_updater(
                build_learned_hyperparams(LiuWest(priors; discount = 0.95, kwargs...), layout);
                rng = Random.MersenneTwister(21),
            )
            rng0 = Random.MersenneTwister(22)
            parts = [SVector{6}(990.0, 50.0, 0.0, 0.0, 0.4 + 0.1 * randn(rng0), 3.0 + 0.1 * randn(rng0)) for _ in 1:500]
            update!(parts, fill(1.0 / 500, 500))
            return parts
        end
        plain = run_once()
        @test run_once(; forgetting_memory_days = (;)) == plain
        @test run_once(; forgetting_memory_days = (R0_baseline = Inf,)) == plain   # Inf disables
        @test run_once(; forgetting_memory_days = (R0_baseline = 30.0,)) != plain

        @test isempty(LiuWest(priors).forgetting_memory_days)
        @test LiuWest(priors; forgetting_memory_days = Dict("phi" => 90)).forgetting_memory_days ==
            Dict(:phi => 90.0)
        @test LiuWest(
            (R0_baseline = priors[1], phi = priors[2]); forgetting_memory_days = (phi = 90.0,),
        ).forgetting_memory_days == Dict(:phi => 90.0)
        for bad in ((nope = 30.0,), (phi = 0.0,), (phi = -5.0,), (phi = NaN,))
            @test_throws ArgumentError LiuWest(priors; forgetting_memory_days = bad)
        end
    end

    @testset "end-to-end: learns R0_baseline toward the truth" begin
        Random.seed!(20260618)
        obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 80.0); mean_modifier = 1.0),)
        truth_hyper = (R0_baseline = 2.0, obs_scale = 0.25)

        x0 = vcat([900.0, 50.0, 0.0], collect(latent_dynamics.to_unconstrained((Rt = 1.0,))))
        P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01]))

        # Truth DGP: fixed R0 = 2.0 (no learning).
        truth_dynamics = build_full_dynamics(
            mock_petri_vf!, latent_dynamics, layout; supersample = 2
        )
        truth_measure, _ny, truth_nv = build_measurement_model(
            layout, obs_specs, latent_dynamics
        )
        pf_truth = AdvancedParticleFilter(
            500,
            build_pf_dynamics(truth_dynamics, layout),
            build_pf_measurement(
                truth_measure, truth_nv, layout, obs_specs, latent_dynamics
            ),
            build_measurement_logpdf(layout, obs_specs, latent_dynamics),
            nothing, MvNormal(x0, P0); p = truth_hyper, ny = 1, nu = 0,
        )
        T = 40
        _, _, y_true = simulate(pf_truth, fill(Float64[], T), truth_hyper)
        y_data = [yt[1:1] for yt in y_true]

        # Inference: particles learn R0 from a prior centered at 1.2 (away from truth).
        learned = build_learned_hyperparams(
            LiuWest(positive_gaussian(:R0_baseline, 1.2, 0.6); discount = 0.97), layout
        )
        updater! = build_hyperparam_updater(learned)
        x0L = vcat(x0, collect(learned.to_unconstrained((R0_baseline = 1.2,))))
        P0L = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01, 0.4]))
        pfL = AdvancedParticleFilter(
            1500,
            build_pf_dynamics(truth_dynamics, layout; learned = learned),
            build_pf_measurement(
                truth_measure, truth_nv, layout, obs_specs, latent_dynamics;
                learned = learned,
            ),
            build_measurement_logpdf(layout, obs_specs, latent_dynamics; learned = learned),
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
            updater!(state(pfL).xprev, expweights(pfL))   # Liu-West refresh before predicting
            predict!(pfL, u[t], truth_hyper)
        end

        @test isfinite(final_R0)
        @test final_R0 > 1.35                               # moved up from the prior mean (1.2)
        @test abs(final_R0 - 2.0) < abs(1.2 - 2.0)          # closer to truth than the prior was
    end
end

# The ascertainment decline rate is an OBSERVATION-side hyperparameter carried in the Liu–West
# tail: with the transmission level known and the latent held tight, the only thing that can
# explain counts falling faster than incidence is the rate, so the cloud must move toward it.
@testset "Liu-West learns the ascertainment decline rate toward the truth" begin
    Random.seed!(20260914)
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.05), sigma_rate = FixedParam(:sigma_Rt, 0.005))
    latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
    path = AscertainmentPath(0.2, 0.0)   # reference at t = 0, floor a fifth of the level
    obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 80.0); mean_modifier = path),)
    r_true = 6.0   # per year: the level falls to ~0.5 over a 60-day window
    truth_hyper = (
        R0_baseline = 1.2, obs_scale = 1.0, ascertainment = 1.0,
        ascertainment_decline_rate = r_true,
    )

    x0 = vcat([900.0, 50.0, 0.0], collect(latent_dynamics.to_unconstrained((Rt = 1.0,))))
    P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01]))
    dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout; supersample = 2)
    measure, _ny, nv = build_measurement_model(layout, obs_specs, latent_dynamics)
    pf_truth = AdvancedParticleFilter(
        500,
        build_pf_dynamics(dynamics, layout),
        build_pf_measurement(measure, nv, layout, obs_specs, latent_dynamics),
        build_measurement_logpdf(layout, obs_specs, latent_dynamics),
        nothing, MvNormal(x0, P0); p = truth_hyper, ny = 1, nu = 0,
    )
    T = 60
    _, _, y_true = simulate(pf_truth, fill(Float64[], T), truth_hyper)
    y_data = [yt[1:1] for yt in y_true]

    # Inference: the rate alone is learned, from a prior centred on "no decline".
    prior = unconstrained_gaussian(:ascertainment_decline_rate, 0.0, 4.0)
    learned = build_learned_hyperparams(LiuWest(prior; discount = 0.97), layout)
    updater! = build_hyperparam_updater(learned)
    start = merge(truth_hyper, (ascertainment_decline_rate = 0.0,))
    x0L = vcat(x0, collect(learned.to_unconstrained((ascertainment_decline_rate = 0.0,))))
    P0L = Matrix(Diagonal([1.0, 1.0, 1.0, 0.01, 16.0]))
    pfL = AdvancedParticleFilter(
        1500,
        build_pf_dynamics(dynamics, layout; learned = learned),
        build_pf_measurement(measure, nv, layout, obs_specs, latent_dynamics; learned = learned),
        build_measurement_logpdf(layout, obs_specs, latent_dynamics; learned = learned),
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
    @test final_r > 0.5 * r_true                        # more than half the gap from the prior mean
    @test abs(final_r - r_true) < abs(0.0 - r_true)     # closer to truth than the prior was
end
