# The UKF forecast draws jointly from the corrected origin Gaussian and simulates the shared
# stochastic model forward with the exact observation sampler.

using Test
using ConfigurableEpi
using LinearAlgebra: Diagonal, Symmetric, eigen
using Statistics: cov, mean, var
using LowLevelParticleFilters: sample_measurement, sample_state
import Random

const CE = ConfigurableEpi

# An SI flow that reads no latent: the latent `Z` reaches the observation only as its scale.
function ukf_forecast_latent_free_vf!(du, u, p, t)
    hyper, _latent = p
    beta = 0.3 * hyper.R0_baseline / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

# The inference fixture's SI flow driven by an `Rt` latent.
function ukf_forecast_rt_vf!(du, u, p, t)
    hyper, latent = p
    beta = 0.3 * hyper.R0_baseline * latent.Rt / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

ukf_forecast_stat(df, parameter, statistic) =
    only(df[(df.parameter .== parameter) .& (df.statistic .== statistic), :value])

@testset "UKF sampled forecast" begin
    # A zero-noise log-chart latent that scales the observation.
    layout = StateLayout((:S, :I), (:O_I_1,), (:Z,))
    z_spec = RWParamSpec(:Z; init = positive_gaussian(:Z, 1.0, 0.5), sigma_rate = FixedParam(:sigma_Z, 0.0))
    stochastic = build_stochastic_update(layout, (z_spec,))
    scale_by_latent = (l, _h, _t) -> l.Z
    nb_spec = SignalObservationSpec(1, NegBinomialNoise(phi = (_l, h, _t) -> h.phi); mean_modifier = scale_by_latent, name = :reports)
    hp = (R0_baseline = 1.5, obs_scale = 2.0, phi = 50.0)
    x0 = [900.0, 50.0, 0.0, stochastic.to_unconstrained((Z = 1.0,))[1]]
    make_model(observation) = EpiModel(;
        vectorfield! = ukf_forecast_latent_free_vf!, layout, stochastic, observation = (observation,), hyperparams = hp,
        priors = (R0_baseline = positive_gaussian(:R0_baseline, 1.5, 0.4),), initial_state = x0,
    )
    model = make_model(nb_spec)
    settings = EngineSettings(; dt = 1.0, supersample = 2, n_ahead = 3, n_draws = 20_000)
    assembly = CE.Assembly(model, settings, 1.0)
    u = Float64[]
    nw = n_latent(layout) + length(layout.accumulator_indices)
    ode = 1:n_ode_states(layout)
    # The accumulator one noise-free step from `x0`: the deterministic incidence `A`.
    A = assembly.forecast_dynamics(x0, u, hp, 0.0, zeros(nw))[3]
    @test A > 100.0

    # Draw `n` origin states from `(x0, R)` and simulate one step; returns `(y, z)`.
    function one_step(R, n; seed = 1)
        rng = Random.Xoshiro(seed)
        draws = CE._draw_origin_states(rng, x0, R, n; ode_range = ode)
        sim = CE._forecast_simulator(assembly, model, rng)
        samples, latents = forecast_ensemble(sim, draws.states, hp; n_ahead = 1, t0 = 0.0, dt = 1.0, latent_range = 4:4)
        return vec(samples[1, :, 1]), vec(latents[1, :, 1])
    end

    @testset "latent uncertainty moves the predictive mean and spread" begin
        R = Matrix(Diagonal([1.0e-6, 1.0e-6, 1.0e-6, 0.25]))
        y, z = one_step(R, 20_000)
        mu = A .* exp.(z)
        @test mean(y) ≈ mean(mu) rtol = 0.01
        @test mean(mu) ≈ A * exp(0.125) rtol = 0.02          # E[A e^Z] = A e^{σ²/2}, not A e^{E Z}
        @test abs(mean(y) - A) > 0.1 * A
        @test var(y) ≈ mean(mu .+ mu .^ 2 ./ hp.phi) + var(mu) rtol = 0.1
        @test var(mu) > 1.0
        base = mean(y)

        # Cov(I, Z) shifts the mean by Stein's term Cov(A(I), e^Z); the two signs straddle the
        # uncorrelated mean by about c / I0 = 4.5 / 50 each.
        function correlated(c)
            Rc = Matrix(Diagonal([1.0e-6, 100.0, 1.0e-6, 0.25]))
            Rc[2, 4] = Rc[4, 2] = c
            return mean(first(one_step(Rc, 20_000; seed = 2)))
        end
        up, down = correlated(4.5), correlated(-4.5)
        @test up > 1.05 * base
        @test down < 0.95 * base
        @test (up - down) / base ≈ 0.18 rtol = 0.15
    end

    @testset "joint origin draws keep every cross-covariance; the repair adds nothing" begin
        L = [1.0 0.0 0.0 0.0; 0.5 2.0 0.0 0.0; 0.2 -0.3 1.5 0.0; 0.1 0.4 -0.2 0.5]
        R = L * L'
        x = [900.0, 50.0, 10.0, 0.0]
        d = CE._draw_origin_states(Random.Xoshiro(3), x, R, 50_000; ode_range = 1:3)
        X = permutedims(reduce(hcat, d.states))
        @test cov(X) ≈ R atol = 0.03 * maximum(abs, R)
        @test vec(mean(X; dims = 1)) ≈ x atol = 0.05
        @test d.n_clamped == 0 && d.min_eigenvalue > 0.0 && d.floored_fraction == 0.0

        # A rounding-level negative direction is clamped to zero variance without a warning.
        V = eigen(Symmetric(randn(Random.Xoshiro(4), 4, 4))).vectors
        indefinite = V * Diagonal([2.0, 1.0, 0.5, -1.0e-9]) * V'
        d2 = @test_logs CE._draw_origin_states(Random.Xoshiro(5), x, indefinite, 50_000; ode_range = 1:3)
        @test d2.n_clamped == 1
        @test d2.min_eigenvalue ≈ -1.0e-9 atol = 1.0e-12
        X2 = permutedims(reduce(hcat, d2.states))
        @test var(X2 * V[:, 4]) < 1.0e-12
        @test cov(X2) ≈ V * Diagonal([2.0, 1.0, 0.5, 0.0]) * V' atol = 0.06
        hostile = V * Diagonal([2.0, 1.0, 0.5, -0.1]) * V'
        @test_logs (:warn, r"materially indefinite") CE._draw_origin_states(Random.Xoshiro(6), x, hostile, 10; ode_range = 1:3)
        asymmetric = copy(R)
        asymmetric[1, 2] += 1.0e-9
        @test length(CE._draw_origin_states(Random.Xoshiro(7), x, asymmetric, 10; ode_range = 1:3).states) == 10
        @test_throws DimensionMismatch CE._draw_origin_states(Random.Xoshiro(7), x, R[1:3, 1:3], 10; ode_range = 1:3)

        # ODE slots are floored at the dynamics' floor; latents are not.
        near_zero = [1.0e-3, 50.0, 10.0, 0.0]
        d3 = CE._draw_origin_states(Random.Xoshiro(8), near_zero, Matrix(Diagonal([1.0, 1.0e-6, 1.0e-6, 1.0])), 20_000; ode_range = 1:3)
        @test d3.floored_fraction ≈ 0.5 rtol = 0.05
        @test all(s -> all(>=(STATE_FLOOR), view(s, 1:3)), d3.states)
        @test any(s -> s[4] < 0.0, d3.states)
    end

    @testset "the shared dynamics and the exact observation law, at the right times" begin
        # The forecast dynamics are the filter step without the accumulator jitter.
        w = randn(Random.Xoshiro(9), nw)
        w2 = copy(w)
        w2[end] += 1.0
        @test assembly.forecast_dynamics(x0, u, hp, 0.0, w) == assembly.forecast_dynamics(x0, u, hp, 0.0, w2)
        @test assembly.dynamics(x0, u, hp, 0.0, w2)[3] - assembly.dynamics(x0, u, hp, 0.0, w)[3] ≈ 1.0
        zero_jitter = CE.Assembly(model, settings, 0.0)
        @test zero_jitter.forecast_dynamics === zero_jitter.dynamics
        sim = CE._forecast_simulator(assembly, model, Random.Xoshiro(10))
        @test sample_state(sim, x0, u, hp, 0.0; noise = false) == assembly.forecast_dynamics(x0, u, hp, 0.0, zeros(nw))

        # The spec is evaluated at each horizon's own time with the fixed parameters, and the
        # observation is an exact draw (an integer count here).
        times, scales = Float64[], Float64[]
        recording = SignalObservationSpec(
            1, NegBinomialNoise(phi = (_l, h, _t) -> h.phi);
            mean_modifier = (l, h, t) -> (push!(times, t); push!(scales, h.obs_scale); l.Z),
        )
        model_t = make_model(recording)
        sim_t = CE._forecast_simulator(CE.Assembly(model_t, settings, 1.0), model_t, Random.Xoshiro(11))
        y, _ = forecast_ensemble(sim_t, [x0], hp; n_ahead = 3, t0 = 10.0, dt = 2.0)
        @test times == [12.0, 14.0, 16.0]
        @test scales == fill(hp.obs_scale, 3)
        @test all(isinteger, y)

        # The measurement is `sample_observation` on the simulator's stream.
        poisson = SignalObservationSpec(1, PoissonNoise(); mean_modifier = scale_by_latent)
        model_p = make_model(poisson)
        x1 = assembly.forecast_dynamics(x0, u, hp, 0.0, zeros(nw))
        latent = stochastic.extract(x1)
        expected = sample_observation(PoissonNoise(), observation_mean(poisson, x1[3], latent, hp, 1.0), latent, hp, 1.0, Random.Xoshiro(12))
        sim_p = CE._forecast_simulator(CE.Assembly(model_p, settings, 1.0), model_p, Random.Xoshiro(12))
        @test only(sample_measurement(sim_p, x1, u, hp, 1.0)) == expected
    end

    @testset "a zero-noise latent keeps its sampled value across horizons" begin
        rng = Random.Xoshiro(13)
        draws = CE._draw_origin_states(rng, x0, Matrix(Diagonal([1.0, 1.0, 1.0e-6, 0.25])), 200; ode_range = ode)
        sim = CE._forecast_simulator(assembly, model, rng)
        _, latents = forecast_ensemble(sim, draws.states, hp; n_ahead = 3, t0 = 0.0, dt = 1.0, latent_range = 4:4)
        @test all(latents[h, j, 1] == draws.states[j][4] for h in 1:3, j in 1:200)
    end

    # The engines, on the inference fixture.
    layout_b = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    obs_b = (
        SignalObservationSpec(
            1, NegBinomialNoise(phi = (_l, h, _t) -> h.phi);
            mean_modifier = (_l, h, _t) -> h.ascertainment, name = :reports,
        ),
    )
    hp_b = (R0_baseline = 1.5, obs_scale = 0.25, ascertainment = 0.8, phi = 50.0)
    function make_model_b(sigma)
        rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, sigma))
        stochastic_b = build_stochastic_update(layout_b, (rt_spec,))
        return EpiModel(;
            vectorfield! = ukf_forecast_rt_vf!, layout = layout_b, stochastic = stochastic_b, observation = obs_b,
            hyperparams = hp_b, priors = (R0_baseline = positive_gaussian(:R0_baseline, 1.5, 0.4),),
            initial_state = [900.0, 50.0, 10.0, stochastic_b.to_unconstrained((Rt = 1.0,))[1]],
        )
    end
    model_b = make_model_b(0.03)
    counts3 = [8.0, 10.0, 12.0]
    optimise(seed; n_draws = 100, model = model_b) = build_inference(
        UKF(), Optimise(; maxiters = 1, maxiters_burnin = 1), model;
        dt = 1.0, supersample = 1, n_ahead = 2, n_draws, rng = Random.MersenneTwister(seed),
    )
    estimate_rows(df) = df[df.statistic .== "estimate", :]

    @testset "the spread of a zero-noise latent is its origin sd at every horizon" begin
        frozen = make_model_b(0.0)
        e = optimise(14; n_draws = 20_000, model = frozen)
        result = fit_forecast!(e, counts3, 1)
        origin = CE._filter_pass!(CE._ukf(Float64, e.core.assembly, frozen, e.hyperparams, 1.0), CE._observation_vectors(counts3), e.hyperparams)
        sd1 = ukf_forecast_stat(result.summary, "Rt", "fc_log_sd_h1")
        @test sd1 == ukf_forecast_stat(result.summary, "Rt", "fc_log_sd_h2")
        @test sd1 ≈ sqrt(origin.Rt[end][4, 4]) rtol = 0.05
        @test ukf_forecast_stat(result.summary, "ukf", "origin_cov_clamped_dims") >= 0.0
        @test 0.0 <= ukf_forecast_stat(result.summary, "ukf", "origin_draws_floored_frac") <= 1.0
        @test isfinite(ukf_forecast_stat(result.summary, "ukf", "origin_cov_min_eigenvalue"))
    end

    @testset "seeded, isolated forecast streams" begin
        a, b, c = optimise(5), optimise(5), optimise(6)
        ra, rb, rc = fit_forecast!(a, counts3, 1), fit_forecast!(b, counts3, 1), fit_forecast!(c, counts3, 1)
        @test ra.samples == rb.samples
        @test ra.quantiles == forecast_quantiles(ra.samples)
        @test rc.samples != ra.samples
        @test rc.fitted_means == ra.fitted_means
        @test estimate_rows(rc.summary) == estimate_rows(ra.summary)
        @test size(ra.samples) == (2, 100)

        # A suppressed origin leaves the stream untouched; an emitted forecast takes one sub-seed.
        e = optimise(9)
        before = copy(e.rng_forecast)
        fit_forecast!(e, counts3, 1; emit_forecast = false)
        @test e.rng_forecast == before
        fit_forecast!(e, counts3, 2)
        advanced = copy(before)
        rand(advanced, UInt64)
        @test e.rng_forecast == advanced
    end
end
