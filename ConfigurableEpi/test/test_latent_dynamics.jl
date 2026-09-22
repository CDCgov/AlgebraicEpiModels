using Test
using ConfigurableEpi
using StaticArrays

@testset "Latent Dynamics" begin
    @testset "get_param_value" begin
        hyperparams = (sigma = 0.1,)
        latent = (Rt = 1.5, rho = 0.9)
        params = merge(hyperparams, latent)

        # FixedParam returns stored value
        fixed = FixedParam(:x, 42.0)
        @test get_param_value(fixed, params) == 42.0

        # HyperParam looks up by name in params
        hyper = HyperParam(positive_gaussian(:sigma, 0.1, 0.01))
        @test get_param_value(hyper, params) == 0.1

        # ProcessParamSpec looks up by name in params
        proc = RWParamSpec(:Rt; init = unconstrained_gaussian(:Rt, 0.0, 1.0), sigma_rate = FixedParam(:s, 0.1))
        @test get_param_value(proc, params) == 1.5
    end

    @testset "build_stochastic_update (unconstrained storage)" begin
        layout = StateLayout((:S, :I, :R), (:O_I,), (:Rt, :rho))

        specs = (
            RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.5), sigma_rate = FixedParam(:sigma_Rt, 0.1)),
            AR1ParamSpec(:rho; init = unconstrained_gaussian(:rho, 0.5, 0.1), mu = FixedParam(:mu_rho, 0.5), tau = FixedParam(:tau_rho, 0.9), sigma = FixedParam(:sigma_rho, 0.05)),
        )

        latent = build_stochastic_update(layout, specs)

        @test latent isa StochasticUpdate{2}
        @test latent.advance isa Function
        @test latent.extract isa Function
        @test latent.to_unconstrained isa Function
    end

    @testset "to_unconstrained and extract roundtrip" begin
        layout = StateLayout((:S, :I), (:O,), (:Rt, :beta))

        specs = (
            RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.5), sigma_rate = FixedParam(:s, 0.1)),
            RWParamSpec(:beta; init = unconstrained_gaussian(:beta, 0.0, 1.0), sigma_rate = FixedParam(:s, 0.05)),
        )

        latent = build_stochastic_update(layout, specs)

        # Convert constrained → unconstrained
        constrained = (Rt = 2.0, beta = 0.3)
        unc = latent.to_unconstrained(constrained)

        @test unc isa SVector{2}
        @test unc[1] ≈ log(2.0)  # log(Rt)
        @test unc[2] ≈ 0.3       # identity(beta)

        # Build state vector with unconstrained values
        state = [100.0, 10.0, 5.0, unc[1], unc[2]]

        # Extract constrained values
        extracted = latent.extract(state)

        @test extracted isa NamedTuple
        @test haskey(extracted, :Rt)
        @test haskey(extracted, :beta)
        @test extracted.Rt ≈ 2.0
        @test extracted.beta ≈ 0.3
    end

    @testset "update in unconstrained space with zero noise" begin
        layout = StateLayout((:S,), (:O,), (:Rt,))
        specs = (RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.5), sigma_rate = FixedParam(:s, 0.1)),)

        latent = build_stochastic_update(layout, specs)

        # Initialize with Rt = 2.0 in unconstrained space
        unc_init = latent.to_unconstrained((Rt = 2.0,))
        state = [100.0, 5.0, unc_init[1]]

        hyperparams = NamedTuple()
        w = [0.0]  # zero noise

        # advance returns the full pre-flow state [compartments | coefficients | jumps]; read the
        # coefficient slot back out (rng = nothing ⇒ no jumps).
        li = first(layout.latent_range)
        new_state = latent.advance(state, hyperparams, w, nothing, 0.0, 1.0)

        @test new_state isa AbstractVector
        # With zero noise, the coefficient (unconstrained) is unchanged.
        @test new_state[li] ≈ unc_init[1]
    end

    @testset "update in unconstrained space with noise" begin
        layout = StateLayout((:S,), (:O,), (:Rt,))
        specs = (RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.5), sigma_rate = FixedParam(:s, 0.1)),)

        latent = build_stochastic_update(layout, specs)

        # Rt = 2.0 → log(2.0) ≈ 0.693
        unc_init = latent.to_unconstrained((Rt = 2.0,))
        state = [100.0, 5.0, unc_init[1]]

        hyperparams = NamedTuple()
        w = [1.0]  # positive noise

        li = first(layout.latent_range)
        new_state = latent.advance(state, hyperparams, w, nothing, 0.0, 1.0)

        # RW: new_unc = old_unc + sigma * w = log(2) + 0.1 * 1.0
        expected_unc = log(2.0) + 0.1
        @test new_state[li] ≈ expected_unc

        # Extracting from the advanced state gives exp(new_unc).
        extracted = latent.extract(new_state)
        @test extracted.Rt ≈ exp(expected_unc)
    end

    @testset "update AR1 in unconstrained space" begin
        layout = StateLayout((:S,), (:O,), (:rho,))
        specs = (
            AR1ParamSpec(
                :rho; init = unconstrained_gaussian(:rho, 0.5, 0.1),
                mu = FixedParam(:mu_rho, 0.5), tau = FixedParam(:tau_rho, -1 / log(0.9)),
                sigma = FixedParam(:sigma_rho, 0.1),
            ),
        )

        latent = build_stochastic_update(layout, specs)

        # Start at rho = 0.8
        unc_init = latent.to_unconstrained((rho = 0.8,))
        state = [100.0, 5.0, unc_init[1]]

        hyperparams = NamedTuple()
        w = [0.0]  # zero noise → pure mean reversion

        li = first(layout.latent_range)
        new_state = latent.advance(state, hyperparams, w, nothing, 0.0, 1.0)

        # OU over dt = 1 with tau = -1/log(0.9), i.e. rho = 0.9:
        #   new = mu + rho * (old - mu) = 0.5 + 0.9 * (0.8 - 0.5) = 0.77
        @test new_state[li] ≈ 0.77
    end

    @testset "update_single RW" begin
        spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.5), sigma_rate = FixedParam(:s, 0.1))
        old_unc = log(2.0)
        w = 1.0
        hyperparams = NamedTuple()
        latent_constrained = (Rt = 2.0,)  # not used for RW
        params = merge(hyperparams, latent_constrained)

        # `sigma_rate` is per sqrt(day), so at dt = 1 the step sd is the rate itself...
        @test update_single(spec, old_unc, w, params, 1.0) ≈ log(2.0) + 0.1
        # ...and at dt = 4 it is rate * sqrt(4). A per-STEP sd would give 0.1 at both, which is
        # exactly the dt-dependence the rate parameterisation removes.
        @test update_single(spec, old_unc, w, params, 4.0) ≈ log(2.0) + 0.2
    end

    @testset "update_single AR1" begin
        spec = AR1ParamSpec(
            :x; init = positive_gaussian(:x, 1.0, 0.5), mu = FixedParam(:mu, 1.0),
            tau = FixedParam(:tau_x, -1 / log(0.9)), sigma = FixedParam(:s, 0.1),
        )
        old_unc = log(2.0)  # x = 2.0 in constrained space
        w = 0.0
        hyperparams = NamedTuple()
        latent_constrained = (x = 2.0,)
        params = merge(hyperparams, latent_constrained)

        new_unc = update_single(spec, old_unc, w, params, 1.0)

        # OU in unc space: link(mu) + rho * (old_unc - link(mu)), rho = exp(-dt/tau) = 0.9
        # = log(1.0) + 0.9 * (log(2.0) - log(1.0)) = 0.9 * log(2.0)
        @test new_unc ≈ 0.9 * log(2.0)
    end

    @testset "update_single AR1 — a learned rho is read from params each step" begin
        # `basic_seir` learns rho, so the AR1 must take its persistence from the resolved
        # hyperparameters (the UKF's optimised `p` / the PF's frozen per-particle tail) rather
        # than from a value baked in at build time. A regression to a FixedParam rho would make
        # the learned value silently inert — and would silently restore the forecast-variance
        # inflation this inference exists to remove.
        spec = AR1ParamSpec(
            :x; init = positive_gaussian(:x, 1.0, 0.5), mu = FixedParam(:mu, 1.0),
            tau = HyperParam(positive_gaussian(:x_tau, 4.35, 1.35)),
            sigma = HyperParam(positive_gaussian(:x_sigma_stat, 0.15, 0.05)),
        )
        old_unc = log(2.0)

        # link(mu) = log(1) = 0, so the update is rho * old_unc + sigma_stat * sqrt(1-rho^2) * w.
        short = update_single(spec, old_unc, 0.0, (x_tau = -1 / log(0.2), x_sigma_stat = 0.1), 1.0)
        sticky = update_single(spec, old_unc, 0.0, (x_tau = -1 / log(0.9), x_sigma_stat = 0.1), 1.0)
        @test short ≈ 0.2 * log(2.0)
        @test sticky ≈ 0.9 * log(2.0)
        @test abs(short) < abs(sticky)   # a shorter memory reverts further toward mu in one step

        # sigma is likewise live. Note it is the STATIONARY sd, so the injected noise is scaled by
        # sqrt(1 - rho^2) — the property that makes the stationary spread independent of the memory.
        rho = 0.2
        @test update_single(
            spec, old_unc, 1.0, (x_tau = -1 / log(rho), x_sigma_stat = 0.3), 1.0
        ) ≈ rho * log(2.0) + 0.3 * sqrt(1 - rho^2)
    end

    @testset "negative noise cannot cause domain error" begin
        layout = StateLayout((:S,), (:O,), (:Rt,))
        specs = (RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.5), sigma_rate = FixedParam(:s, 0.5)),)

        latent = build_stochastic_update(layout, specs)

        # Start at Rt = 1.0 → log(1.0) = 0
        unc_init = latent.to_unconstrained((Rt = 1.0,))
        @test unc_init[1] ≈ 0.0

        state = [100.0, 5.0, unc_init[1]]

        # Very negative noise that would make constrained value near 0
        w = [-5.0]

        # This should NOT throw - we're working in unconstrained space
        li = first(layout.latent_range)
        new_state = latent.advance(state, NamedTuple(), w, nothing, 0.0, 1.0)

        @test new_state[li] ≈ 0.0 - 0.5 * 5.0  # -2.5 (rate 0.5 x sqrt(dt=1) x w)

        # Extract gives very small but positive Rt
        extracted = latent.extract(new_state)
        @test extracted.Rt ≈ exp(-2.5)
        @test extracted.Rt > 0
    end

    @testset "bounded latent initial values reject exact boundary" begin
        layout = StateLayout((:S,), (:O,), (:rho,))
        specs = (RWParamSpec(:rho; init = unit_interval_gaussian(:rho, 0.5, 0.1), sigma_rate = FixedParam(:s, 0.1)),)
        latent = build_stochastic_update(layout, specs)

        @test_throws ArgumentError latent.to_unconstrained((rho = 0.0,))
        @test_throws ArgumentError latent.to_unconstrained((rho = 1.0,))
    end
end
