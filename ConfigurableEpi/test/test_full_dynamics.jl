using Test
using ConfigurableEpi

# Augmented process noise is sized to the genuinely stochastic processes only:
# w = [latent noise (1:L); accumulator whisker (L+1:L+S)]. Derive its length from
# build_R1 so the tests track the package's sizing rather than hard-coding it.
_noise_dim(layout) = size(build_R1(layout), 1)

@testset "Full Dynamics" begin
    @testset "build_full_dynamics unified path" begin
        layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))

        function mock_petri_vf!(du, u, p, t)
            hyper, _latent = p  # full_dynamics threads (hyperparams, latent)
            beta = 0.1
            du[:S] = -beta * u[:S] * u[:I] / 1000.0
            du[:I] = beta * u[:S] * u[:I] / 1000.0
            du[:O_I_1] = hyper.obs_scale * u[:I]
            return nothing
        end

        Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.1))
        latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
        dynamics = build_full_dynamics(
            mock_petri_vf!,
            latent_dynamics,
            layout;
            dt = 1.0,
            supersample = 2
        )

        x = [990.0, 10.0, 5.0, 0.0]  # [S, I, O_I_1 (accumulator), Rt_unc]
        hyperparams = (sigma_Rt = 0.1, obs_scale = 0.1)
        w = zeros(_noise_dim(layout))  # nw = L + S = 2, NOT total_dim

        x_next = dynamics(x, nothing, hyperparams, 0.0, w)

        @test length(x_next) == layout.total_dim
        @test x_next[1] < x[1]   # S depletes
        @test x_next[2] > x[2]   # I grows
        @test x_next[3] > 0.0    # accumulator reset to 0, then integrates this step's incidence
        @test x_next[4] == x[4]  # latent unchanged with zero process noise
    end

    @testset "accumulators reset each step (no cross-step cumulation)" begin
        layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))

        function mock_petri_vf!(du, u, p, t)
            hyper, _latent = p
            du[:S] = 0.0
            du[:I] = 0.0
            du[:O_I_1] = hyper.obs_scale * u[:I]  # positive report flux
            return nothing
        end

        Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.1))
        latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
        dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout)

        hyperparams = (sigma_Rt = 0.1, obs_scale = 0.1)
        w = zeros(_noise_dim(layout))

        # Two states identical except for the prior accumulator value.
        x_small = [990.0, 10.0, 5.0, 0.0]
        x_large = [990.0, 10.0, 500.0, 0.0]

        next_small = dynamics(x_small, nothing, hyperparams, 0.0, w)
        next_large = dynamics(x_large, nothing, hyperparams, 0.0, w)

        # The reset makes the new accumulator independent of its prior value.
        @test next_small[3] ≈ next_large[3]
        @test next_small[3] > 0.0
    end

    @testset "latent state responds to its process-noise term w[1]" begin
        layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))

        function mock_petri_vf!(du, u, p, t)
            du[:S] = 0.0
            du[:I] = 0.0
            du[:O_I_1] = 0.0
            return nothing
        end

        Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.1))
        latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
        dynamics = build_full_dynamics(mock_petri_vf!, latent_dynamics, layout)

        x = [990.0, 10.0, 0.0, 0.0]  # Rt_unc = 0
        hyperparams = (sigma_Rt = 0.1,)

        # w[1] is the latent noise; w[L+1:end] are the accumulator whiskers.
        w_pos = zeros(_noise_dim(layout)); w_pos[1] = 1.0
        w_neg = zeros(_noise_dim(layout)); w_neg[1] = -1.0
        x_next_pos = dynamics(x, nothing, hyperparams, 0.0, w_pos)
        x_next_neg = dynamics(x, nothing, hyperparams, 0.0, w_neg)

        @test x_next_pos[end] > x[end]  # positive noise raises log-Rt
        @test x_next_neg[end] < x[end]
    end

    @testset "accumulator whisker responds to w[L+s]" begin
        layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))

        function mock_petri_vf!(du, u, p, t)
            du[:S] = 0.0
            du[:I] = 0.0
            du[:O_I_1] = 0.0
            return nothing
        end

        Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.1))
        latent_dynamics = build_stochastic_update(layout, (Rt_spec,))
        dynamics = build_full_dynamics(
            mock_petri_vf!, latent_dynamics, layout; obs_jitter = 1.0
        )

        x = [990.0, 10.0, 0.0, 0.0]
        hyperparams = (sigma_Rt = 0.1,)

        # The accumulator (state index 3) is obs position 1, so its whisker is
        # noise term w[L + 1] = w[2]; a positive draw lifts the accumulator.
        w0 = zeros(_noise_dim(layout))
        w_up = zeros(_noise_dim(layout)); w_up[2] = 1.0
        @test dynamics(x, nothing, hyperparams, 0.0, w_up)[3] >
            dynamics(x, nothing, hyperparams, 0.0, w0)[3]
    end
    @testset "a wrongly-sized noise vector is rejected, not read out of bounds" begin
        # REGRESSION. The accumulator whisker is `@inbounds`, so a short `w` used to read past the
        # end instead of erroring. With `obs_jitter = 0` and `0 * NaN === NaN`, that garbage
        # surfaced as a NaN compartment several steps later, naming nothing.
        #
        # The trap is that the second noise block counts RESET-ACCUMULATORS, not observation
        # SIGNALS: an `AggregatedSignalSpec` sums several accumulator chains into one signal, so a
        # caller sizing `w` by the signal count is short by exactly that difference.
        # `examples/prior_predictive_check.jl` did this and lost ~4% of `two_strain_escape` draws
        # to it, miscounted as prior implausibility rather than as a bug in the script.
        layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))

        function guard_vf!(du, u, p, t)
            du[:S] = 0.0
            du[:I] = 0.0
            du[:O_I_1] = 0.0
            return nothing
        end

        Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.1))
        stochastic = build_stochastic_update(layout, (Rt_spec,))
        dynamics = build_full_dynamics(guard_vf!, stochastic, layout; obs_jitter = 0.0)

        x = [990.0, 10.0, 0.0, 0.0]
        hyperparams = (sigma_Rt = 0.1,)
        n = _noise_dim(layout)                     # 1 latent + 1 accumulator

        @test all(isfinite, dynamics(x, nothing, hyperparams, 0.0, zeros(n)))
        @test_throws DimensionMismatch dynamics(x, nothing, hyperparams, 0.0, zeros(n - 1))
        @test_throws DimensionMismatch dynamics(x, nothing, hyperparams, 0.0, zeros(n + 1))

        # The message has to name the accumulator/signal distinction, or the next caller repeats
        # the mistake with a correctly-sized-looking number.
        message = try
            dynamics(x, nothing, hyperparams, 0.0, zeros(n - 1))
            ""
        catch err
            sprint(showerror, err)
        end
        @test occursin("reset-accumulator", message)
        @test occursin("not observation signals", message)
    end
end
