# ============================================================================
# FULL DYNAMICS - UKF predict step factory
# ============================================================================
#
# Uses a unified Petri-net vectorfield for both disease and observation-chain
# dynamics. Cumulative observation compartments remain in the ODE state, while
# latent states are propagated separately in unconstrained space.
# ============================================================================

"""
    build_R1(layout::StateLayout) -> Diagonal

Build the augmented process-noise covariance for the UKF predict step.

`R1` is the identity, sized to the genuinely stochastic processes: one unit
white-noise term per latent process plus one per reset-accumulator. All magnitudes
are applied inside `build_full_dynamics` (latent sigmas from the spec; the
accumulator whisker from `obs_jitter`), so `R1` stays `I` and the noise scales ride
in `p` — robust to index ordering and free for hyperparameter inference.
"""
function build_R1(layout::StateLayout)
    nw = n_latent(layout) + length(layout.accumulator_indices)
    return Diagonal(ones(nw))
end


"""
    build_full_dynamics(petri_vf!, latent_dynamics, layout; dt=1.0, supersample=2, obs_jitter=1.0)

Build the full augmented UKF dynamics updater from a unified Petri-net vectorfield.

The Petri net carries both core disease dynamics and the observation chain. The
terminal observation compartments (`layout.accumulator_indices`) are treated as
**reset-accumulators**: reset to 0 at the start of each step, they integrate the
report flux over the window so each holds that step's incidence directly — no
cross-step cumulation and no previous-cumulative `u` feedback. The augmented noise
is `w = [latent process noise (L); accumulator whisker (S)]`; `obs_jitter` scales
the whisker (kept small; only there to keep the accumulator covariance full-rank
for the RTS smoother).
"""
function build_full_dynamics(
        petri_vf!,
        stochastic::StochasticUpdate{L},
        layout::StateLayout{N, M, K, L2, S};
        dt::Float64 = 1.0,
        supersample::Int = 2,
        obs_jitter::Float64 = 1.0
    ) where {N, M, K, L, L2, S}
    @assert L == L2 "StochasticUpdate dimension ($L) must match layout ($L2)"

    unified_vf = build_unified_vf(petri_vf!, layout)
    ode_stepper = Rk4(unified_vf, dt; supersample = supersample)

    ode_range = 1:n_ode_states(layout)
    acc_indices = layout.accumulator_indices   # absolute indices of the reset-accumulators

    # Augmented noise: w = [coefficient process noise (1:L); accumulator whisker (L+1:L+S)].
    # `rng` (optional) drives jump drivers; the UKF calls this 5-arg (rng = nothing, no jumps).
    @inline function dynamics(x, u, p, t, w, rng = nothing)
        Ty = eltype(x)

        # The accumulator whisker below is `@inbounds`, so a short `w` reads PAST THE END rather
        # than erroring — and because `obs_jitter` may be 0 and `0 * NaN === NaN`, the garbage
        # surfaces as a NaN compartment several steps later with nothing naming the cause.
        # `S` is the number of reset-accumulators, which is NOT the number of observation SIGNALS:
        # an `AggregatedSignalSpec` sums several accumulator chains into one signal, so a caller
        # sizing `w` by the signal count is short by exactly that difference. Cost is one integer
        # compare per step against an ODE solve.
        length(w) == L + S || throw(
            DimensionMismatch(
                "process-noise vector has length $(length(w)); this model needs $(L + S) = " *
                    "$(L) latent coefficient(s) + $(S) reset-accumulator(s). Note the second " *
                    "term counts ACCUMULATORS, not observation signals — see `build_R1`, which " *
                    "sizes it correctly."
            )
        )

        # Noise → flow: apply the full Lévy increment at the START of the step (coefficients from `w`,
        # jumps from `rng`, seeding compartments), THEN integrate the deterministic flow, driving the
        # ODE with the SAME advanced coefficients the output carries.
        xa = stochastic.advance(x, p, w, rng, t, dt)     # [compartments(seeded) | coeff | jump]
        latent_constrained = stochastic.extract(xa)

        # Reset accumulators to 0, then integrate the window so each holds the
        # incidence accrued over THIS step (no cross-step cumulation, no `u`).
        x_ode = max.(view(xa, ode_range), Ty(1.0e-6))
        @inbounds for idx in acc_indices
            x_ode[idx] = zero(Ty)
        end
        x_ode_next = ode_stepper(x_ode, u, (p, latent_constrained), t)
        x_ode_proj = max.(x_ode_next, Ty(1.0e-6))

        # Accumulator whisker: a UKF-smoother regularizer (kept full-rank for the RTS smoother);
        # `obs_jitter = 0` on the PF path turns it off.
        @inbounds for (s, idx) in enumerate(acc_indices)
            x_ode_proj[idx] += Ty(obs_jitter) * w[L + s]
        end

        @inbounds for (k, idx) in enumerate(ode_range)
            xa[idx] = x_ode_proj[k]
        end
        return xa
    end

    return dynamics
end
