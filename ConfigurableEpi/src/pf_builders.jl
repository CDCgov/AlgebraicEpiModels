# ============================================================================
# PARTICLE-FILTER BUILDERS — reuse the UKF dynamics & measurement model
# ============================================================================
#
# A bootstrap particle filter needs the same model pieces as the UKF, adapted to
# LowLevelParticleFilters' AdvancedParticleFilter calling convention:
#   dynamics:    (x, u, p, t, noise=false) -> x⁺
#   measurement: (x, u, p, t, noise=false) -> y
# where `noise=false` means deterministic propagation and `noise=true` means draw
# from the model. These builders return exactly those closures by REUSING the
# UKF's `build_full_dynamics` / `build_measurement_model` (the predict! closure)
# and `sample_observation` (true-distribution draws), so the two filters share one
# model. `build_inference` uses these adapters to construct the concrete filter;
# examples may also assemble `AdvancedParticleFilter` directly.
# ============================================================================

# ----------------------------------------------------------------------------
# Per-thread RNG pool for threaded particle propagation
# ----------------------------------------------------------------------------
#
# Under `threads = true` the `AdvancedParticleFilter` propagates particles inside
# `Threads.@threads :static` loops, so a single RNG shared by the dynamics closure would be a
# data race. Noisy draws instead go through one `Xoshiro` stream per thread id — valid because
# `:static` pins each loop chunk to one thread. The pool is seeded from the caller's `rng` and
# `:static` chunking is deterministic, so a seeded run reproduces exactly for a fixed
# `Threads.nthreads()`; changing the thread count changes which stream serves each particle.

struct PerThreadRNGs{R <: Random.AbstractRNG}
    rngs::Vector{R}
end

# Consume exactly ONE draw from the caller's rng regardless of thread count, so enabling
# threading (or changing `-t`) cannot shift the unrelated streams that share that rng
# (resampling, Liu-West jitter, initial cloud); the seeder then fans out per-thread seeds.
function _per_thread_rngs(rng::Random.AbstractRNG)
    seeder = Random.Xoshiro(rand(rng, UInt64))
    return PerThreadRNGs(
        [Random.Xoshiro(rand(seeder, UInt64)) for _ in 1:Threads.maxthreadid()]
    )
end

@inline _local_rng(pool::PerThreadRNGs) = pool.rngs[Threads.threadid()]
@inline _local_rng(rng::Random.AbstractRNG) = rng

"""
    build_pf_dynamics(dynamics, layout;
                      rng=Random.default_rng(), learned=nothing, threads=false)
        -> pf_dynamics(x, u, p, t, noise=false)

Build the particle-filter dynamics closure for `AdvancedParticleFilter` — a thin wrapper over the
prebuilt augmented `dynamics` returned by [`build_full_dynamics`](@ref). The
stochastic driver already carries the full Lévy noise (diffusion coefficients +
any jump drivers), so this adapter draws the noise, passes `rng` on a stochastic
step, and carries the Liu-West tail:

    pf_dynamics(x, u, p, t, noise) = dynamics(x, u, p, t, noise ? randn(nw) : zeros(nw), noise ? rng : nothing)

- `noise = false` (the default) → deterministic propagation (zeros), used by the
  filter's prediction-without-noise paths and by users.
- `noise = true` → a full stochastic transition (latent process noise + any jump drivers),
  used when propagating particles and simulating.

`rng` seeds the propagation noise. With `threads = false` (the builder default) noisy
draws come straight from `rng`, so a seeded serial run is bit-reproducible. Pass
`threads = true` whenever the enclosing `AdvancedParticleFilter` is constructed with
`threads = true` — its propagate loop then calls this closure concurrently, and draws
instead go through a per-thread pool of `Xoshiro` streams seeded from `rng` at build
time (`PerThreadRNGs`): thread-safe under LLPF's `@threads :static` loops and
reproducible for a fixed `Threads.nthreads()`.
"""
function build_pf_dynamics(
        dynamics,
        layout::StateLayout;
        rng = Random.default_rng(),
        learned = nothing,
        threads::Bool = false
    )
    nw = n_latent(layout) + length(layout.accumulator_indices)
    zerow = zeros(nw)  # reused on the deterministic path (never mutated downstream)
    learned_slots = learned === nothing ? (1:0) : hyper_range(learned)
    noise_rng = threads ? _per_thread_rngs(rng) : rng

    # Thin PF adapter over the one dynamics builder: draw the noise, pass the rng so jump drivers
    # fire (jumps live in `stochastic.advance`, not here), and carry the Liu-West tail as the outer
    # block.
    @inline function pf_dynamics(x, u, p, t, noise = false)
        r = noise ? _local_rng(noise_rng) : nothing  # `noise=false` ⇒ deterministic, no jump draws
        w = noise ? randn(r, nw) : zerow
        if learned === nothing
            return dynamics(x, u, p, t, w, r)
        else
            # Per-particle hyperparameters override the shared `p`; the static learned block is carried
            # forward unchanged (refreshed by the updater between correct! and predict!).
            p_eff = merge(p, learned.extract(x))
            state_next = dynamics(x, u, p_eff, t, w, r)   # drivers (incl. jumps) + flow, length total_dim
            return [state_next; x[learned_slots]]
        end
    end
    return pf_dynamics
end

"""
    build_pf_measurement(measure, n_noise, layout, obs_specs, latent_dynamics;
                         rng=Random.default_rng(), learned=nothing)
        -> pf_measure(x, u, p, t, noise=false)

Adapt a prebuilt augmented `measure` function from
[`build_measurement_model`](@ref) to the `AdvancedParticleFilter` convention.
`n_noise` is the third value returned by that builder.

- `noise = false` (the default) → the deterministic mean (reuses
  `build_measurement_model` with zero noise), i.e. the ascertainment-modified
  signal mean.
- `noise = true` → a draw from the TRUE observation distribution via
  `sample_observation` (exact Poisson / NegativeBinomial counts), used by
  `simulate` to generate model-consistent data.

This is the simulation/observation counterpart to `build_measurement_logpdf`,
which provides the particle weighting `g(x, u, y, p, t)`. Both reuse the same
observation specs, `compute_true_mean`, and `mean_modifier` machinery.
"""
function build_pf_measurement(
        measure,
        nv,
        layout::StateLayout{N, M, K, L, S},
        obs_specs::Tuple{Vararg{ObservationSpec}},
        latent_dynamics::StochasticUpdate{L};
        rng = Random.default_rng(),
        learned = nothing
    ) where {N, M, K, L, S}
    zerov = zeros(nv)

    resolved_specs = _resolve_obs_specs(obs_specs, Val(S))
    accumulator_indices = layout.accumulator_indices
    extract_fn = latent_dynamics.extract
    obs_dim_val = Val(length(obs_specs))

    @inline function pf_measure(x, u, p, t, noise = false)
        # Per-particle learned hyperparameters override the shared `p`.
        p_eff = learned === nothing ? p : merge(p, learned.extract(x))
        if !noise
            return measure(x, u, p_eff, t, zerov)  # deterministic mean
        end
        latent_nt = extract_fn(x)
        obs = ntuple(obs_dim_val) do i
            spec = resolved_specs[i]
            raw_mean = compute_true_mean(spec, x, accumulator_indices)
            # Through `observation_mean`, not the modifier alone: the additive baseline is part
            # of the mean the likelihood scores against (`build_measurement_logpdf`), so the
            # draw must carry it too.
            true_mean = observation_mean(spec, raw_mean, latent_nt, p_eff, t)
            sample_observation(spec.noise_spec, true_mean, latent_nt, p_eff, t, rng)
        end
        return _svector_from_tuple(obs_dim_val, obs)
    end

    return pf_measure
end
