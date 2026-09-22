# The particle-filter adapters over the shared dynamics and observation model, and the online
# PF + Liu-West engine.

# Under `threads = true` LLPF propagates particles in `@threads :static` loops, so noisy draws
# come from one Xoshiro stream per thread, all seeded from a single draw of the caller's rng.
struct PerThreadRNGs{R <: Random.AbstractRNG}
    rngs::Vector{R}
end

function _per_thread_rngs(rng::Random.AbstractRNG)
    seeder = Random.Xoshiro(rand(rng, UInt64))
    return PerThreadRNGs([Random.Xoshiro(rand(seeder, UInt64)) for _ in 1:Threads.maxthreadid()])
end

@inline _local_rng(pool::PerThreadRNGs) = pool.rngs[Threads.threadid()]
@inline _local_rng(rng::Random.AbstractRNG) = rng

"""
    build_pf_dynamics(dynamics, layout; rng = Random.default_rng(), learned = nothing, threads = false)
        -> pf_dynamics(x, u, p, t, noise = false)

Adapt the augmented `dynamics` from [`build_full_dynamics`](@ref) to the `AdvancedParticleFilter`
convention: `noise = true` draws the process noise (and fires the jump drivers) from `rng`, or
from a per-thread pool when `threads = true`, so a seeded run reproduces for a fixed thread count.
With `learned` each particle's tail overrides `p` and is carried forward unchanged.
"""
function build_pf_dynamics(
        dynamics, layout::StateLayout; rng = Random.default_rng(), learned = nothing, threads::Bool = false,
    )
    nw = n_latent(layout) + length(layout.accumulator_indices)
    zerow = zeros(nw)
    slots = learned === nothing ? (1:0) : hyper_range(learned)
    noise_rng = threads ? _per_thread_rngs(rng) : rng
    @inline function pf_dynamics(x, u, p, t, noise = false)
        r = noise ? _local_rng(noise_rng) : nothing
        w = noise ? randn(r, nw) : zerow
        learned === nothing && return dynamics(x, u, p, t, w, r)
        return [dynamics(x, u, _effective_params(learned, p, x), t, w, r); x[slots]]
    end
    return pf_dynamics
end

"""
    build_pf_measurement(layout, obs_specs, stochastic; rng = Random.default_rng(), learned = nothing)
        -> pf_measure(x, u, p, t, noise = false)

The particle filter's measurement: the observation means (floored at `1e-6`), or with
`noise = true` a draw from the exact observation distribution via [`sample_observation`](@ref).
"""
function build_pf_measurement(
        layout::StateLayout, obs_specs::Tuple{Vararg{ObservationSpec}}, stochastic::StochasticUpdate;
        rng = Random.default_rng(), learned = nothing,
    )
    specs = _resolve_specs(obs_specs, layout)
    accumulators = layout.accumulator_indices
    extract = stochastic.extract
    n_val = Val(length(specs))
    @inline function pf_measure(x, u, p, t, noise = false)
        p_eff = _effective_params(learned, p, x)
        latent = extract(x)
        means = _observation_means(specs, x, latent, p_eff, t, accumulators, n_val)
        noise || return SVector(map(m -> max(m, 1.0e-6), means))
        return SVector(ntuple(i -> sample_observation(specs[i].noise_spec, means[i], latent, p_eff, t, rng), n_val))
    end
    return pf_measure
end

# Kish effective sample size, `(Σw)² / Σw²`.
function _effective_sample_size(weights)
    total, sq = sum(weights), sum(abs2, weights)
    return (total > 0 && sq > 0) ? Float64(total^2 / sq) : 0.0
end

function _weighted_sample_indices(weights, n_draws::Integer, rng)
    cumulative = cumsum(weights)
    total = cumulative[end]
    isfinite(total) && total > 0 || throw(ArgumentError("particle weights must have a positive finite sum, got $total"))
    return [searchsortedfirst(cumulative, rand(rng) * total) for _ in 1:n_draws]
end

# Initial particle distribution whose model state is a function of the particle's own sampled
# parameters: draw the tail from `tail_prior`, rebuild the state through `build_x0`, and add the
# per-compartment noise the shared `P0` would have applied.
struct ParameterDependentInitial{F, D, L, H, S}
    build_x0::F
    tail_prior::D
    learned::L
    base_hyperparams::H
    model_sd::S
end

Base.length(d::ParameterDependentInitial) = length(d.model_sd) + length(d.tail_prior)

function Base.rand(rng::Random.AbstractRNG, d::ParameterDependentInitial)
    θ = rand(rng, d.tail_prior)
    hyperparams = merge(d.base_hyperparams, d.learned.extract(vcat(zeros(length(d.model_sd)), θ)))
    x0 = d.build_x0(hyperparams)
    return vcat(x0 .+ d.model_sd .* randn(rng, length(x0)), θ)
end

"""
    PFLiuWestEngine

PF + Liu-West: one persistent particle cloud carrying the learned hyperparameters in its tail,
assimilating only the observations it has not yet seen.
"""
struct PFLiuWestEngine{M <: EpiModel, F, L <: LearnedHyperparams, U, R} <: InferenceEngine
    model::M
    settings::EngineSettings
    filter_cfg::PF
    hyper::LiuWest
    filter::F
    learned::L
    update!::U
    rng::R
end

function build_inference(
        filter::PF, hyper::LiuWest, model::EpiModel;
        dt, supersample = 2, n_ahead, n_draws = 2000, rng = Random.default_rng(),
    )
    _validate(filter)
    _validate(hyper)
    settings = EngineSettings(; dt, supersample, n_ahead, n_draws)
    assembly = Assembly(model, settings, 0.0)
    assembly.n_obs == 1 ||
        throw(ArgumentError("PF + LiuWest needs exactly one observation signal; got $(assembly.n_obs)"))
    layout, base = model.layout, model.hyperparams
    learned = build_learned_hyperparams(model.priors, layout)
    as_days(entries) = Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in entries)
    forgetting = merge(as_days(pairs(model.forgetting_memory_days)), as_days(hyper.forgetting_memory_days))
    update! = build_hyperparam_updater(
        learned; discount = hyper.discount, jitter_floor_fraction = hyper.jitter_floor_fraction,
        forgetting_memory_days = forgetting, rng, dt = settings.dt,
    )
    pf_dynamics = build_pf_dynamics(assembly.dynamics, layout; rng, learned, threads = filter.threads)
    pf_measure = build_pf_measurement(layout, model.observation, model.stochastic; rng, learned)
    logpdf = build_measurement_logpdf(layout, model.observation, model.stochastic; learned)

    θ0 = collect(learned.to_unconstrained(NamedTuple{learned.names}(Tuple(base[n] for n in learned.names))))
    learned_variance = [
        Float64(get(model.initial_learned_variance, name, prior_unconstrained_variance(prior)))
            for (name, prior) in zip(learned.names, learned.priors)
    ]
    P0 = _initial_state_covariance(layout, initial_state(model), learned_variance; latent_variance = model.initial_latent_variance)
    d = layout.total_dim
    d0 = ParameterDependentInitial(
        model.build_x0, MvNormal(θ0, Matrix(Diagonal(diag(P0)[(d + 1):end]))), learned, base, sqrt.(diag(P0)[1:d]),
    )
    pf = AdvancedParticleFilter(
        filter.n_particles, pf_dynamics, pf_measure, logpdf, nothing, d0;
        p = base, ny = 1, nu = 0, rng, Ts = settings.dt, threads = filter.threads,
    )
    reset!(pf)
    return PFLiuWestEngine(model, settings, filter, hyper, pf, learned, update!, rng)
end

function fit_forecast!(
        e::PFLiuWestEngine, observations, forecast_number;
        update_range = eachindex(observations), emit_forecast::Bool = true,
    )
    T = _validate_fit(observations, update_range, forecast_number)
    ys = _observation_vectors(observations)
    pf, layout, s, base, learned = e.filter, e.model.layout, e.settings, e.model.hyperparams, e.learned
    n_particles = e.filter_cfg.n_particles
    fitted, ess, latent_path = Float64[], Float64[], Vector{Vector{Float64}}()
    final_particles = final_weights = nothing
    n_updates = length(update_range)
    progress_interval = max(cld(n_updates, 10), 1)
    @info "PF: assimilating observations" forecast_number observations_to_assimilate = n_updates total_observations =
        T particles = n_particles
    for (progress, k) in enumerate(update_range)
        t = (k - 1) * s.dt
        correct!(pf, _NO_INPUT, ys[k], base, t)
        cloud, weights = particles(pf), expweights(pf)
        wsum = sum(weights)
        push!(ess, _effective_sample_size(weights))
        push!(fitted, sum(weights[i] * pf.measurement(cloud[i], _NO_INPUT, base, t, false)[1] for i in eachindex(cloud)) / wsum)
        push!(latent_path, [sum(weights[i] * cloud[i][slot] for i in eachindex(cloud)) / wsum for slot in layout.latent_range])
        if emit_forecast && k == last(update_range)
            final_particles, final_weights = deepcopy(cloud), copy(weights)
        end
        e.update!(state(pf).xprev, weights)
        predict!(pf, _NO_INPUT, base, t)
        (progress == 1 || progress == n_updates || progress % progress_interval == 0) &&
            @info "PF: assimilation progress" forecast_number completed = progress total = n_updates observation_index = k
    end
    emit_forecast || return (; quantiles = nothing, fitted_means = fitted, summary = _summary(), samples = nothing)

    @info "PF: generating forecast ensemble" forecast_number draws = s.n_draws horizons = s.n_ahead
    forecast_filter = deepcopy(pf)   # forecast sampling must not touch the live filter's RNG
    initial_states = final_particles[_weighted_sample_indices(final_weights, s.n_draws, forecast_filter.rng)]
    samples, latent_samples = forecast_ensemble(
        forecast_filter, initial_states, base;
        n_ahead = s.n_ahead, t0 = (T - 1) * s.dt, dt = s.dt, latent_range = layout.latent_range,
    )

    summary = _summary()
    function summarise!(name, values)
        for (stat, q) in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))
            push!(summary, (string(name), stat, quantile(values, q)))
        end
        push!(summary, (string(name), "mean", sum(values) / length(values)))
        return nothing
    end
    # `sd_ratio` is the filtered cloud's sd over the prior's (unconstrained): a collapsed cloud
    # reads far below 1 while its quantiles merely look precise.
    _, V = _weighted_theta_moments(_theta_block(final_particles, learned), final_weights)
    for (slot, name) in enumerate(learned.names)
        summarise!(name, [learned.extract(p)[name] for p in initial_states])
        push!(summary, (string(name), "sd_ratio", sqrt(max(V[slot, slot], 0.0) / prior_unconstrained_variance(learned.priors[slot]))))
    end
    derived = e.model.derived_hyperparameters
    if derived !== nothing && !isempty(initial_states)
        values = [derived(merge(base, learned.extract(p))) for p in initial_states]
        for name in keys(first(values))
            summarise!(name, [Float64(v[name]) for v in values])
        end
    end
    if !isempty(ess)
        fraction = ess ./ n_particles
        for (stat, value) in (("ess_frac_min", minimum(fraction)), ("ess_frac_median", quantile(fraction, 0.5)), ("ess_frac_final", last(fraction)))
            push!(summary, ("particle_filter", stat, value))
        end
        minimum(fraction) < 0.02 && @warn(
            "PF: effective sample size collapsed; Liu-West estimates are frozen, not converged. " *
                "Raise `jitter_floor_fraction`, lower `discount`, or add particles.",
            forecast_number, min_ess_fraction = minimum(fraction), n_particles,
        )
    end
    latent_sd = [_cloud_sd(view(latent_samples, h, :, l)) for h in 1:s.n_ahead, l in axes(latent_samples, 3)]
    append_latent_spread!(summary, layout.latent_names, latent_sd)
    append_latent_audit!(summary, layout.latent_names, latent_path, 1:length(layout.latent_names); dt = s.dt)
    @info "PF: forecast complete" forecast_number learned_parameters = collect(learned.names)
    return (; quantiles = forecast_quantiles(samples), fitted_means = fitted, summary, samples)
end
