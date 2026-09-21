# ============================================================================
# TYPED INFERENCE ASSEMBLY
# ============================================================================

"""
    build_dynamics_and_measurement(
        petri_vf!, layout, obs_model, stochastic_dynamics;
        dt=1.0, supersample=2, obs_jitter=1.0,
    ) -> (dynamics, measure, n_observations, n_measurement_noise)

Build the augmented state dynamics and observation function shared by all
filter/hyperparameter-inference combinations.
"""
function build_dynamics_and_measurement(
        petri_vf!, layout, obs_model, stochastic_dynamics;
        dt::Real = 1.0,
        supersample::Integer = 2,
        obs_jitter::Real = 1.0,
    )
    dynamics = build_full_dynamics(
        petri_vf!, stochastic_dynamics, layout;
        dt = float(dt), supersample = Int(supersample), obs_jitter = float(obs_jitter),
    )
    measure, n_observations, n_measurement_noise = build_measurement_model(
        layout, obs_model, stochastic_dynamics
    )
    return dynamics, measure, n_observations, n_measurement_noise
end

# Default initial variance of a latent coefficient slot (unconstrained space), used when
# the model does not name that slot in `initial_latent_variance`. Deliberately wide: in a
# log chart 0.2 is sd 0.45, i.e. a ±1sd multiplicative spread of about [0.64, 1.57].
const DEFAULT_LATENT_VARIANCE = 0.2

"""
    DEFAULT_MODEL_RELATIVE_SD

Initial uncertainty on each compartment, as a fraction of that compartment's own initial value.

Compartment scales span many orders of magnitude within one model — `S` of order 1e7 alongside
an observation stage seeded at 0 — so a single **absolute** variance cannot be right for all of
them. The historical flat `1e-6` amounted to a standard deviation of 1e-3 on a compartment of
1e7, i.e. the initial state was pinned to ten significant figures and the filter could not revise
it at all. That matters now that `basic_seir` seeds `S(0)` at the endemic equilibrium
`N/R0_baseline`: a seed is a starting guess, and pinning it turns it into an unrevisable
assumption.

At 5% the ±1sd spread on `S` is a few percent of the population, comfortably larger than the
infected pool but far short of the immune fraction's own uncertainty.

Caveat: the perturbations are diagonal, so they do not preserve `sum(compartments) == N`, while
the force of infection normalises by a fixed `hyper.N`. At a few percent that inconsistency is
small; if this is raised a lot, revisit.
"""
const DEFAULT_MODEL_RELATIVE_SD = 0.05

# Variance floor, so a compartment seeded at exactly 0 (an empty observation stage) still leaves
# `P0` positive definite.
const MIN_MODEL_VARIANCE = 1.0e-6

# Initial augmented-state covariance. `x0_state` sets the scale of the compartment block (see
# [`DEFAULT_MODEL_RELATIVE_SD`](@ref)). `learned_variance` gives the per-slot initial variance of
# the learned-hyperparameter tail (unconstrained space), in the filter's learned-name order; an
# empty vector (the UKF path, no tail) reproduces the model+latent-only matrix.
# `latent_variance` overrides the initial variance of individual latent coefficient slots by name
# (`layout.latent_names` order), so a model can make its latent's initial spread follow that
# latent's own prior rather than the shared default.
function _initial_state_covariance(
        layout, x0_state, learned_variance::AbstractVector{<:Real} = Float64[];
        latent_variance::NamedTuple = (;),
        accumulator_variance::NamedTuple = (;),
        model_relative_sd::Real = DEFAULT_MODEL_RELATIVE_SD,
    )
    _check_known_names(latent_variance, layout.latent_names, "initial_latent_variance")
    _check_known_names(
        accumulator_variance, layout.signal_names, "initial_accumulator_variance"
    )
    model_relative_sd >= 0 || throw(
        ArgumentError("model_relative_sd must be non-negative, got $model_relative_sd")
    )
    model_variance = [
        max((model_relative_sd * float(x0_state[i]))^2, MIN_MODEL_VARIANCE)
            for i in eachindex(ode_names(layout))
    ]
    # Reset accumulators are the one block whose initial value is known to be WRONG rather than
    # uncertain: they are 0 at t = 0 and only mean "this window's incidence" once a window has been
    # integrated. The relative-sd rule above therefore floors them at `MIN_MODEL_VARIANCE`, i.e.
    # pins them at zero — and `forward_trajectory` corrects before it ever predicts, so the first
    # innovation is scored against a predicted observation of ~0 with almost no variance.
    #
    # A Gaussian filter tolerates that: its compartment/accumulator cross-covariance is EXACTLY
    # zero at t = 0, so the analytic gain puts nothing into the compartments. An ensemble filter
    # cannot represent an exact zero — its sample cross-covariance is Monte-Carlo noise, and
    # dividing that by a near-zero innovation variance produces a spurious gain large enough to
    # throw the compartments to 1e10 and the next step to NaN. Hence this override, which the
    # ensemble path supplies and the Gaussian paths leave alone (so their `P0` is unchanged).
    for (signal, slot) in zip(layout.signal_names, layout.accumulator_indices)
        haskey(accumulator_variance, signal) || continue
        variance = Float64(accumulator_variance[signal])
        variance >= 0 || throw(
            ArgumentError(
                "initial_accumulator_variance[$signal] must be non-negative, got $variance"
            )
        )
        model_variance[slot] = max(variance, MIN_MODEL_VARIANCE)
    end
    latent_variances = [
        Float64(get(latent_variance, name, DEFAULT_LATENT_VARIANCE))
            for name in layout.latent_names
    ]
    return Matrix(
        Diagonal(vcat(model_variance, latent_variances, collect(float.(learned_variance)))),
    )
end

# A by-name override map must only name slots that exist — otherwise a typo silently keeps
# the default, which is exactly the kind of drift the strict config boundary exists to stop.
function _check_known_names(overrides::NamedTuple, known, what::AbstractString)
    for name in keys(overrides)
        name in known || throw(
            ArgumentError(
                "$what names `$name`, which is not one of $(collect(known))",
            )
        )
    end
    return nothing
end

"""
    prior_unconstrained_variance(prior) -> Float64

The variance of a scalar EKP `ParameterDistribution` in **unconstrained** space — the chart
both the Liu-West θ-tail and the latent coefficient slots are stored in.

This is a learned slot's DEFAULT initial θ-cloud variance, so the prior's `sd` sets the
initial spread instead of a magic constant; `initial_learned_variance` overrides it per name
when a wider exploration cloud is wanted (e.g. a logit-scale `escape_mean` whose truth would
otherwise sit in the tail). Public because submodels live outside `src/` and need it to make
a latent's `initial_latent_variance` follow that latent's own prior.
"""
prior_unconstrained_variance(prior) =
    (v = var(prior); v isa Number ? Float64(v) : Float64(only(v)))

"""
    prior_unconstrained_mean(prior) -> Float64

The mean of a scalar EKP `ParameterDistribution` in **unconstrained** space: the point the
Liu-West forgetting pulls a learned parameter back toward (see [`LiuWest`](@ref)).
"""
prior_unconstrained_mean(prior) =
    (m = mean(prior); m isa Number ? Float64(m) : Float64(only(m)))

"""
    _fitted_observation_means(specs, accumulator_indices, extract, xt, hyper, dt, update_range)

Fitted observation means for the reporting tables: each filtered state mean `xt[k]` pushed through
its spec's `observation_mean` at that observation's OWN model time `(k - 1) * dt` — the same mapping
the measurement function applies, so a time-varying ascertainment is honoured. Returns a `Vector`
for one spec (the historical single-signal shape) and a `[time, observation]` matrix otherwise.
Deliberately `compute_true_mean` + `observation_mean` rather than the measurement closure at zero
noise, whose `apply_noise` floors the mean at `1e-6`.
"""
function _fitted_observation_means(
        specs, accumulator_indices, extract, xt, hyper, dt, update_range
    )
    means = [
        observation_mean(
            spec, compute_true_mean(spec, xt[k], accumulator_indices), extract(xt[k]), hyper,
            (k - 1) * dt,
        ) for k in update_range, spec in specs
    ]
    return length(specs) == 1 ? vec(means) : means
end

function _check_initial_state(layout, x0_state, n_extra::Integer = 0)
    expected = layout.total_dim + n_extra
    length(x0_state) == expected || throw(
        DimensionMismatch(
            "initial state has length $(length(x0_state)); expected $expected " *
                "($(layout.total_dim) model slots + $n_extra learned slots)",
        )
    )
    return nothing
end

"""
    ParameterDependentInitial(build_x0, tail_prior, learned, base_hyperparams, model_sd)

Initial particle distribution in which the model state is a **function of that particle's own
sampled parameters**, rather than a shared constant.
Sampling order here is parameters first, then state: draw the tail from `tail_prior` (unconstrained),
map it to constrained values, rebuild the model state through `build_x0`, and add the same
relative per-compartment noise the shared `P0` would have applied. `reset!` on a particle filter
only needs `rand`, so this composes with mid-run resets (e.g. a revision replay) automatically.
"""
struct ParameterDependentInitial{F, D, L, H, S}
    build_x0::F
    tail_prior::D
    learned::L
    base_hyperparams::H
    model_sd::S
end

Base.length(d::ParameterDependentInitial) = length(d.model_sd) + length(d.tail_prior)

function Base.rand(rng::Random.AbstractRNG, d::ParameterDependentInitial)
    theta_unconstrained = rand(rng, d.tail_prior)
    # `extract` reads only the tail slots, so a zero-padded model block is enough to decode.
    probe = vcat(zeros(length(d.model_sd)), theta_unconstrained)
    hyperparams = merge(d.base_hyperparams, d.learned.extract(probe))
    x0 = d.build_x0(hyperparams)
    return vcat(x0 .+ d.model_sd .* randn(rng, length(x0)), theta_unconstrained)
end

"""
    positive_cholesky!(R)

Positive-definite Cholesky of `R` via `PositiveFactorizations`, usable under automatic
differentiation.

`PositiveFactorizations.cholesky!(Positive, A)` is signature-bounded to
`AbstractMatrix{T} where T<:AbstractFloat`, and `ForwardDiff.Dual <: Real` but **not**
`<: AbstractFloat`, so the UKF's cholesky hook would reject Dual-valued covariances. The
`ldlt!` it delegates to is already generic (`where {T}`, self-checking `eltype(A)<:Real`), so
this calls that directly — same algorithm and same result, minus the bound. Doing it here
rather than adding a method to `cholesky!` avoids pirating a function/type pair we don't own
(which Aqua would rightly flag).
"""
function positive_cholesky!(R)
    A = Matrix(R)
    T = eltype(A)
    return ldlt!(
        Positive{T}, A;
        tol = default_tol(A), blocksize = default_blocksize(floattype(T)),
    )[1]
end

"""
    marginal_loglik(kf, us, ys, p) -> Real

Filter marginal log-likelihood of `ys` under hyperparameters `p`, accumulated at the promoted
element type so it can be differentiated.

This exists instead of `forward_trajectory(kf, us, ys, p).ll` because that convenience wrapper
sizes its own buffers from the **observations** — `e = similar(y)` and
`ll = zero(eltype(particletype(kf)))` — which are `Float64`, so a Dual-valued `p` dies writing
back into them. The hyperparameter objective only ever wants `.ll`, so run the
`correct!`/`predict!` recursion directly and skip the buffers. `forward_trajectory` is still the
right call for the actual filtering pass, which needs the smoothed/filtered trajectories.
"""
function marginal_loglik(kf, us, ys, p)
    reset!(kf)
    return _filter_loglik!(kf, us, ys, p, eachindex(ys))
end

_loglik_zero(kf::AbstractKalmanFilter) = zero(eltype(state(kf)))
_loglik_zero(pf::LLPF.AbstractParticleFilter) = zero(eltype(LLPF.weights(pf)))

"""
    _filter_loglik!(filter, us, ys, p, update_range) -> Real

Advance an already-initialized filter through the original observation indices in
`update_range`, returning their marginal log-likelihood. This deliberately does not call
`reset!`: a full replay is initialized by [`marginal_loglik`](@ref), while a windowed replay
starts from a retained filter checkpoint. Keeping the original indices preserves absolute model
time when a window starts after the first observation.
"""
function _filter_loglik!(filter, us, ys, p, update_range)
    ll = _loglik_zero(filter)
    for k in update_range
        time = (k - 1) * filter.Ts
        lli = first(correct!(filter, us[k], ys[k], p, time))
        ll += lli
        predict!(filter, us[k], p, time)
    end
    return ll
end

function _validate_window_length(window_length)
    window_length === nothing && return nothing
    window_length > 0 || throw(
        ArgumentError("window_length must be positive when provided, got $window_length")
    )
    return Int(window_length)
end

_loss_range(T::Integer, ::Nothing) = 1:Int(T)
_loss_range(T::Integer, window_length::Integer) =
    max(1, Int(T) - Int(window_length) + 1):Int(T)

# A window checkpoint is a complete, ordinary LLPF filter stopped immediately before the first
# scored correction. It is retained in memory and advanced only when the left edge of the window
# moves; no trajectory of per-step copies is built. Value revisions that have already moved behind
# the checkpoint are intentionally ignored.
#
# A checkpoint is built EXACTLY ONCE, when the window first becomes active, and thereafter can only
# move FORWARD through the same history. There is deliberately no rebuild-on-invalidation path.
# Re-running the prefix at the current parameters computes the constant-θ prefix that
# `window_length = nothing` already computes honestly, so a silent fallback would swap one estimand
# for another under a name that claims otherwise — and it would re-apply `build_x0`, whose
# equilibrium seed `S(0) = N / R0` is only meant to act once, at the real start of the data. When
# the checkpoint cannot be advanced the caller is told to ask for full replay explicitly.
function _prepare_window_checkpoint!(
        checkpoint,
        checkpoint_start,
        checkpoint_date,
        build_anchor,
        p,
        us,
        ys,
        dates,
        loss_range,
    )
    new_start = first(loss_range)
    if new_start == 1
        checkpoint[] = nothing
        checkpoint_start[] = 1
        checkpoint_date[] = nothing
        return nothing, :full_history, 0
    end

    if checkpoint[] === nothing
        # Cold start: no earlier state exists to be faithful to, so filtering the prefix once at
        # the current parameters is initialization rather than recovery. The only build.
        anchor = build_anchor()
        reset!(anchor)
        _filter_loglik!(anchor, us, ys, p, 1:(new_start - 1))
        checkpoint[] = anchor
        checkpoint_start[] = new_start
        checkpoint_date[] = dates[new_start]
        return anchor, :anchored, new_start - 1
    end

    old_start = checkpoint_start[]
    saved_date = checkpoint_date[]
    new_start >= old_start || error(
        "window checkpoint cannot move backwards: it stands before observation $(old_start) " *
            "but the window now starts at $(new_start). Set `window_length = nothing` to score " *
            "the complete history instead."
    )
    old_start <= length(dates) && saved_date !== nothing && dates[old_start] == saved_date ||
        error(
        "window checkpoint no longer matches the observation history: it was taken before " *
            "$(saved_date) at observation $(old_start), which now carries " *
            "$(old_start <= length(dates) ? string(dates[old_start]) : "no observation"). " *
            "Set `window_length = nothing` to score the complete history instead."
    )

    new_start == old_start && return checkpoint[], :advanced, 0

    # Advance a COPY and swap on success. `correct!` errors outright when the innovation covariance
    # loses rank, and advancing in place would leave the retained filter part-way through the
    # prefix while `checkpoint_start` still named its old position — after which the next origin
    # would silently replay that stretch a second time onto an already-advanced state. One copy per
    # recalibration, against one per candidate, so the cost is noise.
    advancing = deepcopy(checkpoint[])
    _filter_loglik!(advancing, us, ys, p, old_start:(new_start - 1))
    checkpoint[] = advancing
    checkpoint_start[] = new_start
    checkpoint_date[] = dates[new_start]
    return advancing, :advanced, new_start - old_start
end

# Gaussian candidate filters used by ForwardDiff must be constructed at the candidate element
# type. The pre-window state is a constant with respect to the new parameters, so assignment into
# the typed buffers correctly gives it zero partials — and building through `build_candidate`
# already carries the candidate's own `p`.
function _candidate_from_checkpoint(
        checkpoint::LLPF.AbstractUnscentedKalmanFilter, build_candidate, ::Type{T}, p
    ) where {T}
    candidate = build_candidate(T, p)
    candidate.x = typeof(candidate.x)(state(checkpoint))
    candidate.R .= covariance(checkpoint)
    candidate.t = LLPF.index(checkpoint)
    return candidate
end

# Ensemble candidates stay Float64 and copy the complete in-place filter, including its RNG
# position. `p` is reassigned to the candidate's own parameters: `_filter_loglik!` passes them
# explicitly to every `correct!`/`predict!`, so a stale `f.p` cannot change a loss today, but it
# would leave the copy disagreeing with itself for any call that falls back on `parameters(f)` —
# and `enkf.p` is load-bearing on the live filter, so the trap is a real one to leave lying around.
function _candidate_from_checkpoint(
        checkpoint::AugmentedEnsembleKalmanFilter, _build_candidate, _state_type, p
    )
    candidate = deepcopy(checkpoint)
    candidate.p = p
    return candidate
end

function _candidate_from_checkpoint(
        checkpoint::LLPF.EnsembleKalmanFilter, _build_candidate, _state_type, p
    )
    candidate = deepcopy(checkpoint)
    candidate.p = p
    return candidate
end

# `AdvancedParticleFilter` is an immutable struct, so `p` cannot be reassigned on a copy. No
# windowed PF path is wired up today, and `_filter_loglik!` passes `p` explicitly, so the copy is
# correct as it stands — but wiring one up must RECONSTRUCT the filter at the candidate's
# parameters rather than lean on this method.
_candidate_from_checkpoint(
    checkpoint::LLPF.AbstractParticleFilter, _build_candidate, _state_type, _p
) = deepcopy(checkpoint)

# Objective value standing in for "the filter diverged here".
#
# This MUST be `Inf`, not a large finite constant. The filter marginal log-posterior has no
# bounded scale — on `basic_seir` it is ~5e10 at the start point — so any finite penalty risks
# sitting *below* the feasible objective, at which point the optimizer is actively rewarded for
# driving the filter to diverge. (A 1e10 penalty did exactly that: NelderMead never explored far
# enough to notice, but Adam/LBFGS found it immediately.)
#
# `Inf` is safe for gradients here precisely because the UKF objective evaluates its prior term
# OUTSIDE the `try`: the penalty branch returns `Inf - log_prior`, i.e. value `Inf` with finite
# partials from the prior. A line search backtracks on the `Inf` value, and any gradient that is
# requested points back toward the prior mode rather than being `NaN`.
const _DIVERGED_PENALTY = Inf

function _check_single_signal(n_observations::Integer)
    n_observations == 1 || throw(
        ArgumentError(
            "build_inference currently produces one count-forecast matrix and " *
                "therefore requires exactly one observation signal; got $n_observations",
        )
    )
    return nothing
end

"""
    _ukf_backend([T=Float64,] dynamics, measure, layout, x0_state, dt,
                 n_observations, n_measurement_noise, base_hyperparams,
                 initial_latent_variance)

Assemble the concrete `UnscentedKalmanFilter` used by the
`UKF` + `OptimiseHyperparams` inference combination. `initial_latent_variance` overrides
the initial (unconstrained) variance of individual latent coefficient slots by name.

`T` is the **element type of the filter's own state buffers** (`x`, `R`, `R1`, `R2`, `P0`).
The filter is a preallocated mutable struct mutated in place, so differentiating the
hyperparameter objective needs it built at the AD element type — a `Float64`-typed filter
rejects a Dual-valued update with `MethodError: no method matching Float64(::Dual)`. The
optimizer therefore rebuilds a filter per objective evaluation via this factory; the cost is
one small allocation against a full-window filter pass.
"""
function _ukf_backend(
        ::Type{T},
        dynamics,
        measure,
        layout,
        x0_state,
        dt,
        n_observations,
        n_measurement_noise,
        base_hyperparams,
        initial_latent_variance,
    ) where {T}
    _check_initial_state(layout, x0_state)
    R1 = T.(Matrix(build_R1(layout)))
    R2 = Matrix{T}(I, n_measurement_noise, n_measurement_noise)
    P0 = T.(
        _initial_state_covariance(
            layout, x0_state; latent_variance = initial_latent_variance
        )
    )
    return UnscentedKalmanFilter{false, false, true, true}(
        dynamics, measure, R1, R2, MvNormal(T.(x0_state), P0);
        p = base_hyperparams,
        ny = n_observations,
        nu = 0,
        weight_params = TrivialParams(),
        Ts = float(dt),
        cholesky! = positive_cholesky!,
    )
end

_ukf_backend(dynamics, args...) = _ukf_backend(Float64, dynamics, args...)

"""
    _pf_backend(filter_method, hyper_method, dynamics, measure,
                n_observations, n_measurement_noise, layout,
                stochastic_dynamics, obs_model, base_hyperparams,
                x0_state, rng, initial_learned_variance,
                initial_latent_variance) -> (pf, learned, updater!)

Assemble an `AdvancedParticleFilter` whose particle tail contains the complete
joint `LiuWest` prior block. The returned updater is paired with this filter and
uses the same RNG. `initial_learned_variance` overrides the initial (unconstrained)
variance of individual learned slots by name; unnamed slots default to each prior's
own unconstrained variance. `initial_latent_variance` does the same for the latent
coefficient slots (default `DEFAULT_LATENT_VARIANCE`).
"""
function _pf_backend(
        filter_method::PF,
        hyper_method::LiuWest,
        dynamics,
        measure,
        n_observations,
        n_measurement_noise,
        layout,
        stochastic_dynamics,
        obs_model,
        base_hyperparams,
        x0_state,
        build_x0,
        rng,
        initial_learned_variance,
        initial_latent_variance,
    )
    learned = build_learned_hyperparams(hyper_method, layout)
    # `dt` makes any configured forgetting a rate per DAY rather than per filter step.
    updater! = build_hyperparam_updater(learned; rng, dt = filter_method.dt)
    pf_dynamics = build_pf_dynamics(
        dynamics, layout; rng, learned, threads = filter_method.threads
    )
    pf_measure = build_pf_measurement(
        measure,
        n_measurement_noise,
        layout,
        obs_model,
        stochastic_dynamics;
        rng,
        learned,
    )
    measurement_logpdf = build_measurement_logpdf(
        layout, obs_model, stochastic_dynamics; learned
    )

    initial_learned = collect(
        learned.to_unconstrained(
            (; (name => base_hyperparams[name] for name in learned.names)...)
        )
    )
    x0 = vcat(x0_state, initial_learned)
    _check_initial_state(layout, x0, n_learned(learned))
    learned_variance = [
        Float64(get(initial_learned_variance, name, prior_unconstrained_variance(prior)))
            for (name, prior) in zip(learned.names, learned.priors)
    ]
    P0 = _initial_state_covariance(
        layout, x0_state, learned_variance; latent_variance = initial_latent_variance
    )
    # Model state per particle from that particle's own θ (see `ParameterDependentInitial`);
    # `P0`'s model block supplies the per-compartment spread, its tail block the θ prior.
    model_dim = layout.total_dim
    d0 = ParameterDependentInitial(
        build_x0,
        MvNormal(
            initial_learned,
            Matrix(Diagonal(diag(P0)[(model_dim + 1):end])),
        ),
        learned,
        base_hyperparams,
        sqrt.(diag(P0)[1:model_dim]),
    )
    pf = AdvancedParticleFilter(
        filter_method.n_particles,
        pf_dynamics,
        pf_measure,
        measurement_logpdf,
        nothing,
        d0;
        p = base_hyperparams,
        ny = n_observations,
        nu = 0,
        rng,
        Ts = filter_method.dt,
        threads = filter_method.threads,
    )
    reset!(pf)
    return pf, learned, updater!
end

function _symmetrize_covariance(R)
    matrix = Matrix(R)
    return (matrix .+ matrix') ./ 2 +
        1.0e-8 * Diagonal(ones(eltype(matrix), size(matrix, 1)))
end

# Across-cloud sd for the forecast-spread diagnostic. `std` needs ≥ 2 points, and a
# single-draw forecast is legal (`n_draws = 1`), so report a degenerate cloud as 0 rather
# than writing NaN into the summary CSV.
_cloud_sd(values) = length(values) < 2 ? 0.0 : Float64(std(values))

# Kish effective sample size of a weight vector: (Σw)² / Σw². Equals `n` for uniform weights and
# 1 when a single particle carries everything, so `ESS / n` reads directly as "what fraction of
# the cloud is actually doing work".
function _effective_sample_size(weights)
    total = sum(weights)
    sq = sum(abs2, weights)
    return (total > 0 && sq > 0) ? Float64(total^2 / sq) : 0.0
end

function _weighted_sample_indices(weights, n_draws::Integer, rng)
    n_draws > 0 || throw(ArgumentError("n_draws must be positive, got $n_draws"))
    cumulative = cumsum(weights)
    total = cumulative[end]
    isfinite(total) && total > 0 || throw(
        ArgumentError("particle weights must have a positive finite sum, got $total")
    )
    return [searchsortedfirst(cumulative, rand(rng) * total) for _ in 1:n_draws]
end

"""
    _observation_vectors(counts) -> Vector{Vector{Float64}}

LLPF observation vectors from an `asof` frame's `counts` column.

Two methods, dispatched on what the column holds. A column of numbers is the single-signal case and
reproduces the historical `[[Float64(c)] for c in counts]` exactly. A column of vectors is the
multi-observation case: `run_backtest` widens a long (date, location, counts) frame to one row per
date whose `counts` cell is the per-location vector in the model's signal order, so the observation
for each date is already assembled.

Widening before the origin loop is deliberate. `nrow(asof)`, the burn-in gate, `drop_recent_weeks`
and `_validate_fit_inputs` all count ROWS, which with six locations per week would be six times
wrong; widening first makes a row a date again, so none of those need to know about locations.
"""
_observation_vectors(counts::AbstractVector{<:Real}) =
    [[Float64(count)] for count in counts]
_observation_vectors(counts::AbstractVector{<:AbstractVector}) =
    [collect(Float64, count) for count in counts]

function _validate_fit_inputs(asof, update_range, us)
    n_observations = length(asof.counts)
    length(us) == n_observations || throw(
        DimensionMismatch(
            "observations and inputs must have equal lengths; got " *
                "$n_observations and $(length(us))",
        )
    )
    isempty(update_range) && throw(ArgumentError("update_range must not be empty"))
    first(update_range) >= 1 && last(update_range) <= n_observations || throw(
        BoundsError(asof.counts, update_range)
    )
    last(update_range) == n_observations || throw(
        ArgumentError(
            "update_range must end at the forecast origin $n_observations; " *
                "got $(first(update_range)):$(last(update_range))",
        )
    )
    return n_observations
end

"""
    build_inference(filter_method, hyper_method, args...; kwargs...)

Build a concrete state filter and its paired in-place fit/forecast callable.
Supported combinations are:

- [`UKF`](@ref) + [`OptimiseHyperparams`](@ref)
- [`PF`](@ref) + [`LiuWest`](@ref)
- [`EnKF`](@ref) + [`EKPCalibration`](@ref)

Every supported method returns `(filter, fit_forecast!)`. The callable has the
common contract

```julia
fit_forecast!(filter, asof, update_range, us, forecast_number;
              emit_forecast=true, sample_callback=nothing)
    -> (forecast_quantiles, mean_updates, hyperparameter_summary)
```

`update_range` identifies the observations assimilated by this call and
`mean_updates` has the same length. The caller owns the observation-history
cursor and the complete vector of fitted weekly means. The callable must be
used with the filter returned alongside it. With `emit_forecast=false`, filtering and
hyperparameter learning still occur, but forecast simulation is skipped and the first return
value is `nothing`. Ensemble engines call `sample_callback(samples)`, when supplied, with their
coherent `[horizon, draw]` (or `[horizon, draw, signal]`) predictive sample array before returning.
The analytic UKF path rejects a callback because it does not generate joint forecast draws.
"""
function build_inference(
        filter_method::StateFilterMethod,
        hyper_method::HyperparamInferenceMethod,
        args...;
        kwargs...,
    )
    throw(
        ArgumentError(
            "build_inference is not implemented for " *
                "$(typeof(filter_method)) + $(typeof(hyper_method))",
        )
    )
end

"""
    build_inference(
        filter_method::UKF, hyper_method::OptimiseHyperparams,
        petri_vf!, layout, stochastic_dynamics, obs_model,
        base_hyperparams, x0_state;
        n_ahead, reopt_interval=1, window_length=nothing, warm_start=true,
        initial_optim_options=(;), initial_latent_variance=(;),
    ) -> (ukf, fit_forecast!)

Build the UKF inference path. At every origin the live UKF replays the complete series; every
`reopt_interval` origins it maximizes the marginal posterior over `hyper_method.priors`.
`window_length` optionally limits that objective to the most recent observations, starting from a
rolling in-memory filter checkpoint. `warm_start` (default `true`) starts each re-optimization
from the previous optimum; `false` restarts from `base_hyperparams` every time and gives every
re-optimization the `initial_optim_options` budget. `initial_optim_options` override ordinary
optimizer options on the first origin (and on every cold-started one).
`initial_latent_variance` sets the initial (unconstrained) variance of named latent coefficient
slots, defaulting to `DEFAULT_LATENT_VARIANCE`.
"""
function build_inference(
        filter_method::UKF,
        hyper_method::OptimiseHyperparams,
        petri_vf!,
        layout,
        stochastic_dynamics,
        obs_model,
        base_hyperparams,
        x0_state;
        n_ahead::Integer,
        reopt_interval::Integer = 1,
        window_length::Union{Nothing, Integer} = nothing,
        warm_start::Bool = true,
        initial_optim_options::NamedTuple = (;),
        initial_latent_variance::NamedTuple = (;),
        build_x0 = _ -> x0_state,
    )
    n_ahead > 0 || throw(ArgumentError("n_ahead must be positive, got $n_ahead"))
    reopt_interval > 0 || throw(
        ArgumentError("reopt_interval must be positive, got $reopt_interval")
    )
    validated_window_length = _validate_window_length(window_length)
    dynamics, measure, n_observations, n_measurement_noise =
        build_dynamics_and_measurement(
        petri_vf!, layout, obs_model, stochastic_dynamics;
        dt = filter_method.dt,
        supersample = filter_method.supersample,
        obs_jitter = filter_method.obs_jitter,
    )
    _check_single_signal(n_observations)
    # One factory, used twice: once for the live Float64 filter the caller drives, and once per
    # objective evaluation at the AD element type (the filter's state buffers are preallocated,
    # so the optimizer cannot reuse a Float64 one — see `_ukf_backend`).
    # `hp` sets BOTH the filter's parameters and its initial state — the equilibrium seed
    # `S(0) = N / R0` is only coherent when it uses the same `R0` the iterate carries, so a filter
    # built at one θ must not be reused at another.
    build_backend(::Type{E}, hp) where {E} = _ukf_backend(
        E,
        dynamics,
        measure,
        layout,
        build_x0(hp),
        filter_method.dt,
        n_observations,
        n_measurement_noise,
        hp,
        initial_latent_variance,
    )
    ukf = build_backend(Float64, base_hyperparams)
    priors = hyper_method.priors
    learned_names = priors.names
    accumulator = only(layout.accumulator_indices)
    # The (single) observation spec, resolved, and the constrained-latent extractor: the reporting
    # paths below push filtered and predicted accumulators through the SAME `observation_mean` the
    # measurement function applies, at each observation's own model time — never through a
    # hyperparameter read directly, which a time-varying ascertainment would silently contradict.
    observation_spec = only(_resolve_obs_specs(obs_model, Val(n_signals(layout))))
    extract_latent = stochastic_dynamics.extract
    ordinary_options = hyper_method.options
    initial_options = merge(ordinary_options, initial_optim_options)
    window_checkpoint = Ref{Union{Nothing, typeof(ukf)}}(nothing)
    window_checkpoint_start = Ref(1)
    window_checkpoint_date = Ref{Union{Nothing, Date}}(nothing)

    function fit_forecast!(
            ukf, asof, update_range, us, forecast_number;
            emit_forecast::Bool = true, sample_callback = nothing,
        )
        sample_callback === nothing || throw(
            ArgumentError(
                "UKF forecasts analytic marginal quantiles and cannot emit coherent samples"
            )
        )
        T = _validate_fit_inputs(asof, update_range, us)
        update_range == (1:T) || throw(
            ArgumentError(
                "UKF + OptimiseHyperparams requires the complete update range 1:$T; " *
                    "got $(first(update_range)):$(last(update_range))",
            )
        )
        forecast_number > 0 || throw(
            ArgumentError("forecast_number must be positive, got $forecast_number")
        )
        observations = _observation_vectors(asof.counts)

        if (forecast_number - 1) % reopt_interval == 0
            loss_range = _loss_range(T, validated_window_length)
            checkpoint_filter, checkpoint_mode, checkpoint_steps =
                _prepare_window_checkpoint!(
                window_checkpoint,
                window_checkpoint_start,
                window_checkpoint_date,
                () -> build_backend(Float64, ukf.p),
                ukf.p,
                us,
                observations,
                asof.date,
                loss_range,
            )
            # `ukf.p` carries the previous optimum (it is assigned from `result.θ` below and is
            # the only thing that persists across origins on this path), so reading the start
            # point from it IS the warm start. A cold start reads the same names out of
            # `base_hyperparams` instead — the configured starting values, untouched by any
            # earlier origin — which is what makes each estimate independent.
            start_from = warm_start ? ukf.p : base_hyperparams
            initial_values = (; (name => start_from[name] for name in learned_names)...)
            neg_logposterior = (unconstrained, _) -> begin
                log_prior = prior_logpdf(priors, unconstrained)
                try
                    hyperparameters = merge(
                        base_hyperparams,
                        constrained_values(priors, unconstrained),
                    )
                    # A filter at the caller's element type: `Float64` for a plain evaluation,
                    # `Dual` when the AD backend is differentiating. Never the live `ukf`, so
                    # the optimizer cannot leave it in a half-advanced state.
                    candidate_type = eltype(unconstrained)
                    kf = if checkpoint_filter === nothing
                        build_backend(candidate_type, hyperparameters)
                    else
                        _candidate_from_checkpoint(
                            checkpoint_filter, build_backend, candidate_type, hyperparameters
                        )
                    end
                    ll = checkpoint_filter === nothing ?
                        marginal_loglik(kf, us, observations, hyperparameters) :
                        _filter_loglik!(
                            kf, us, observations, hyperparameters, loss_range
                        )
                    value = -ll - log_prior
                    # A diverging filter does not always throw — it can return a NaN/Inf
                    # log-likelihood, which a gradient-based optimizer will happily follow into
                    # nonsense. Treat non-finite the same as thrown.
                    isfinite(value) ? value : _DIVERGED_PENALTY - log_prior
                catch
                    # A diverged filter. `log_prior` is deliberately computed outside this
                    # `try`, so the returned value is `Inf` while its partials stay finite and
                    # point back toward the prior mode — see `_DIVERGED_PENALTY`.
                    _DIVERGED_PENALTY - log_prior
                end
            end
            # The reduced `opt_maxiters` is only defensible because a warm start begins near the
            # optimum. Without one every re-optimization is as cold as the first, so it gets the
            # same (larger) budget — otherwise `warm_start = false` would silently compare a
            # converged fit against a truncated one and call the difference path dependence.
            options = (forecast_number == 1 || !warm_start) ? initial_options :
                ordinary_options
            optimization = OptimiseHyperparams(
                priors; method = hyper_method.method,
                adtype = hyper_method.adtype, options,
            )
            @info "UKF: hyperparameter optimization starting" forecast_number observations = T loss_observations =
                length(loss_range) window_start = first(loss_range) checkpoint_mode checkpoint_steps parameters =
                collect(learned_names) warm_started = warm_start && forecast_number > 1 maxiters =
                get(options, :maxiters, missing)
            result = optimize_hyperparams(
                neg_logposterior, initial_values, optimization
            )
            ukf.p = merge(base_hyperparams, result.θ)
            @info "UKF: hyperparameter optimization complete" forecast_number retcode =
                result.retcode log_posterior = result.ll estimates = result.θ
        end

        @info "UKF: filtering observation history" forecast_number observations = T
        # Rebuilt at the CURRENT hyperparameters so the initial state matches them; the caller's
        # `ukf` still carries `p` across origins (the UKF replays the full window each time, so
        # `p` is the only state that persists).
        kf_live = build_backend(Float64, ukf.p)
        solution = forward_trajectory(kf_live, us, observations, ukf.p)
        mean_updates = _fitted_observation_means(
            (observation_spec,), layout.accumulator_indices, extract_latent, solution.xt,
            ukf.p, filter_method.dt, update_range,
        )
        hyperparameter_summary = DataFrame(
            parameter = String[], statistic = String[], value = Float64[]
        )
        for name in learned_names
            push!(
                hyperparameter_summary,
                (string(name), "estimate", Float64(ukf.p[name])),
            )
        end
        if !emit_forecast
            @info "UKF: forecast suppressed after sequential fit" forecast_number
            return nothing, mean_updates, hyperparameter_summary
        end

        @info "UKF: generating forecast" forecast_number horizons = n_ahead
        means, covariances = forecast_states(
            kf_live,
            solution.xt[end],
            _symmetrize_covariance(solution.Rt[end]);
            n_ahead = Int(n_ahead),
            t0 = float(T - 1) * filter_method.dt,
            dt = filter_method.dt,
            p = ukf.p,
        )
        quantile_matrix = Matrix{Float64}(
            undef, Int(n_ahead), length(DEFAULT_QS)
        )
        forecast_origin = float(T - 1) * filter_method.dt
        for horizon in 1:Int(n_ahead)
            # The predictive count at horizon `h`, evaluated at the horizon's OWN model time
            # `t0 + h * dt` (the time `forecast_ensemble` samples at): with a time-varying
            # ascertainment the scale, the mean and the NB variance all move with `t`.
            moments = observation_gaussian_moments(
                observation_spec,
                means[horizon][accumulator],
                covariances[horizon][accumulator, accumulator],
                extract_latent(means[horizon]),
                ukf.p,
                forecast_origin + horizon * filter_method.dt,
            )
            mean_observation = moments.mean
            standard_deviation = sqrt(max(moments.var, 1.0e-12))
            for (column, probability) in enumerate(DEFAULT_QS)
                quantile_matrix[horizon, column] = max(
                    0.0,
                    quantile(Normal(mean_observation, standard_deviation), probability),
                )
            end
        end

        # Forecast-spread diagnostic: the unscented transform already propagated the full
        # covariance, so each latent coefficient's per-horizon predictive sd is free to read
        # off the diagonal.
        latent_log_sd = [
            sqrt(max(covariances[horizon][slot, slot], 0.0))
                for horizon in 1:Int(n_ahead), slot in layout.latent_range
        ]
        append_latent_spread!(hyperparameter_summary, layout.latent_names, latent_log_sd)
        # In-sample counterpart: how the latent actually behaved over the assimilated window,
        # which the forecast-spread diagnostic above cannot show.
        append_latent_audit!(
            hyperparameter_summary, layout.latent_names, solution.xt, layout.latent_range;
            dt = filter_method.dt,
        )
        return quantile_matrix, mean_updates, hyperparameter_summary
    end
    return ukf, fit_forecast!
end

"""
    build_inference(
        filter_method::PF, hyper_method::LiuWest,
        petri_vf!, layout, stochastic_dynamics, obs_model,
        base_hyperparams, x0_state;
        n_ahead, n_draws=2000, rng=Random.default_rng(),
        initial_learned_variance=(;), initial_latent_variance=(;),
        derived_hyperparameters=nothing,
    ) -> (pf, fit_forecast!)

Build the online particle-filter path. `fit_forecast!` assimilates only the
explicit `update_range`, mutates `pf` through the Liu-West
`correct!`/refresh/`predict!` sequence, and forecasts on a deep copy so forecast
sampling cannot perturb the live filter's RNG.

`initial_learned_variance` sets the initial (unconstrained-space) variance of named
learned slots — the real width of the Liu-West θ-cloud. Unnamed slots default to the
prior's own unconstrained variance (so the prior `sd` sets the spread). Override a
slot whose truth would otherwise sit in the tail (e.g. a logit-scale `escape_mean`).
`initial_latent_variance` does the same for the latent coefficient slots, which default
to `DEFAULT_LATENT_VARIANCE`.

`derived_hyperparameters`, when given, is a function `hyper -> NamedTuple` evaluated on each
forecast-initial particle's `merge(base_hyperparams, learned θ)`. Each returned field is summarised
with the same `q05`/`q50`/`q95`/`mean` rows as a learned parameter. Use it when the learned
coordinates are not themselves interpretable (e.g. zero-sum Helmert coordinates of weekday effects).

Each learned parameter also gets an `sd_ratio` row: the filtered θ-cloud's sd over the prior's, in
unconstrained space. It is the direct reading of cloud collapse, which the ESS rows only hint at.
"""
function build_inference(
        filter_method::PF,
        hyper_method::LiuWest,
        petri_vf!,
        layout,
        stochastic_dynamics,
        obs_model,
        base_hyperparams,
        x0_state;
        n_ahead::Integer,
        n_draws::Integer = 2000,
        rng = Random.default_rng(),
        initial_learned_variance::NamedTuple = (;),
        initial_latent_variance::NamedTuple = (;),
        build_x0 = _ -> x0_state,
        derived_hyperparameters = nothing,
    )
    n_ahead > 0 || throw(ArgumentError("n_ahead must be positive, got $n_ahead"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive, got $n_draws"))
    dynamics, measure, n_observations, n_measurement_noise =
        build_dynamics_and_measurement(
        petri_vf!, layout, obs_model, stochastic_dynamics;
        dt = filter_method.dt,
        supersample = filter_method.supersample,
        obs_jitter = filter_method.obs_jitter,
    )
    _check_single_signal(n_observations)
    pf, learned, updater! = _pf_backend(
        filter_method,
        hyper_method,
        dynamics,
        measure,
        n_observations,
        n_measurement_noise,
        layout,
        stochastic_dynamics,
        obs_model,
        base_hyperparams,
        x0_state,
        build_x0,
        rng,
        initial_learned_variance,
        initial_latent_variance,
    )

    function fit_forecast!(
            pf, asof, update_range, us, forecast_number;
            emit_forecast::Bool = true, sample_callback = nothing,
        )
        T = _validate_fit_inputs(asof, update_range, us)
        forecast_number > 0 || throw(
            ArgumentError("forecast_number must be positive, got $forecast_number")
        )
        observations = _observation_vectors(asof.counts)
        mean_updates = Float64[]
        ess_history = Float64[]
        # Weighted-mean latent path, for the in-sample audit. Stored as the latent SUB-vector
        # (length L, 1-based), not the full state, so it is indexed by `1:L` below.
        latent_path = Vector{Vector{Float64}}()
        filtered_particles = nothing
        filtered_weights = nothing
        n_updates = length(update_range)
        progress_interval = max(cld(n_updates, 10), 1)
        @info "PF: assimilating observations" forecast_number observations_to_assimilate =
            n_updates total_observations = T particles = filter_method.n_particles

        for (progress, k) in enumerate(update_range)
            time = (k - 1) * filter_method.dt
            correct!(pf, us[k], observations[k], base_hyperparams, time)
            posterior = particles(pf)
            weights = expweights(pf)
            weight_sum = sum(weights)
            push!(ess_history, _effective_sample_size(weights))
            push!(
                mean_updates,
                sum(
                    weights[i] * pf.measurement(
                        posterior[i], us[k], base_hyperparams, time, false
                    )[1] for i in eachindex(posterior)
                ) / weight_sum,
            )
            push!(
                latent_path,
                [
                    sum(weights[i] * posterior[i][slot] for i in eachindex(posterior)) /
                        weight_sum for slot in layout.latent_range
                ],
            )
            if emit_forecast && k == last(update_range)
                filtered_particles = deepcopy(posterior)
                filtered_weights = copy(weights)
            end
            updater!(state(pf).xprev, weights)
            predict!(pf, us[k], base_hyperparams, time)
            if progress == 1 || progress == n_updates || progress % progress_interval == 0
                @info "PF: assimilation progress" forecast_number completed =
                    progress total = n_updates observation_index = k
            end
        end

        if !emit_forecast
            @info "PF: forecast suppressed after sequential assimilation" forecast_number
            return nothing, mean_updates, DataFrame(
                    parameter = String[], statistic = String[], value = Float64[]
                )
        end

        @info "PF: generating forecast ensemble" forecast_number draws = n_draws horizons =
            n_ahead
        forecast_filter = deepcopy(pf)
        indices = _weighted_sample_indices(
            filtered_weights, Int(n_draws), forecast_filter.rng
        )
        initial_states = [filtered_particles[index] for index in indices]
        samples, latent_samples = forecast_ensemble(
            forecast_filter,
            initial_states,
            base_hyperparams;
            n_ahead = Int(n_ahead),
            t0 = float(T - 1) * filter_method.dt,
            dt = filter_method.dt,
            latent_range = layout.latent_range,
        )

        hyperparameter_summary = DataFrame(
            parameter = String[], statistic = String[], value = Float64[]
        )
        # The filtered cloud's width against the prior's, per learned parameter (unconstrained
        # space): the collapse signature the ESS rows only hint at. A Liu-West cloud that has
        # degenerated reads far below 1 here while its quantiles merely look precise. Weighted,
        # from the filtered cloud rather than the forecast draws.
        _, V_filtered = _weighted_theta_moments(
            _theta_block(filtered_particles, learned), filtered_weights
        )
        function summarise!(name, values)
            for (statistic, probability) in
                (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))
                push!(
                    hyperparameter_summary,
                    (string(name), statistic, quantile(values, probability)),
                )
            end
            push!(
                hyperparameter_summary,
                (string(name), "mean", sum(values) / length(values)),
            )
            return nothing
        end
        for (slot, name) in enumerate(learned.names)
            summarise!(name, [learned.extract(particle)[name] for particle in initial_states])
            push!(
                hyperparameter_summary,
                (
                    string(name), "sd_ratio",
                    sqrt(
                        max(V_filtered[slot, slot], 0.0) /
                            prior_unconstrained_variance(learned.priors[slot])
                    ),
                ),
            )
        end
        if derived_hyperparameters !== nothing && !isempty(initial_states)
            derived = [
                derived_hyperparameters(merge(base_hyperparams, learned.extract(particle)))
                    for particle in initial_states
            ]
            for name in keys(first(derived))
                summarise!(name, [Float64(d[name]) for d in derived])
            end
        end
        # Weight-degeneracy diagnostic. Liu-West jitter is proportional to the cloud's own
        # variance, so a run whose ESS collapses freezes its θ estimates wherever they happened
        # to be — and that is invisible in the estimates themselves, which merely look precise.
        # Reported as a FRACTION of the particle count, so it is comparable across runs.
        if !isempty(ess_history)
            ess_fraction = ess_history ./ filter_method.n_particles
            for (statistic, value) in (
                    ("ess_frac_min", minimum(ess_fraction)),
                    ("ess_frac_median", quantile(ess_fraction, 0.5)),
                    ("ess_frac_final", last(ess_fraction)),
                )
                push!(hyperparameter_summary, ("particle_filter", statistic, value))
            end
            minimum(ess_fraction) < 0.02 && @warn(
                "PF: effective sample size collapsed — Liu-West θ estimates are frozen, not " *
                    "converged. Raise `jitter_floor_fraction`, lower `discount`, or add particles.",
                forecast_number,
                min_ess_fraction = minimum(ess_fraction),
                n_particles = filter_method.n_particles,
            )
        end

        # Forecast-spread diagnostic: the realised across-particle spread of each latent
        # coefficient per horizon (unconstrained chart), the PF counterpart of the UKF's
        # propagated covariance diagonal.
        latent_log_sd = [
            _cloud_sd(view(latent_samples, horizon, :, l))
                for horizon in 1:Int(n_ahead), l in axes(latent_samples, 3)
        ]
        append_latent_spread!(hyperparameter_summary, layout.latent_names, latent_log_sd)
        # In-sample counterpart. `latent_path` holds the latent sub-vector, so the range is
        # `1:L` rather than `layout.latent_range`.
        append_latent_audit!(
            hyperparameter_summary, layout.latent_names, latent_path,
            1:length(layout.latent_names); dt = filter_method.dt,
        )
        quantile_matrix = forecast_quantiles(samples)
        sample_callback === nothing || sample_callback(samples)
        @info "PF: forecast complete" forecast_number learned_parameters =
            collect(learned.names)
        return quantile_matrix, mean_updates, hyperparameter_summary
    end
    return pf, fit_forecast!
end
