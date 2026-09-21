# ============================================================================
# Configuration schema — Configurations.jl @option types
# ============================================================================
#
# Typed, validated, TOML-native config parsing. The goal here is that all the
# CLI usage from dagster_defs.py is now based around generating a single TOML
# file, which is then parsed into a single `RunConfig` object.
#
# Reading is STRICT: an unknown or misspelled key raises a clear `InvalidKeyError`
# naming the offending field and listing the valid ones — at load time, not as a
# `KeyError` deep inside `build_model`. Defaults live here once (the single source
# of truth), so a field absent from a config file is filled, not fatal.
#
# This file holds only the submodel-AGNOSTIC schema (the "how"): the priors, the
# inference-method blocks, shared config primitives, and `RunConfig`. Each submodel's
# OWN epi config type lives in its `submodels/<name>.jl` (the "what"), co-located with
# `build_model`, so `src/` never needs editing to add a submodel.
#
# `RunConfig.epi` therefore carries the raw `[epi.<name>]` table as an opaque
# `Dict{String,Any}`; `submodel_name` reads the single alias key, and the selected
# submodel parses that table into its typed `@option` config via its `parse_epi_config`
# hook (in `build_model`, `validate_config.jl`, and the submodel tests). Strict epi-key
# validation still works there, but it is owned by the submodel, not by `src/`.

# **NB**: The config versions of inference methods (UKF, PF, Optimise, Liu–West)
# are *not* the same as the method objects (UKF, PF, OptimiseHyperparams, LiuWest)
# that implement the algorithms. The config types are just typed TOML containers;
# the method objects are constructed from them by `build_filter`/`build_hyper` and
# then wired together by `setup_inference` (see inference_setup.jl).


# ---------------------------------------------------------------------------
# Priors — a shared, open-ended `name => Gaussian spec` map (one table/param).
# ---------------------------------------------------------------------------

"""
Constant mapping from the `constraint` string in the TOML to the EnsembleKalmanFilter.jl (EKF)
    prior constructor function used in this project.
"""
const _PRIOR_CONSTRUCTORS = Dict(
    "positive" => positive_gaussian,
    "unit_interval" => unit_interval_gaussian,
    "unconstrained" => unconstrained_gaussian,
)

"""
    PriorSpec

This is what the TOML parser produces for each parameter's prior spec. It is validated and then
    converted to a `ParameterDistribution` by `build_prior`.
"""
@option struct PriorSpec
    mean::Float64
    std::Float64
    constraint::String = "positive"
end

"""
    build_prior(name, spec::PriorSpec) -> ParameterDistribution

This is the single place that maps a `PriorSpec` to a `ParameterDistribution` (the EKF prior type).
    The `name` is used for error messages only.
"""
function build_prior(name::Symbol, spec::PriorSpec)
    ctor = get(_PRIOR_CONSTRUCTORS, spec.constraint) do
        error(
            "unknown prior constraint '$(spec.constraint)' for $name " *
                "(positive|unit_interval|unconstrained)"
        )
    end
    return ctor(name, spec.mean, spec.std)
end

"""
    prior_upper(spec::PriorSpec; probability = 0.99) -> Float64

An upper quantile of a `PriorSpec` on its **constrained** scale.

`positive` specs use the moment-matched lognormal, `unit_interval` is capped at 1, and
`unconstrained` is Gaussian.
"""
function prior_upper(spec::PriorSpec; probability::Real = 0.99)
    0 < probability < 1 || throw(
        ArgumentError("probability must be in (0,1), got $probability")
    )
    z = quantile(Normal(), probability)
    if spec.constraint == "positive"
        spec.mean > 0 || return spec.mean + z * spec.std
        sigma = sqrt(log1p((spec.std / spec.mean)^2))
        mu = log(spec.mean) - sigma^2 / 2
        return exp(mu + z * sigma)
    elseif spec.constraint == "unit_interval"
        return min(spec.mean + z * spec.std, 1.0)
    end
    return spec.mean + z * spec.std
end

"""
    prior_R_eff_bound(priors; chi_max = 1.55, probability = 0.95) -> Float64

A conservative upper bound on `R_eff = R0_baseline * Rt * chi * S/N` for ODE-step sizing, using
`S/N <= 1` and marginal prior quantiles. Missing priors contribute 1. The generic `chi_max`
default is retained for direct numerical use; config-validation callers must pass
[`seasonal_forcing_upper_bound`](@ref) so a configurable cosine is not mistaken for the default
forcing.
"""
function prior_R_eff_bound(
        priors::AbstractDict{String, PriorSpec};
        chi_max::Real = 1.55, probability::Real = 0.95
    )
    upper(name, default) = haskey(priors, name) ?
        prior_upper(priors[name]; probability) : default
    return upper("R0_baseline", 1.0) * upper("Rt", 1.0) * chi_max
end

"""
    build_priors(specs) -> NamedTuple

Map a `name => PriorSpec` collection to a `name => ParameterDistribution`
NamedTuple (the flat prior map shared across submodels; each selects its subset).
"""
build_priors(specs) =
    (; (Symbol(k) => build_prior(Symbol(k), v) for (k, v) in specs)...)

"""
    load_prior_specs(path) -> Dict{String,PriorSpec}

Parse a flat priors TOML (one `[name]` table per parameter) into validated specs.
"""
load_prior_specs(path) = Dict(k => from_dict(PriorSpec, v) for (k, v) in TOML.parsefile(path))

"""
    load_priors(path) -> NamedTuple

Load a priors TOML into a `name => ParameterDistribution` NamedTuple.
"""
load_priors(path) = build_priors(load_prior_specs(path))

# ---------------------------------------------------------------------------
# Durations — shared mean-days schema (Erlang rates = n_stages / mean_duration).
# A reusable config primitive both submodels compose into their own epi config;
# lives here (not in a submodel file) because it is shared across many models.
# ---------------------------------------------------------------------------

"""
    Durations

Mean durations, in days, for the shared epidemiological and observation
processes. Submodels embed this option block and convert the durations to their
stage-specific transition rates.

# The reporting delay

Observation is attached to the INFECTION EVENT (`AtEvent(:transmission)`), so the whole
infection-to-report delay is the observation chain and nothing else:

    delay ~ Erlang(n_obs_stages - 1, obs_progression)
    mean  = (n_obs_stages - 1) * obs_progression
    var   = (n_obs_stages - 1) * obs_progression^2

Only `n_obs_stages - 1` stages are delay stages; the terminal one is the reset-accumulator
(`src/full_dynamics.jl`), which registers a count on arrival and so accrues no residence time.

This is what makes the delay INDEPENDENT of `latent` and `infectious`, which in turn frees both to
take literature values. Before, the chain hung off the `I` compartment and the delay carried
`latent + infectious` whether or not that made sense — nobody presents to an ED at recovery — so
`infectious` had to set the generation interval AND the reporting lag at once.

There is deliberately no `obs_inflow`. It used to set the tap's own rate, which is meaningless now
the tap IS the transmission transition: the observation fires at the infection rate, so exactly one
observation is recorded per infection with nothing to configure and nothing that can drift. A
config field with no effect is precisely the failure this removes.

`generation interval mean = latent + infectious`, unchanged and now independent of the above.

The defaults and their coupling are sourced in `docs/parameter-provenance.md` and checked by
`examples/parameter_model_check.jl`. In particular, `infectious` is a generation-interval
residence time rather than a clinical infectious period.
"""
@option struct Durations
    latent::Float64 = 2.0
    infectious::Float64 = 1.5
    immunity::Float64 = 180.0
    obs_progression::Float64 = 6.6
end

# Classical RK4 is stable on a real negative eigenvalue only while `|lambda * h| < 2.785`.
const RK4_STABILITY_LIMIT = 2.785

# A proximity warning, not an accuracy target: 2.0 leaves about 1.4x margin to divergence.
# Against a 32x-finer reference, relative error at this threshold was 2.5e-6.
const RK4_WARN_LAMBDA_H = 2.0

"""
    max_transition_rate(dur::Durations, n_E, n_I; R_eff_max = 1.0) -> Float64

The fastest model rate in 1/day, including Erlang progression, observation, waning and the
transmission block. Linearising `(E, I)` gives

    lambda^2 + (sigma + gamma) lambda + sigma*gamma*(1 - R_eff) = 0

so the magnitude of its faster eigenvalue is

    |lambda_-| = (sigma + gamma)/2 + sqrt(((sigma + gamma)/2)^2 + sigma*gamma*(R_eff - 1))

with `sigma = n_E / latent` and `gamma = n_I / infectious`. `R_eff_max = 0` recovers the pure
linear-rate bound; use a prior-tail bound for prior-predictive work.
"""
function max_transition_rate(
        dur::Durations, n_E::Integer, n_I::Integer; R_eff_max::Real = 1.0
    )
    R_eff_max >= 0 || throw(
        ArgumentError("R_eff_max must be non-negative, got $R_eff_max")
    )
    sigma = n_E / dur.latent
    gamma = n_I / dur.infectious
    half_sum = (sigma + gamma) / 2
    # No clamp on `R_eff_max - 1`: the discriminant is `((sigma-gamma)/2)^2 + sigma*gamma*R_eff`,
    # non-negative for every `R_eff >= 0`, and the sub-critical case is meaningful — at `R_eff = 0`
    # this collapses to `max(sigma, gamma)` as it must.
    transmission = half_sum + sqrt(half_sum^2 + sigma * gamma * (R_eff_max - 1))
    return maximum(
        (
            sigma, gamma, 1 / dur.obs_progression, 1 / dur.immunity, transmission,
        )
    )
end

"""
    required_supersample(dur::Durations, n_E, n_I, weekly_dt; R_eff_max = 1.0,
                         target = RK4_WARN_LAMBDA_H) -> Int

The smallest `supersample` whose substep keeps `|lambda * h| <= target`. The default leaves a
proximity margin; pass `RK4_STABILITY_LIMIT` for the bare stability bound.
"""
function required_supersample(
        dur::Durations, n_E::Integer, n_I::Integer, weekly_dt::Real;
        R_eff_max::Real = 1.0, target::Real = RK4_WARN_LAMBDA_H
    )
    target > 0 || throw(ArgumentError("target must be positive, got $target"))
    rate = max_transition_rate(dur, n_E, n_I; R_eff_max)
    return max(1, ceil(Int, rate * weekly_dt / target))
end

"""
    assert_integration_stable(dur::Durations, n_E, n_I, weekly_dt, supersample;
                              R_eff_max = 1.0) -> Float64

Validate a fixed RK4 substep against these durations and `R_eff_max`. Errors beyond
[`RK4_STABILITY_LIMIT`](@ref) and warns beyond [`RK4_WARN_LAMBDA_H`](@ref).
"""
function assert_integration_stable(
        dur::Durations, n_E::Integer, n_I::Integer, weekly_dt::Real, supersample::Integer;
        R_eff_max::Real = 1.0
    )
    supersample >= 1 || throw(ArgumentError("supersample must be >= 1, got $supersample"))
    h = weekly_dt / supersample
    rate = max_transition_rate(dur, n_E, n_I; R_eff_max)
    lambda_h = rate * h
    if lambda_h >= RK4_STABILITY_LIMIT
        needed = max(
            supersample + 1,
            required_supersample(
                dur, n_E, n_I, weekly_dt; R_eff_max, target = RK4_STABILITY_LIMIT
            ),
        )
        throw(
            ArgumentError(
                "ODE substep is unstable: the fastest rate the substep must resolve is " *
                    "$(round(rate, sigdigits = 3))/day (at R_eff <= " *
                    "$(round(R_eff_max, sigdigits = 3))) and the substep is " *
                    "$(round(h, sigdigits = 3)) days, giving |lambda*h| = " *
                    "$(round(lambda_h, sigdigits = 3)) against RK4's stability limit of " *
                    "$(RK4_STABILITY_LIMIT). Raise `supersample` to at least $(needed) " *
                    "(currently $(supersample)), or lengthen the shortest duration in " *
                    "`durations_days`. Left alone this diverges to NaN and surfaces as a " *
                    "Cholesky failure inside the filter, which names neither knob."
            )
        )
    elseif lambda_h > RK4_WARN_LAMBDA_H
        @warn(
            "ODE substep is within 1.4x of RK4's stability limit. Accuracy here is still fine " *
                "(measured ~1e-5), but `R_eff_max` is a prior quantile and a run that exceeds it " *
                "has little margin left. Raise `supersample` for headroom.",
            lambda_h,
            warn_threshold = RK4_WARN_LAMBDA_H,
            stability_limit = RK4_STABILITY_LIMIT,
            supersample,
            suggested_supersample = required_supersample(
                dur, n_E, n_I, weekly_dt; R_eff_max
            ),
            R_eff_max,
        )
    end
    return lambda_h
end

"""
    initial_infection_state(y0, dur::Durations, ascertainment, accumulation_window_days)
        -> (; daily_incidence, exposed, infectious, obs_stage)

Invert one partially ascertained count into quasi-steady compartment occupancies. Because the
accumulator integrates the infection event directly,

    y0 = ascertainment * accumulation_window_days * daily_incidence

and each upstream compartment holds `daily_incidence * mean_duration`. The required
`accumulation_window_days` is the run's observation cadence, not its reporting delay.
"""
function initial_infection_state(y0, dur::Durations, ascertainment, accumulation_window_days)
    # Both arguments are DENOMINATORS. Unchecked, `ascertainment = 0` or a missing window yields
    # `Inf`/`NaN` compartment occupancies that propagate into `x0` and only surface much later as
    # a filter divergence, with nothing pointing back to here.
    (isfinite(ascertainment) && ascertainment > 0) || throw(
        ArgumentError(
            "ascertainment must be finite and > 0 to invert an observed count; got $ascertainment"
        )
    )
    (isfinite(accumulation_window_days) && accumulation_window_days > 0) || throw(
        ArgumentError(
            "accumulation_window_days (the run's `weekly_dt`) must be finite and > 0; " *
                "got $accumulation_window_days"
        )
    )
    daily_incidence = y0 / (ascertainment * accumulation_window_days)
    return (
        daily_incidence = daily_incidence,
        exposed = daily_incidence * dur.latent,
        infectious = daily_incidence * dur.infectious,
        obs_stage = daily_incidence * dur.obs_progression,
    )
end

"""
    SeasonalityConfig

How transmission seasonality is forced. Submodels embed this block (TOML:
`[epi.<submodel>.seasonality]`) and hand it to `build_seasonal_forcing`; see
`seasonality.jl` for the forcings themselves.

- `mode` — `"indoor_activity"` (default; empirical location-specific curve), `"cosine"`
  (the learned annual harmonic retained for explicit legacy configurations) or `"none"`
  (flat control).
- `kappa` — strength of the empirical forcing, `f = 1 + κ(σ − 1)`. Ignored by the other modes.

  Equivalently `(1 − κ) + κσ`: `κ` is the seasonally forced share and is bounded to `[0, 1]`, so
  the result is a positive convex combination. `κ = 1` is the source paper's one-to-one forcing —
  and, read as a share, the claim that **100% of transmission is seasonally forced**. The shipped
  0.8 leaves a fifth aseasonal, which is the natural reading of household transmission: it is a
  large, roughly year-round component of β that no indoor-activity metric should be scaling.
  The national curve is roughly a ±30% seasonal swing, 2–3× the `cosine` mode's default amplitude;
  location-specific extrema are wider. Switching modes therefore changes amplitude as well as
  shape. Scale `kappa` down for a shape-only comparison.
- `fallback` — what to do for a location the climatology does not cover (only the territories):
  `"us"` (national curve), `"none"` (flat) or `"error"`.
"""
@option struct SeasonalityConfig
    mode::String = "indoor_activity"
    kappa::Float64 = 0.8
    fallback::String = "us"
end

# --- Weekday observation effect (daily cadence; see day_of_week.jl) ---------------------------
#
# Alias-discriminated like the inference axes: exactly one of
# `[epi.<submodel>.day_of_week.none]`, `[… .plugin]` or `[… .learned]`. The model dispatches on
# the type, so there is no mode string to branch on and no field that means nothing in some mode.

"""No weekday effect: weights 1 and no extra observation variance (the step-1 daily model)."""
@option "none" struct NoDayOfWeekConfig end

"""
Weekday multipliers AND per-weekday extra observation variance estimated from history, then held
fixed. `fit_policy = "per_origin"` re-estimates from each origin's as-of history;
`"first_vintage"` estimates once from the first available report vintage and reuses that result.
`window_days` is the trailing history used; `exclude_recent_days` is the newest days ignored,
because the nowcast-inflated tail has a weekday pattern of its own (the reporting delay depends on
the report weekday).
"""
@option "plugin" struct PluginDayOfWeekConfig
    window_days::Int = 182
    exclude_recent_days::Int = 14
    fit_policy::String = "per_origin"
end

"""
Weekday multipliers learned by Liu–West as six zero-sum Helmert coordinates. It adds no extra
observation variance: the ensemble's spread over the learned weights is the uncertainty, and it
reads no history.
"""
@option "learned" struct LearnedDayOfWeekConfig end

const DayOfWeekConfig = Union{NoDayOfWeekConfig, PluginDayOfWeekConfig, LearnedDayOfWeekConfig}

# ---------------------------------------------------------------------------
# Run-level config (inference choices) — the single-TOML boundary target
#
# Inference is TWO orthogonal, alias-discriminated axes; the engine is built by multiple
# dispatch over both (see inference_setup.jl):
#   - `filter` — the state filter for the dynamic latent state (UKF / PF / …).
#   - `hyper`  — how the fixed-in-time hyperparameters are inferred: replay-based
#     marginal-loglik (optimise / VI / MCMC) or online carried-in-the-cloud (Liu–West).
# Valid combinations are exactly those with a `setup_inference` method; an invalid pairing
# (e.g. UKF + Liu–West, which has no particle cloud to carry hyperparameters) has no method.
# ---------------------------------------------------------------------------

# --- Axis: state filter ---

"""
    UKFFilterConfig

Unscented Kalman filter for the latent state. No tunables yet (the UKF's `dt`/`supersample`
come from the top-level run config); a marker for dispatch that can gain fields (e.g. a
sigma-point strategy) later.
"""
@option "ukf" struct UKFFilterConfig end

"""
    PFFilterConfig

Bootstrap particle filter for the latent state; `n_particles` is the cloud size.
`threads` (default `true`) parallelizes the per-particle propagation and weighting across
Julia's threads — size the pool with `julia --threads=N`. Propagation noise is drawn from a
per-thread RNG pool seeded from the run seed, so a seeded run reproduces exactly for a fixed
thread count (a different `--threads` is a different draw sequence).
"""
@option "pf" struct PFFilterConfig
    n_particles::Int
    threads::Bool = true
end

"""
    EnKFFilterConfig

Ensemble Kalman filter for the latent state (`ConfigurableEpi.AugmentedEnsembleKalmanFilter`),
sized by `n_ensemble`. `inflation` multiplies the ensemble spread after each propagation and must
be at least 1 — below that it shrinks the spread every step, which is filter divergence by
construction. `threads` parallelizes the per-member propagation; the noise is pre-drawn for the
whole ensemble, so a seeded run reproduces either way.
"""
@option "enkf" struct EnKFFilterConfig
    n_ensemble::Int
    inflation::Float64 = 1.0
    threads::Bool = false
end

# --- Axis: hyperparameter/static parameter inference ---

"""
    OptimiseConfig

Replay-based hyperparameter inference: maximize the filter's marginal likelihood over data
replays (NelderMead), re-optimizing every `reopt_interval` forecast origins. The iteration
limits distinguish the initial burn-in fit from later re-optimizations. `window_length` limits
the objective to the most recent observations while retaining the filter state at the beginning
of that window; `nothing` preserves full-history scoring.

`warm_start` (default `true`) starts each re-optimization from the PREVIOUS optimum rather than
from the configured starting values, making θ a continuation path across origins — which is the
right model when θ drifts slowly, and is what makes the reduced `opt_maxiters` sufficient after
the burn-in. Set it to `false` for an INDEPENDENT estimate at each origin: more expensive (every
re-optimization then gets the `opt_maxiters_burnin` budget, since it is starting cold), but free
of path dependence. That is the ablation that distinguishes "θ is genuinely stable across
origins" from "the warm start never travelled far enough to find out".
"""
@option "optimise" struct OptimiseConfig
    reopt_interval::Int
    opt_maxiters::Int
    opt_maxiters_burnin::Int
    window_length::Union{Nothing, Int} = nothing
    warm_start::Bool = true
end

"""
    LiuWestConfig

Online hyperparameter inference: hyperparameters are carried in the particle cloud and
evolve by the Liu–West shrink-jitter random walk (`discount`). `replay_on_revision` rebuilds
the filter when an already-assimilated observation is revised (rather than ignoring it).

`forgetting_memory_days` (`[hyper.liu_west.forgetting_memory_days]`, learned-parameter name =>
days) turns on Kulhavý forgetting toward the prior for the named parameters: see `LiuWest`. It is
merged OVER any defaults the submodel declares (`learned_forgetting_memory_days` on its bundle),
so an entry here overrides the model's own, and a very large value switches one off. Naming a
parameter that is not in the learned block is an error.
"""
@option "liu_west" struct LiuWestConfig
    discount::Float64
    replay_on_revision::Bool
    jitter_floor_fraction::Float64 = DEFAULT_JITTER_FLOOR_FRACTION
    forgetting_memory_days::Dict{String, Float64} = Dict{String, Float64}()
end

"""
    EKPConfig

Outer ensemble-Kalman-inversion calibration of the static parameters: an outer ensemble of
`n_ensemble` candidate parameter vectors, each scored by a complete inner filter replay, updated
by `EnsembleKalmanProcesses.TransformInversion`. Recalibrates every `reopt_interval` forecast
origins, warm-starting from the previous final ensemble; `burnin_iterations` at the first origin
and `iterations` at later recalibrations. `window_length` limits each candidate loss to the most
recent observations, starting from a rolling filter checkpoint; `nothing` preserves full-history
scoring.

`warm_start` (default `true`) is what makes that possible: each recalibration resumes from the
previous final ENSEMBLE — not just its mean, so the spread is carried too — which is why
`iterations` can be a small fraction of `burnin_iterations`. Set it to `false` to draw a fresh
ensemble from the prior at every recalibration, in which case each one gets the
`burnin_iterations` budget because it is starting cold. Use that as an ablation: a warm-started
θ-path can look stable simply because a 5-iteration update cannot move far from where it began,
and a cold start is the only way to tell that apart from genuine stability.

`inflation` is `TransformInversion`'s `default_multiplicative_inflation`. It defaults to 0, but the
prior-imposing update that this path uses is the case where EKP's own docstring notes inflation
"is often required" — reach for it first if the outer ensemble collapses over a long burn-in.

`threads` parallelizes the independent outer-candidate filter replays. Do not also enable
`filter.enkf.threads`: nested ensemble parallelism oversubscribes the process and is rejected.
"""
@option "ekp" struct EKPConfig
    n_ensemble::Int
    reopt_interval::Int
    iterations::Int
    burnin_iterations::Int
    inflation::Float64 = 0.0
    window_length::Union{Nothing, Int} = nothing
    warm_start::Bool = true
    threads::Bool = false
end

# --- Axis: observation cadence and scale -------------------------------------

"""Weekly NSSP percentages converted to counts using an annual ED-visit rate."""
@option "weekly_nssp_percent" struct WeeklyNSSPPercentInputConfig
    ed_rate_num::Float64 = 47.0
end

"""Daily NSSP incident ED-visit counts, already on the model's count scale."""
@option "daily_nssp_count" struct DailyNSSPCountInputConfig end

const InputConfig = Union{WeeklyNSSPPercentInputConfig, DailyNSSPCountInputConfig}

"""
    RunIO

Per-run orchestration I/O: the one reporting-triangle `data` path plus the labels Python owns
(`model_id`, `forecast_df`, `loc`). Replaces the old positional CLI contract — everything the run
reads/writes now rides in the `[io]` block of the run.toml. No defaults (Python always supplies
them). Defined locally (not shared via EpiMech) because ConfigurableEpi does not depend on EpiMech.
"""
@option struct RunIO
    data::String
    model_id::String
    forecast_df::String
    loc::String
    # Ordered locations for a JOINT multi-location model. `loc` stays the run/artifact label (and
    # the singleton fallback), so every existing run.toml keeps validating under Configurations.jl's
    # strict load. The order here is the single source of truth downstream: population vector,
    # radiation rows and columns, Petri strata, seasonal curves, latent processes, observation
    # vectors and forecast rows.
    locations::Vector{String} = String[]
end

"""
    RunConfig

Top-level, TOML-backed configuration for one ConfigurableEpi backtest: the per-run `[io]` block,
the required experiment knobs, the two inference axes (`filter` × `hyper`), the selected
`[epi.<submodel>]` block, and the named parameter priors.
"""
@option struct RunConfig
    io::RunIO
    n_ahead::Int
    n_draws::Int          # forecast-sample count (orthogonal to how you fit)
    supersample::Int
    seed::Int
    # New cadence-neutral names. The legacy weekly names remain readable for saved
    # configurations, but a run must use exactly one member of each pair.
    step_days::Union{Nothing, Float64} = nothing
    burnin_observations::Union{Nothing, Int} = nothing
    drop_recent_observations::Union{Nothing, Int} = nothing
    weekly_dt::Union{Nothing, Float64} = nothing
    burnin_weeks::Union{Nothing, Int} = nothing
    drop_recent_weeks::Union{Nothing, Int} = nothing
    # New input-discriminated scale contract. `ed_rate_num` is retained only as
    # the legacy spelling for an implicit weekly_nssp_percent input.
    input::Union{Nothing, InputConfig} = nothing
    ed_rate_num::Union{Nothing, Float64} = nothing
    # Sequential preserves the established online/replay behavior. The other two are implemented
    # only for basic_seir_daily_nssp: `independent` reconstructs and refits each report vintage from
    # scratch; `forked` (PF + Liu-West) keeps one filter on the settled observations and forks it
    # over each vintage's provisional nowcast tail.
    origin_mode::String = "sequential"
    # Inclusive forecast-origin cutoff. Earlier qualifying origins still update the filter and
    # hyperparameters; they simply do not spend work on or emit forecast simulations.
    forecast_start::Union{Nothing, String} = nothing
    learn_params::Vector{String} = String[]
    # The two discriminated inference axes. TOML: `[filter.<name>]` + `[hyper.<name>]`.
    filter::Union{UKFFilterConfig, PFFilterConfig, EnKFFilterConfig}
    hyper::Union{OptimiseConfig, LiuWestConfig, EKPConfig}
    epi::Dict{String, Any} = Dict{String, Any}()
    priors::Dict{String, PriorSpec} = Dict{String, PriorSpec}()
end

function _resolve_legacy_pair(new_value, old_value, new_name, old_name)
    if new_value !== nothing && old_value !== nothing
        throw(ArgumentError("configure `$new_name` or legacy `$old_name`, not both"))
    end
    value = new_value === nothing ? old_value : new_value
    value === nothing && throw(
        ArgumentError("missing `$new_name` (or legacy `$old_name`)")
    )
    return value
end

resolved_step_days(cfg::RunConfig) = Float64(
    _resolve_legacy_pair(cfg.step_days, cfg.weekly_dt, "step_days", "weekly_dt")
)
resolved_burnin_observations(cfg::RunConfig) = Int(
    _resolve_legacy_pair(
        cfg.burnin_observations, cfg.burnin_weeks,
        "burnin_observations", "burnin_weeks",
    )
)
resolved_drop_recent_observations(cfg::RunConfig) = Int(
    _resolve_legacy_pair(
        cfg.drop_recent_observations, cfg.drop_recent_weeks,
        "drop_recent_observations", "drop_recent_weeks",
    )
)

"""Resolve the explicit input block or the legacy top-level weekly scale."""
function resolved_input(cfg::RunConfig)
    if cfg.input === nothing
        cfg.ed_rate_num === nothing && throw(
            ArgumentError(
                "missing [input.daily_nssp_count] or [input.weekly_nssp_percent]; " *
                    "legacy configs may instead supply top-level `ed_rate_num`",
            )
        )
        return WeeklyNSSPPercentInputConfig(ed_rate_num = cfg.ed_rate_num)
    end
    cfg.ed_rate_num === nothing || throw(
        ArgumentError(
            "top-level legacy `ed_rate_num` conflicts with the explicit [input] block",
        )
    )
    return cfg.input
end

function validate_run_semantics(cfg::RunConfig)
    step_days = resolved_step_days(cfg)
    burnin = resolved_burnin_observations(cfg)
    drop_recent = resolved_drop_recent_observations(cfg)
    input = resolved_input(cfg)
    isfinite(step_days) && step_days > 0 || throw(
        ArgumentError("step_days must be finite and > 0; got $step_days")
    )
    isinteger(step_days) || throw(
        ArgumentError("step_days must be an integer number of calendar days; got $step_days")
    )
    burnin >= 1 || throw(ArgumentError("burnin_observations must be >= 1; got $burnin"))
    drop_recent >= 0 || throw(
        ArgumentError("drop_recent_observations must be non-negative; got $drop_recent")
    )
    cfg.origin_mode in ("sequential", "independent", "forked") || throw(
        ArgumentError(
            "origin_mode must be `sequential`, `independent` or `forked`; got $(repr(cfg.origin_mode))"
        )
    )
    selected_model = submodel_name(cfg)
    if cfg.origin_mode != "sequential" && !(selected_model in DAILY_NSSP_SUBMODELS)
        throw(
            ArgumentError(
                "origin_mode = `$(cfg.origin_mode)` is only supported for $(DAILY_NSSP_SUBMODELS); " *
                    "submodel $selected_model requires `sequential`",
            )
        )
    end
    if cfg.origin_mode == "forked" &&
            !(cfg.filter isa PFFilterConfig && cfg.hyper isa LiuWestConfig)
        throw(
            ArgumentError(
                "origin_mode = `forked` carries a particle cloud between origins and requires " *
                    "[filter.pf] + [hyper.liu_west]",
            )
        )
    end
    if input isa WeeklyNSSPPercentInputConfig
        isfinite(input.ed_rate_num) && input.ed_rate_num > 0 || throw(
            ArgumentError(
                "input.weekly_nssp_percent.ed_rate_num must be finite and > 0; " *
                    "got $(input.ed_rate_num)",
            )
        )
    end
    return cfg
end

# Names for logging, and the state-filter symbol the submodel's `build_model` dispatches
# on (its `supports` set and default learned-parameter set are keyed on the filter).
filter_name(::UKFFilterConfig) = "ukf"
filter_name(::PFFilterConfig) = "pf"
filter_name(::EnKFFilterConfig) = "enkf"
hyper_name(::OptimiseConfig) = "optimise"
hyper_name(::LiuWestConfig) = "liu_west"
hyper_name(::EKPConfig) = "ekp"
filter_symbol(::UKFFilterConfig) = :ukf
filter_symbol(::PFFilterConfig) = :pf
filter_symbol(::EnKFFilterConfig) = :enkf

"""
    resolve_priors(cfg::RunConfig, default_priors) -> NamedTuple

Resolve the effective prior set: the submodel's `default_priors()` (its model-science, as code)
with any run.toml `[priors]` entries merged **over** it, per key. A run config overriding one
prior therefore does not have to restate the rest — which is what makes a one-prior sweep cheap.

This is the single definition of the override rule; `run_model.jl` and `validate_config.jl` both
call it, so the entrypoint and the drift guard cannot disagree about it.
"""
resolve_priors(cfg::RunConfig, default_priors) =
    build_priors(resolve_prior_specs(cfg, default_priors))

"""
    RETIRED_PRIOR_NAMES

Prior names that no longer exist, mapped to what replaced them and how to convert.

`RunConfig.priors` is a `Dict`, so unknown keys merge cleanly and are then **silently ignored** —
a `run.toml` still carrying `[priors.Rt_rho]` would validate, report a prior count, and have no
effect whatsoever. That is the worst failure mode available: a config that looks applied and is
not. Hence a hard reject with the arithmetic printed, rather than an alias.
"""
const RETIRED_PRIOR_NAMES = Dict(
    "Rt_rho" => "Rt_tau — an OU correlation time in DAYS: tau = -weekly_dt / log(rho)",
    "Rt_sigma" => "Rt_sigma_stat — the STATIONARY sd: sigma_stat = sigma / sqrt(1 - rho^2)",
    "mu_rho" => "mu_log_tau — log(-weekly_dt / log(logistic(mu_rho)))",
    "sigma_rho" => "Rt_sigma_stat — see above",
    "tau_rho" => "sd_log_tau — approximately tau_rho * (1 - rho) / (-log rho) at the central rho",
    "z_rho" => "z_tau — unchanged, N(0, 1); only the name moved",
)

"""
    assert_no_retired_priors(specs, weekly_dt)

Reject a `[priors]` table naming a parameter that no longer exists, printing the replacement and
the conversion evaluated at this run's own `weekly_dt`.

Called from [`resolve_prior_specs`](@ref) so `run_model.jl` and `validate_config.jl` cannot
disagree about it — the same single-definition argument the prior-merge rule already makes.
"""
function assert_no_retired_priors(specs, weekly_dt::Real)
    retired = sort([k for k in keys(specs) if haskey(RETIRED_PRIOR_NAMES, k)])
    isempty(retired) && return specs
    lines = join(("  `$name` -> $(RETIRED_PRIOR_NAMES[name])" for name in retired), "\n")
    example = "e.g. at weekly_dt = $(weekly_dt), rho = 0.2 becomes tau = " *
        "$(round(-weekly_dt / log(0.2), sigdigits = 6)) days"
    throw(
        ArgumentError(
            "the [priors] table names $(length(retired)) parameter(s) that no longer exist:\n" *
                lines * "\n\nThe latent Rt process is now parameterised as an " *
                "Ornstein-Uhlenbeck process by CORRELATION TIME (days) and STATIONARY sd, not by " *
                "a per-step persistence and innovation sd. $(example). These are rejected rather " *
                "than aliased because `RunConfig.priors` is a Dict: an unknown key merges cleanly " *
                "and is then silently ignored, so a stale entry would validate and do nothing."
        )
    )
end

"""
    resolve_prior_specs(cfg::RunConfig, default_priors) -> Dict{String, PriorSpec}

The effective prior SPECS, before they are built into `ParameterDistribution`s — the same per-key
merge [`resolve_priors`](@ref) performs, stopping one step earlier.

Exposed because a submodel with per-location parameters cannot use the built distributions: an
`AR1ParamSpec` named `Rt_ca` requires an init prior whose own name is `Rt_ca`
(`_validate_process_prior_name`), and the location set is only known at run time, so
`default_priors()` cannot enumerate them. Handing the submodel the spec lets it mint a
correctly-named prior per location from one template entry, while the override rule stays defined
in exactly one place.
"""
function resolve_prior_specs(cfg::RunConfig, default_priors)
    assert_no_retired_priors(cfg.priors, resolved_step_days(cfg))
    return merge(default_priors(), cfg.priors)
end

"""
    function submodel_name(cfg::RunConfig)
Return the single submodel name from the `[epi.<name>]` table in the run config.
Raises an error if the table is empty (no submodel selected).
"""
function submodel_name(cfg::RunConfig)
    isempty(cfg.epi) &&
        error("run config has no [epi.<submodel>] block (cannot determine the submodel)")
    return only(keys(cfg.epi))
end
