# TOML-backed run configuration: Configurations.jl `@option` types, read strictly with `from_toml`.

# --- priors -----------------------------------------------------------------------------------

const _PRIOR_CONSTRUCTORS = Dict(
    "positive" => positive_gaussian,
    "unit_interval" => unit_interval_gaussian,
    "unconstrained" => unconstrained_gaussian,
)

"""
    PriorSpec(; mean, std, constraint = "positive")

One parameter's prior as written in TOML; `constraint` is `positive`, `unit_interval` or
`unconstrained`. [`build_prior`](@ref) turns it into an EKP `ParameterDistribution`.
"""
@option struct PriorSpec
    mean::Float64
    std::Float64
    constraint::String = "positive"
end

function build_prior(name, spec::PriorSpec)
    ctor = get(_PRIOR_CONSTRUCTORS, spec.constraint) do
        throw(
            ArgumentError(
                "unknown prior constraint $(repr(spec.constraint)) for $name; use one of " *
                    "$(sort!(collect(keys(_PRIOR_CONSTRUCTORS))))"
            )
        )
    end
    return ctor(name, spec.mean, spec.std)
end

"""
    build_priors(specs) -> NamedTuple

`name => PriorSpec` pairs to a `name => ParameterDistribution` NamedTuple.
"""
build_priors(specs) = (; (Symbol(k) => build_prior(Symbol(k), v) for (k, v) in specs)...)

"""
    load_prior_specs(path) -> Dict{String, PriorSpec}
    load_priors(path) -> NamedTuple

Read a flat priors TOML (one `[name]` table per parameter).
"""
load_prior_specs(path) = Dict(k => from_dict(PriorSpec, v) for (k, v) in TOML.parsefile(path))
load_priors(path) = build_priors(load_prior_specs(path))

"""
    prior_upper(spec::PriorSpec; probability = 0.99) -> Float64

Upper quantile of a prior on its constrained scale (moment-matched lognormal for `positive`).
"""
function prior_upper(spec::PriorSpec; probability::Real = 0.99)
    0 < probability < 1 || throw(ArgumentError("probability must be in (0,1), got $probability"))
    z = quantile(Normal(), probability)
    if spec.constraint == "positive" && spec.mean > 0
        sigma = sqrt(log1p((spec.std / spec.mean)^2))
        return exp(log(spec.mean) - sigma^2 / 2 + z * sigma)
    end
    upper = spec.mean + z * spec.std
    return spec.constraint == "unit_interval" ? min(upper, 1.0) : upper
end

"""
    prior_R_eff_bound(priors; chi_max = 1.55, probability = 0.95) -> Float64

Conservative bound on `R_eff = R0_baseline * Rt * chi * S/N` from marginal prior quantiles
(`S/N <= 1`; a missing prior contributes 1). Pass `chi_max = seasonal_forcing_upper_bound(...)`.
"""
function prior_R_eff_bound(
        priors::AbstractDict{String, PriorSpec}; chi_max::Real = 1.55, probability::Real = 0.95,
    )
    upper(name) = haskey(priors, name) ? prior_upper(priors[name]; probability) : 1.0
    return upper("R0_baseline") * upper("Rt") * chi_max
end

# --- shared model config primitives -----------------------------------------------------------

"""
    Durations(; latent = 2.0, infectious = 1.5, immunity = 180.0, obs_progression = 6.6)

Mean durations in days. Erlang stage rates are `n_stages / mean`. Because observation is attached
to the infection event, the reporting delay is the observation chain alone:
`(n_obs_stages - 1) * obs_progression` (the terminal stage is the reset accumulator).
"""
@option struct Durations
    latent::Float64 = 2.0
    infectious::Float64 = 1.5
    immunity::Float64 = 180.0
    obs_progression::Float64 = 6.6
end

"""
    SeasonalityConfig(; mode = "indoor_activity", kappa = 0.8, fallback = "us")

Transmission seasonality: `mode` is `indoor_activity` (empirical location curve scaled by
`kappa ∈ [0, 1]`, the seasonally forced share of transmission), `cosine` (learned annual
harmonic) or `none`. `fallback` (`us`, `none`, `error`) covers a location the climatology lacks.
"""
@option struct SeasonalityConfig
    mode::String = "indoor_activity"
    kappa::Float64 = 0.8
    fallback::String = "us"
end

"""
    NoDayOfWeekConfig()                                   # [day_of_week.none]
    PluginDayOfWeekConfig(; window_days = 182, exclude_recent_days = 14, fit_policy = "per_origin")
    LearnedDayOfWeekConfig()                              # [day_of_week.learned]

Weekday observation effect for daily counts: none, multipliers and per-weekday extra dispersion
estimated from history (`fit_policy` is `per_origin` or `first_vintage`), or six zero-sum Helmert
coordinates learned by Liu-West.
"""
@option "none" struct NoDayOfWeekConfig end

@option "plugin" struct PluginDayOfWeekConfig
    window_days::Int = 182
    exclude_recent_days::Int = 14
    fit_policy::String = "per_origin"
end

@option "learned" struct LearnedDayOfWeekConfig end

const DayOfWeekConfig = Union{NoDayOfWeekConfig, PluginDayOfWeekConfig, LearnedDayOfWeekConfig}

"""
    CountInput()                                          # [input.counts]
    PercentInput(; annual_rate_per_100)                   # [input.percent]

The observation scale. `counts` are already on the model's count scale. `percent` observations
are percentages of a denominator series (all visits, say) whose annual volume is
`annual_rate_per_100` per 100 population; the runner converts them to counts with the location's
population and the observation interval.
"""
@option "counts" struct CountInput end

@option "percent" struct PercentInput
    annual_rate_per_100::Float64
end

const InputConfig = Union{CountInput, PercentInput}

# --- inference axis 1: the state filter -------------------------------------------------------

"""
    UKF(; obs_jitter = 1.0)                               # [filter.ukf]

Unscented Kalman filter. `obs_jitter` scales the accumulator process-noise whisker that keeps the
smoother covariance full-rank.
"""
@option "ukf" struct UKF
    obs_jitter::Float64 = 1.0
end

"""
    PF(; n_particles, threads = true)                    # [filter.pf]

Bootstrap particle filter. `threads` parallelises particle propagation; a seeded run reproduces
exactly for a fixed thread count.
"""
@option "pf" struct PF
    n_particles::Int
    threads::Bool = true
end

"""
    EnKF(; n_ensemble, inflation = 1.0, threads = false)  # [filter.enkf]

Augmented ensemble Kalman filter. `inflation >= 1` multiplies the ensemble spread after each
propagation; `threads` parallelises member propagation.
"""
@option "enkf" struct EnKF
    n_ensemble::Int
    inflation::Float64 = 1.0
    threads::Bool = false
end

const StateFilter = Union{UKF, PF, EnKF}

# --- inference axis 2: hyperparameter inference -----------------------------------------------

"""
    Optimise(; reopt_interval = 1, maxiters = 50, maxiters_burnin = 300, window_length = nothing,
             warm_start = true)                          # [hyper.optimise]

Maximise the filter's marginal log-posterior over data replays every `reopt_interval` origins.
`maxiters_burnin` caps every optimiser stage on the first (and any cold-started) optimisation,
`maxiters` on warm-started ones. `window_length` scores only the most recent observations from a
rolling filter checkpoint. `warm_start = false` restarts from the configured values each time.
"""
@option "optimise" struct Optimise
    reopt_interval::Int = 1
    maxiters::Int = 50
    maxiters_burnin::Int = 300
    window_length::Union{Nothing, Int} = nothing
    warm_start::Bool = true
end

"""
    DEFAULT_JITTER_FLOOR_FRACTION

Minimum Liu-West jitter variance per parameter as a fraction of the prior's unconstrained
variance, so a collapsed cloud can re-expand rather than freeze.
"""
const DEFAULT_JITTER_FLOOR_FRACTION = 1.0e-3

"""
    LiuWest(; discount = 0.95, jitter_floor_fraction = DEFAULT_JITTER_FLOOR_FRACTION,
            forgetting_memory_days = Dict(), replay_on_revision = true)   # [hyper.liu_west]

Learn static hyperparameters online in the particle cloud by the Liu-West shrink-jitter kernel.
`forgetting_memory_days` (parameter => days) adds Kulhavý forgetting toward the prior, merged
over the model's own defaults. `replay_on_revision` tells a backtest runner whether to rebuild the
filter when already-assimilated data are revised.
"""
@option "liu_west" struct LiuWest
    discount::Float64 = 0.95
    jitter_floor_fraction::Float64 = DEFAULT_JITTER_FLOOR_FRACTION
    forgetting_memory_days::Dict{String, Float64} = Dict{String, Float64}()
    replay_on_revision::Bool = true
end

"""
    EKP(; n_ensemble, iterations, burnin_iterations, reopt_interval = 1, inflation = 0.0,
        window_length = nothing, warm_start = true, threads = false)   # [hyper.ekp]

Outer ensemble Kalman inversion of the static parameters, each candidate scored by a complete
inner filter replay. `burnin_iterations` at the first origin, `iterations` when warm-starting
from the previous ensemble. `threads` parallelises candidate replays (not together with
`filter.enkf.threads`).
"""
@option "ekp" struct EKP
    n_ensemble::Int
    iterations::Int
    burnin_iterations::Int
    reopt_interval::Int = 1
    inflation::Float64 = 0.0
    window_length::Union{Nothing, Int} = nothing
    warm_start::Bool = true
    threads::Bool = false
end

const HyperMethod = Union{Optimise, LiuWest, EKP}

"""
    option_alias(x) -> String

The TOML alias of an `@option` value, e.g. `option_alias(UKF()) == "ukf"`.
"""
option_alias(x) = Configurations.type_alias(typeof(x))

# --- run config -------------------------------------------------------------------------------

"""
    RunIO(; data, model_id, forecast_df, loc, locations = String[])   # [io]

Per-run I/O labels: the reporting-triangle `data` path, the `model_id` and `forecast_df` output
labels, the run `loc`, and the ordered `locations` of a joint multi-location model.
"""
@option struct RunIO
    data::String
    model_id::String
    forecast_df::String
    loc::String
    locations::Vector{String} = String[]
end

"""
    RunConfig

One run's TOML: the `[io]` block, the forecast horizon and draw count, the observation cadence
(`step_days`, `burnin_observations`, `drop_recent_observations`), the `[input.<name>]` scale, the
inference axes `[filter.<name>]` and `[hyper.<name>]`, the `[epi.<submodel>]` block (kept opaque
for the submodel to parse with its own `@option` type) and `[priors]` overriding the submodel's
defaults per key. `origin_mode` is `sequential`, `independent` or `forked` (PF + Liu-West only).
"""
@option struct RunConfig
    io::RunIO
    n_ahead::Int
    n_draws::Int = 2000
    step_days::Float64
    supersample::Int = 2
    seed::Int = 1
    burnin_observations::Int
    drop_recent_observations::Int = 0
    forecast_start::Union{Nothing, String} = nothing
    origin_mode::String = "sequential"
    learn_params::Vector{String} = String[]
    input::InputConfig
    filter::StateFilter
    hyper::HyperMethod
    epi::Dict{String, Any} = Dict{String, Any}()
    priors::Dict{String, PriorSpec} = Dict{String, PriorSpec}()
end

_require(ok::Bool, message) = ok || throw(ArgumentError(message))

_validate(f::UKF) = (_require(f.obs_jitter > 0, "ukf.obs_jitter must be positive, got $(f.obs_jitter)"); f)
_validate(f::PF) = (_require(f.n_particles > 0, "pf.n_particles must be positive, got $(f.n_particles)"); f)
function _validate(f::EnKF)
    _require(f.n_ensemble >= 2, "enkf.n_ensemble must be at least 2, got $(f.n_ensemble)")
    _require(f.inflation >= 1, "enkf.inflation must be at least 1, got $(f.inflation)")
    return f
end

function _validate(h::Optimise)
    _require(h.reopt_interval > 0, "optimise.reopt_interval must be positive, got $(h.reopt_interval)")
    _require(h.maxiters > 0 && h.maxiters_burnin > 0, "optimise.maxiters and maxiters_burnin must be positive")
    _validate_window(h.window_length)
    return h
end
function _validate(h::LiuWest)
    _require(1 / 3 <= h.discount <= 1, "liu_west.discount must be in [1/3, 1], got $(h.discount)")
    _require(h.jitter_floor_fraction >= 0, "liu_west.jitter_floor_fraction must be non-negative")
    for (name, days) in h.forgetting_memory_days
        _require(days > 0, "liu_west.forgetting_memory_days[$name] must be positive (Inf disables), got $days")
    end
    return h
end
function _validate(h::EKP)
    _require(h.n_ensemble >= 2, "ekp.n_ensemble must be at least 2, got $(h.n_ensemble)")
    _require(h.iterations > 0 && h.burnin_iterations > 0, "ekp.iterations and burnin_iterations must be positive")
    _require(h.reopt_interval > 0, "ekp.reopt_interval must be positive, got $(h.reopt_interval)")
    _require(h.inflation >= 0, "ekp.inflation must be non-negative, got $(h.inflation)")
    _validate_window(h.window_length)
    return h
end
_validate_window(::Nothing) = nothing
_validate_window(w::Integer) = _require(w > 0, "window_length must be positive when given, got $w")

_validate(::CountInput) = nothing
_validate(i::PercentInput) =
    _require(isfinite(i.annual_rate_per_100) && i.annual_rate_per_100 > 0, "input.percent.annual_rate_per_100 must be finite and positive")

"""
    validate_run_config(cfg::RunConfig) -> cfg

Check what the schema cannot: a positive whole number of `step_days`, the cadence counts,
`origin_mode`, the input scale and both inference axes.
"""
function validate_run_config(cfg::RunConfig)
    _require(
        isfinite(cfg.step_days) && isinteger(cfg.step_days) && cfg.step_days > 0,
        "step_days must be a positive whole number of days, got $(cfg.step_days)",
    )
    _require(cfg.n_ahead >= 1 && cfg.n_draws >= 1 && cfg.supersample >= 1, "n_ahead, n_draws and supersample must be >= 1")
    _require(cfg.burnin_observations >= 1, "burnin_observations must be >= 1, got $(cfg.burnin_observations)")
    _require(cfg.drop_recent_observations >= 0, "drop_recent_observations must be non-negative")
    _require(
        cfg.origin_mode in ("sequential", "independent", "forked"),
        "origin_mode must be sequential, independent or forked; got $(repr(cfg.origin_mode))",
    )
    _require(
        cfg.origin_mode != "forked" || (cfg.filter isa PF && cfg.hyper isa LiuWest),
        "origin_mode = forked carries a particle cloud between origins and requires [filter.pf] + [hyper.liu_west]",
    )
    _validate(cfg.input)
    _validate(cfg.filter)
    _validate(cfg.hyper)
    return cfg
end

"""
    submodel_name(cfg::RunConfig) -> String

The single `[epi.<name>]` key.
"""
function submodel_name(cfg::RunConfig)
    _require(length(cfg.epi) == 1, "run config needs exactly one [epi.<submodel>] block, got $(sort!(collect(keys(cfg.epi))))")
    return only(keys(cfg.epi))
end

"""
    resolve_prior_specs(cfg::RunConfig, default_priors) -> Dict{String, PriorSpec}
    resolve_priors(cfg::RunConfig, default_priors) -> NamedTuple

The submodel's `default_priors()` with the run config's `[priors]` merged over them per key, as
specs or as built distributions.
"""
resolve_prior_specs(cfg::RunConfig, default_priors) = merge(default_priors(), cfg.priors)
resolve_priors(cfg::RunConfig, default_priors) = build_priors(resolve_prior_specs(cfg, default_priors))
