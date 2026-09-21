# ============================================================================
# Trend ascertainment — observations per infection as an integrated Brownian motion
# ============================================================================
#
# Why this exists. The declining path in src/ascertainment.jl makes ascertainment a deterministic
# function of ONE static rate: `alpha(t) = alpha_ref · exp(-r · (t - t_ref) / 365.25)` (plus a
# floor). That suits a full-history replay, which re-integrates the whole state under each
# candidate rate. It is hostile to a one-pass particle filter, for two reasons:
#
#   * The rate's leverage on the observation is proportional to `t - t_ref`. It is zero at the
#     reference date and grows without bound, so the cloud is pruned by resampling while the rate
#     is unidentified, and the signal arrives when the cloud can no longer move.
#   * The rate is a PATH parameter. Liu-West jitters it but keeps a susceptible pool that was
#     built under the old value, so parameter and state stop agreeing.
#
# Here log-ascertainment is a STATE instead: a level advanced each step by a decline-rate state,
# the rate itself a driftless random walk. That pair is an integrated Brownian motion (a local
# linear trend). It favours log-linear extrapolation — the conditional mean continues along the
# current slope — a change to the rate only alters FUTURE increments, and its leverage on the next
# observation is the same at every time. Particle state and parameter stay consistent by
# construction.
#
# The one hyperparameter is how fast the line may bend, stated as an OUTPUT so the prior excludes
# rather than identifies: `ascertainment_trend_wander` is the 1-sd departure of log-ascertainment
# from a straight line after ONE YEAR. It is also the level-ridge guard. The level is confounded
# with the transmission level (ADR-0002), so the data cannot pull a wandering level back; this
# number is what bounds it. It is learned by default in the trend submodel because, unlike the old
# path-wide slope, changing it only controls FUTURE state innovations. Its prior is deliberately
# tight; an explicit `learn_params` list may omit it for a fixed-wander ablation.
# ============================================================================

"Latent name of the ascertainment LEVEL (observations per infection now). Positive, so its slot is `log alpha`."
const ASCERTAINMENT_TREND_LEVEL = :ascertainment_level

"""
Latent name of the ascertainment DECLINE RATE per year. It deliberately reuses the declining
path's hyperparameter name: the quantity, its sign convention and its prior are the same, and the
latent shadows the (then unused) hyperparameter wherever both are merged.
"""
const ASCERTAINMENT_TREND_RATE = :ascertainment_decline_rate

"Hyperparameter name of the trend's bend scale; see [`DEFAULT_ASCERTAINMENT_TREND_WANDER`](@ref)."
const ASCERTAINMENT_TREND_WANDER = :ascertainment_trend_wander

"""
    DEFAULT_ASCERTAINMENT_TREND_WANDER

The 1-sd departure of log-ascertainment from a straight line after one year, in log units. `0.1`
says a year's drift may bend the log-linear extrapolation by about 10 %: far too stiff to mimic a
wave (whose log amplitude is of order 1 over a few months, the latent `Rt`'s job), loose enough
that the decline rate re-diversifies to most of its prior sd over a season, so the rate can track
a genuine change. The departure grows as `T^1.5`: about 0.4 log units after 2.5 years.
"""
const DEFAULT_ASCERTAINMENT_TREND_WANDER = 0.1

"""
    DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS

Memory, in days, of the Kulhavý forgetting toward the prior that the trend submodel applies to a
LEARNED `ascertainment_trend_wander`: a season and a half, `1.5 × 365.25`.

The wander is weakly identified — one day's count says almost nothing about how fast the decline
rate may change — and Liu-West uses a prior only once, to scatter the initial cloud. Without a
standing pull the learned wander is free to drift, and it is also the level-ridge guard, so a
drifting wander loosens its own guard. With forgetting the prior is the fixed point: information
about the wander older than about this long is discounted, and with none the cloud relaxes to the
prior. It is long next to the latent `Rt`'s 30-day correlation time, so the two stay on separate
clocks. The pull is precision-weighted, so it governs a healthy cloud; a collapsed one is
re-widened by the Liu-West jitter floor first (see `build_hyperparam_updater`).
"""
const DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS = 1.5 * 365.25

"""
    ASCERTAINMENT_TREND_LEVEL_LOG_SD

Initial spread of the level in log units. The level is PINNED at the start of the window (it is not
identifiable alongside the transmission level, ADR-0002); 1 % is "known", kept nonzero only so no
covariance is degenerate. Level diversity then arises solely from each particle's own rate history.
"""
const ASCERTAINMENT_TREND_LEVEL_LOG_SD = 0.01

"""
    ascertainment_trend_sigma_rate(wander) -> Float64

The decline rate's random-walk diffusion, per sqrt-day in per-year rate units, that makes the
level's 1-sd departure from a straight line after one year equal `wander`.

With the rate `r` (per year) a Brownian motion of diffusion `s` per sqrt-day and the level
advancing by `-r dt / 365.25`, the departure after `T` days has sd `(s / 365.25) sqrt(T^3 / 3)`.
At `T = 365.25` that is `s sqrt(365.25 / 3)`, hence `s = wander * sqrt(3 / 365.25)`.
"""
ascertainment_trend_sigma_rate(wander::Real) = wander * sqrt(3 / 365.25)

"""
    TrendAscertainment()

The observation `mean_modifier` of the trend model: observations per infection are the latent
level, `(latent, hyper, t) -> latent.ascertainment_level`. There is no `(hyper, t)` form — without
the latent there is no ascertainment to read; the initial-state inversion uses
[`TrendAscertainmentSeed`](@ref) instead.
"""
struct TrendAscertainment end
@inline (::TrendAscertainment)(latent, hyper, t) = latent[ASCERTAINMENT_TREND_LEVEL]

"""
    TrendAscertainmentSeed(level0, decline_rate)

`alpha_at(t) = level0 · exp(-decline_rate · t / 365.25)` for the initial-state inversion, which
evaluates ascertainment at and BEFORE `t = 0`, when historical infections happened. It is a fixed
function — the level's starting value extended backwards at the prior-mean rate — because no
particle carries a level before the window starts.
"""
struct TrendAscertainmentSeed
    level0::Float64
    decline_rate::Float64
end
@inline (s::TrendAscertainmentSeed)(t) = s.level0 * exp(-s.decline_rate * t / 365.25)

"""
    ascertainment_trend_specs(; rate_prior, level0) -> (rate_spec, level_spec)

The two coupled drivers of the trend model: the decline rate as a driftless random walk whose
diffusion is derived from the `ascertainment_trend_wander` hyperparameter (the trend submodel's
default Liu-West ascertainment target), and the level as the noise-free integral of `-rate` per
year. `rate_prior` must carry the name [`ASCERTAINMENT_TREND_RATE`](@ref); it is the declining
path's own prior. `level0` is the level at `t = 0`.
"""
function ascertainment_trend_specs(; rate_prior::ParameterDistribution, level0::Real)
    isfinite(level0) && level0 > 0 || throw(
        ArgumentError("trend ascertainment level0 must be positive and finite, got $level0")
    )
    level_prior = positive_gaussian(
        ASCERTAINMENT_TREND_LEVEL, level0, ASCERTAINMENT_TREND_LEVEL_LOG_SD * level0
    )
    sigma_rate = DerivedParam(
        Symbol(ASCERTAINMENT_TREND_RATE, :_sigma_rate),
        θ -> ascertainment_trend_sigma_rate(θ[ASCERTAINMENT_TREND_WANDER]),
    )
    return (
        RWParamSpec(ASCERTAINMENT_TREND_RATE; init = rate_prior, sigma_rate),
        IntegratedParamSpec(
            ASCERTAINMENT_TREND_LEVEL;
            init = level_prior, rate = ASCERTAINMENT_TREND_RATE, per_days = -365.25,
        ),
    )
end
