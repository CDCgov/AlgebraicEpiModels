# Ascertainment (observations per infection) as a function of model time: a declining path with
# one learned rate, or a trend model in which log-ascertainment is a latent state. The level,
# rate prior, floor, reference date and rate bound are signal-specific and belong to the caller's
# submodel config and priors; this file carries no defaults for them.

"""
    AscertainmentPath(floor_fraction, t_ref_days, rate_bound = Inf)
    AscertainmentPath(; floor_fraction, reference_date::Date, start_date::Date, rate_bound = Inf)

Callable `(hyper, t)` and `(latent, hyper, t)` giving observations per infection at model time `t`,

    alpha(t) = level * (1 + (1 - floor_fraction) * expm1(-rate * (t - t_ref_days) / 365.25))

with `level = hyper.ascertainment` and `rate = clamp(hyper.ascertainment_decline_rate, ±rate_bound)`
read on every call, so a learned rate reaches each particle or optimiser candidate. A zero rate
returns `level` bit-for-bit. Use it as an observation spec's `mean_modifier` and as
`t -> path(hyper, t)` in the initial-state inversion.
"""
struct AscertainmentPath
    floor_fraction::Float64
    t_ref_days::Float64
    rate_bound::Float64
    function AscertainmentPath(floor_fraction::Real, t_ref_days::Real, rate_bound::Real = Inf)
        isfinite(floor_fraction) && 0 <= floor_fraction <= 1 ||
            throw(ArgumentError("ascertainment_floor_fraction must lie in [0, 1]; got $floor_fraction"))
        isfinite(t_ref_days) || throw(ArgumentError("the ascertainment reference time must be finite; got $t_ref_days"))
        !isnan(rate_bound) && rate_bound > 0 ||
            throw(ArgumentError("ascertainment_rate_bound must be > 0 (per year); got $rate_bound"))
        return new(floor_fraction, t_ref_days, rate_bound)
    end
end

AscertainmentPath(; floor_fraction, reference_date::Date, start_date::Date, rate_bound::Real = Inf) =
    AscertainmentPath(floor_fraction, Dates.value(reference_date - start_date), rate_bound)

"""
    ascertainment_at(level, floor_fraction, rate, days_since_ref)

The path's kernel, `level * (1 + (1 - floor_fraction) * expm1(-rate * days_since_ref / 365.25))`.
"""
@inline ascertainment_at(level, floor_fraction, rate, days_since_ref) =
    level * (1 + (1 - floor_fraction) * expm1(-rate * (days_since_ref / 365.25)))

@inline function (a::AscertainmentPath)(hyper, t)
    rate = clamp(hyper.ascertainment_decline_rate, -a.rate_bound, a.rate_bound)
    return ascertainment_at(hyper.ascertainment, a.floor_fraction, rate, t - a.t_ref_days)
end
@inline (a::AscertainmentPath)(latent, hyper, t) = a(hyper, t)

"""
    parse_ascertainment_reference_date(value) -> Date

Parse the ISO `YYYY-MM-DD` reference date carried as a string in the epi config.
"""
function parse_ascertainment_reference_date(value)
    value isa AbstractString ||
        throw(ArgumentError("ascertainment_reference_date must be an ISO date string (YYYY-MM-DD); got $(repr(value))"))
    try
        return Date(value, "yyyy-mm-dd")
    catch err
        throw(
            ArgumentError(
                "ascertainment_reference_date must be an ISO date string (YYYY-MM-DD); got $(repr(value)) " *
                    "($(sprint(showerror, err)))",
            )
        )
    end
end

"""
    validate_ascertainment(fixed)

Check the `ascertainment`, `ascertainment_decline_rate`, `ascertainment_floor_fraction`,
`ascertainment_rate_bound` and `ascertainment_reference_date` fields of a submodel's fixed config.
"""
function validate_ascertainment(fixed)
    isfinite(fixed.ascertainment) && fixed.ascertainment > 0 ||
        throw(ArgumentError("ascertainment must be finite and > 0; got $(fixed.ascertainment)"))
    isfinite(fixed.ascertainment_decline_rate) ||
        throw(ArgumentError("ascertainment_decline_rate (per year) must be finite; got $(fixed.ascertainment_decline_rate)"))
    f, b = fixed.ascertainment_floor_fraction, fixed.ascertainment_rate_bound
    isfinite(f) && 0 <= f <= 1 || throw(ArgumentError("ascertainment_floor_fraction must lie in [0, 1]; got $f"))
    !isnan(b) && b > 0 || throw(ArgumentError("ascertainment_rate_bound must be > 0 (per year; Inf for none); got $b"))
    parse_ascertainment_reference_date(fixed.ascertainment_reference_date)
    return nothing
end

"""
    build_ascertainment_path(fixed, start_date::Date) -> AscertainmentPath

Anchor a submodel's ascertainment block so model time `t = 0` is `start_date`.
"""
function build_ascertainment_path(fixed, start_date::Date)
    validate_ascertainment(fixed)
    return AscertainmentPath(;
        floor_fraction = fixed.ascertainment_floor_fraction,
        reference_date = parse_ascertainment_reference_date(fixed.ascertainment_reference_date),
        start_date, rate_bound = fixed.ascertainment_rate_bound,
    )
end

"""
    assert_ascertainment_learnable(fixed, learn_params)

Refuse to learn the decline rate when `ascertainment_floor_fraction == 1`, where the path never
reads it.
"""
function assert_ascertainment_learnable(fixed, learn_params)
    :ascertainment_decline_rate in Symbol.(learn_params) && fixed.ascertainment_floor_fraction == 1 && error(
        "cannot learn `ascertainment_decline_rate` with ascertainment_floor_fraction = 1: the path " *
            "is then the constant level and never reads the rate",
    )
    return nothing
end

# --- trend ascertainment: log-ascertainment as an integrated Brownian motion --------------------

"Latent name of the ascertainment level (observations per infection now); positive, stored as `log alpha`."
const ASCERTAINMENT_TREND_LEVEL = :ascertainment_level

"Latent name of the decline rate per year; the same name (and prior) as the declining path's hyperparameter."
const ASCERTAINMENT_TREND_RATE = :ascertainment_decline_rate

"Hyperparameter name of the trend's bend scale; see [`DEFAULT_ASCERTAINMENT_TREND_WANDER`](@ref)."
const ASCERTAINMENT_TREND_WANDER = :ascertainment_trend_wander

"""
The 1-sd departure of log-ascertainment from a straight line after one year, in log units: too
stiff to mimic a wave, loose enough for the rate to re-diversify over a season.
"""
const DEFAULT_ASCERTAINMENT_TREND_WANDER = 0.1

"Memory (days) of the Kulhavý forgetting applied to a learned wander: a season and a half."
const DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS = 1.5 * 365.25

"Initial spread of the level in log units (pinned: the level is not identifiable alongside transmission)."
const ASCERTAINMENT_TREND_LEVEL_LOG_SD = 0.01

"""
    ascertainment_trend_sigma_rate(wander) -> Float64

The rate's random-walk diffusion per sqrt-day that makes the level's one-year departure from a
straight line equal `wander`: `wander * sqrt(3 / 365.25)`.
"""
ascertainment_trend_sigma_rate(wander::Real) = wander * sqrt(3 / 365.25)

"""
    TrendAscertainment()

The trend model's observation `mean_modifier`: `(latent, hyper, t) -> latent.ascertainment_level`.
"""
struct TrendAscertainment end
@inline (::TrendAscertainment)(latent, hyper, t) = latent[ASCERTAINMENT_TREND_LEVEL]

"""
    TrendAscertainmentSeed(level0, decline_rate)

`t -> level0 * exp(-decline_rate * t / 365.25)` for the initial-state inversion, which evaluates
ascertainment at and before `t = 0`.
"""
struct TrendAscertainmentSeed
    level0::Float64
    decline_rate::Float64
end
@inline (s::TrendAscertainmentSeed)(t) = s.level0 * exp(-s.decline_rate * t / 365.25)

"""
    ascertainment_trend_specs(; rate_prior, level0) -> (rate_spec, level_spec)

The trend model's two drivers: the decline rate as a random walk whose diffusion derives from
the `ascertainment_trend_wander` hyperparameter, and the level as the noise-free integral of
`-rate` per year. `rate_prior` must be named `ascertainment_decline_rate`.
"""
function ascertainment_trend_specs(; rate_prior::ParameterDistribution, level0::Real)
    isfinite(level0) && level0 > 0 ||
        throw(ArgumentError("trend ascertainment level0 must be positive and finite, got $level0"))
    level_prior = positive_gaussian(ASCERTAINMENT_TREND_LEVEL, level0, ASCERTAINMENT_TREND_LEVEL_LOG_SD * level0)
    sigma_rate = DerivedParam(
        Symbol(ASCERTAINMENT_TREND_RATE, :_sigma_rate),
        θ -> ascertainment_trend_sigma_rate(θ[ASCERTAINMENT_TREND_WANDER]),
    )
    return (
        RWParamSpec(ASCERTAINMENT_TREND_RATE; init = rate_prior, sigma_rate),
        IntegratedParamSpec(ASCERTAINMENT_TREND_LEVEL; init = level_prior, rate = ASCERTAINMENT_TREND_RATE, per_days = -365.25),
    )
end
