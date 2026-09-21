# ============================================================================
# Declining ascertainment path — observations per infection as a function of model time
# ============================================================================
#
# Why this exists: `reports/susceptible-reconstruction/report.md`. Under a CONSTANT ascertainment
# the multi-season fall in the observed level has to be a fall in infections, so the susceptible
# pool refills, the mechanism runs supercritical and the AR(1) latent must hold it down in-sample
# — a correction that reverts within one horizon step out of sample (issue #562, H5). A declining
# ascertainment lets the same observations be flat infections.
#
# The path is the report's functional form,
#
#     alpha(t) = alpha_min + (alpha_ref - alpha_min) * exp(-r * (t - t_ref) / 365.25)
#
# with `alpha_min = floor_fraction * alpha_ref`, implemented as
#
#     alpha(t) = alpha_ref * (1 + (1 - floor_fraction) * expm1(-r * (t - t_ref) / 365.25))
#
# which is the same function but bit-exact at `r = 0` (`expm1(0.0) === 0.0`, so the path IS the
# constant it replaces) and free of `1 - exp(x)` cancellation for small rates.
#
# Three quantities, three provenances (docs/parameter-provenance.md §2.3):
#   * `hyper.ascertainment`            the LEVEL at the reference date — fixed, not identifiable
#                                      alongside the transmission level.
#   * `hyper.ascertainment_decline_rate` the RATE per year — LEARNED, informed Normal prior whose
#                                      90 % interval runs from "no decline" to "all of the observed
#                                      decline is ascertainment". Not truncated at zero.
#   * `floor_fraction`                 the FLOOR as a fraction of the level — asserted from the
#                                      SEIRS waning ceiling on infections; captured in the struct.
#
# The level and the rate are read from `hyper` on EVERY call rather than captured at build time, so
# a learned rate reaches the observation mean per particle (`merge(p, learned.extract(x))`) and per
# UKF candidate, and `learn_params = ["ascertainment"]` stays a live dimension.
#
# Time: `t` is the model clock, days since `ctx.start_date` (the first reference date of the
# window, constant across backtest origins). The reference date is a FIXED CALENDAR DATE converted
# once at build time to `t_ref_days`, exactly as `build_seasonal_forcing` anchors the seasonal
# curve — tying it to the window start instead would rotate the level whenever the window moved.
# ============================================================================

"""
    ASCERTAINMENT_DECLINE_RATE_PRIOR_MEAN, ASCERTAINMENT_DECLINE_RATE_PRIOR_SD

Prior on the decline rate `r` (per year) for the NSSP ED-visit signal, from
`reports/susceptible-reconstruction/report.md` Table 4: the annual factor `phi = exp(-r)` has
`log(phi) ~ N(log(phi_obs) / 2, sigma^2)` with `phi_obs = 0.678` the all-ascertainment reading and
`sigma = -log(phi_obs) / (2 * 1.6449)` so that `phi = 1` and `phi = phi_obs` are the 90 % interval.
Equivalently `r ~ N(0.1943, 0.1181^2)`. The NHSN admissions signal (`phi_obs = 0.574`) gives
`N(0.2775, 0.1687^2)`; set it through `[priors.ascertainment_decline_rate]` when observing admissions.
"""
const ASCERTAINMENT_DECLINE_RATE_PRIOR_MEAN = 0.1943
const ASCERTAINMENT_DECLINE_RATE_PRIOR_SD = 0.1181

"""
    DEFAULT_ASCERTAINMENT_DECLINE_RATE

Fixed-config default for the rate: the prior mean, because the config value is an inference START
value (the UKF optimiser seed, the Liu–West cloud centre, the EKP initial estimate), not an
operating constant. `0.0` recovers the constant ascertainment exactly.
"""
const DEFAULT_ASCERTAINMENT_DECLINE_RATE = ASCERTAINMENT_DECLINE_RATE_PRIOR_MEAN

"""
    DEFAULT_ASCERTAINMENT_FLOOR_FRACTION

Lowest ascertainment the path can reach, as a fraction of the level. Reasoned from the model rather
than asserted blind: SEIRS waning caps the supply of infections at `365 * omega ≈ 2.03` per person
per year (`immunity = 180 d`), and under a constant ratio the latest season implies about 0.50
infections per person per year, so at today's observed level the model cannot support an
ascertainment below `0.50 / 2.03 ≈ 0.25` of the level. `0.2` leaves a further ~20 % fall in the
observed level before the floor binds; at the prior-mean rate the path only reaches it about
eleven years after the reference date, so the floor is inert in-window and exists to bound the
exponential at long horizons.
"""
const DEFAULT_ASCERTAINMENT_FLOOR_FRACTION = 0.2

"""
    DEFAULT_ASCERTAINMENT_REFERENCE_DATE

Calendar date at which `ascertainment` is the level (ISO `YYYY-MM-DD`). The July season boundary in
the middle of the report's evidence window: the identity-implied level `0.005` was fitted to that
window's mean, and the all-ascertainment log-linear fit through the season levels crosses `0.005`
between May and July 2024, so the level keeps its register meaning. Moving this date requires
re-deriving the level.
"""
const DEFAULT_ASCERTAINMENT_REFERENCE_DATE = "2024-07-01"

"""
    DEFAULT_ASCERTAINMENT_RATE_BOUND

Symmetric bound (per year) on the rate the path actually uses: `clamp(rate, -b, b)`. Measured need,
not a prior statement: at the first backtest origins (12–24 weeks after a peak start) the UKF
objective sits on its divergence plateau, and an unbounded rate let the optimiser walk to
`r ≈ 23`/yr — where the pre-reference growth makes the seed absurd (hundreds of observations per
infection at `t = 0`) but the filter is finite — and a warm start then never left that basin.
`1.0`/yr bounds the annual factor to `[e^-1, e] = [0.37, 2.7]`, far outside the prior's 90 % band,
so nothing the report considers coherent is touched; a rate of zero is untouched by construction.
"""
const DEFAULT_ASCERTAINMENT_RATE_BOUND = 1.0

const _ASCERTAINMENT_DATE_FORMAT = "yyyy-mm-dd"

"""
    AscertainmentPath(floor_fraction, t_ref_days)
    AscertainmentPath(; floor_fraction, reference_date::Date, start_date::Date)

Callable `(hyper, t) -> Real`: observations per infection at model time `t` (days since the
window start),

    alpha(t) = level * (1 + (1 - floor_fraction) * expm1(-rate * (t - t_ref_days) / 365.25))

with `level = hyper.ascertainment` and `rate = hyper.ascertainment_decline_rate` (per year) read
from `hyper` on every call. Away from the floor `log(alpha)` is linear in `t` with slope `-rate`
per year, so the annual factor is `exp(-rate)`; `alpha(t_ref_days) == level` for every rate, and a
rate of exactly zero returns `level` bit-for-bit. The path is monotone: decreasing towards
`floor_fraction * level` for a positive rate, increasing without bound for a negative one (the prior
is deliberately not truncated at zero, so a genuine increase can surface rather than be absorbed
elsewhere).

A single concrete type, so unlike the seasonal forcing it needs no `where {F}` function barrier.
Use it directly as an observation spec's `mean_modifier`, and as `t -> path(hyper, t)` wherever a
seed or anchor reconstructs infections from counts (`initial_infection_state`,
`peak_anchored_susceptible_fraction`).
"""
struct AscertainmentPath
    floor_fraction::Float64
    t_ref_days::Float64
    rate_bound::Float64   # the rate is clamped to ±rate_bound per year before use (Inf = none)

    function AscertainmentPath(floor_fraction::Real, t_ref_days::Real, rate_bound::Real = Inf)
        (isfinite(floor_fraction) && 0 <= floor_fraction <= 1) || throw(
            ArgumentError(
                "ascertainment_floor_fraction must lie in [0, 1] (a fraction of the level); " *
                    "got $floor_fraction"
            )
        )
        isfinite(t_ref_days) || throw(
            ArgumentError("the ascertainment reference time must be finite; got $t_ref_days")
        )
        (!isnan(rate_bound) && rate_bound > 0) || throw(
            ArgumentError("ascertainment_rate_bound must be > 0 (per year); got $rate_bound")
        )
        return new(Float64(floor_fraction), Float64(t_ref_days), Float64(rate_bound))
    end
end

function AscertainmentPath(;
        floor_fraction, reference_date::Date, start_date::Date, rate_bound::Real = Inf
    )
    return AscertainmentPath(
        floor_fraction, Float64(Dates.value(reference_date - start_date)), rate_bound
    )
end

"""
    ascertainment_at(level, floor_fraction, rate, days_since_ref) -> Real

The path's kernel: `level * (1 + (1 - floor_fraction) * expm1(-rate * days_since_ref / 365.25))`.
`days_since_ref` may be negative (before the reference date the path sits ABOVE the level). Pure
arithmetic on its arguments, so it differentiates through `ForwardDiff` when `level` or `rate` are
duals, and returns `level` exactly when `rate` is zero.
"""
@inline function ascertainment_at(level, floor_fraction, rate, days_since_ref)
    return level * (1 + (1 - floor_fraction) * expm1(-rate * (days_since_ref / 365.25)))
end

@inline function (a::AscertainmentPath)(hyper, t)
    # `clamp` is the identity inside the bound (so a zero rate stays bit-exact) and has zero
    # likelihood gradient outside it, where only the prior pulls the rate back.
    rate = clamp(hyper.ascertainment_decline_rate, -a.rate_bound, a.rate_bound)
    return ascertainment_at(hyper.ascertainment, a.floor_fraction, rate, t - a.t_ref_days)
end

# The observation spec's `mean_modifier` contract is `(latent, hyper, t)`; the path reads nothing
# from the latent state, so the same struct serves as a modifier and as the `(hyper, t)` callable
# the seed, the anchor and the accumulator variance use.
@inline (a::AscertainmentPath)(latent, hyper, t) = a(hyper, t)

"""
    parse_ascertainment_reference_date(value) -> Date

Parse the ISO `YYYY-MM-DD` reference date carried as a `String` in the epi config (a `String`
rather than a `Date` field so the TOML boundary and the provenance register see a scalar), naming
the field in the error so a malformed value fails at config-validation time.
"""
function parse_ascertainment_reference_date(value)
    value isa AbstractString || throw(
        ArgumentError(
            "ascertainment_reference_date must be an ISO date string (YYYY-MM-DD); " *
                "got $(repr(value))"
        )
    )
    try
        return Date(value, _ASCERTAINMENT_DATE_FORMAT)
    catch err
        throw(
            ArgumentError(
                "ascertainment_reference_date must be an ISO date string (YYYY-MM-DD); " *
                    "got $(repr(value)) ($(sprint(showerror, err)))"
            )
        )
    end
end

"""
    validate_ascertainment(fixed)

Check a submodel's ascertainment block (the `ascertainment`, `ascertainment_decline_rate`,
`ascertainment_floor_fraction`, `ascertainment_rate_bound` and `ascertainment_reference_date`
fields of its `<Model>Fixed` config). Called from every submodel's `parse_epi_config`, which is also where `validate_config.jl`
and the cross-language drift test look, so a bad value fails at config-parse time rather than
deep inside model construction.
"""
function validate_ascertainment(fixed)
    level = fixed.ascertainment
    (isfinite(level) && level > 0) || throw(
        ArgumentError(
            "ascertainment (observations per infection at the reference date) must be finite " *
                "and > 0; got $level"
        )
    )
    isfinite(fixed.ascertainment_decline_rate) || throw(
        ArgumentError(
            "ascertainment_decline_rate (per year) must be finite; got " *
                "$(fixed.ascertainment_decline_rate)"
        )
    )
    f = fixed.ascertainment_floor_fraction
    (isfinite(f) && 0 <= f <= 1) || throw(
        ArgumentError(
            "ascertainment_floor_fraction must lie in [0, 1] (a fraction of the level); got $f"
        )
    )
    b = fixed.ascertainment_rate_bound
    (!isnan(b) && b > 0) || throw(
        ArgumentError("ascertainment_rate_bound must be > 0 (per year; Inf for none); got $b")
    )
    parse_ascertainment_reference_date(fixed.ascertainment_reference_date)
    return nothing
end

"""
    build_ascertainment_path(fixed, start_date::Date) -> AscertainmentPath

Anchor a submodel's ascertainment block so that model time `t = 0` is `start_date`: the
configured calendar reference date becomes `t_ref_days`, once, at build time — the same contract
as `build_seasonal_forcing`. `fixed` is the submodel's `<Model>Fixed` config (duck-typed: any
object carrying the four ascertainment fields).
"""
function build_ascertainment_path(fixed, start_date::Date)
    validate_ascertainment(fixed)
    return AscertainmentPath(;
        floor_fraction = fixed.ascertainment_floor_fraction,
        reference_date = parse_ascertainment_reference_date(fixed.ascertainment_reference_date),
        start_date,
        rate_bound = fixed.ascertainment_rate_bound,
    )
end

"""
    assert_ascertainment_learnable(fixed, learn_params)

Refuse to learn `ascertainment_decline_rate` when `ascertainment_floor_fraction == 1`: the path
then never reads the rate, so it is an unidentified dimension that a learner would wander along
without error. The observation-side analogue of `assert_seasonal_learnable`.
"""
function assert_ascertainment_learnable(fixed, learn_params)
    if :ascertainment_decline_rate in Symbol.(learn_params) && fixed.ascertainment_floor_fraction == 1
        error(
            "cannot learn `ascertainment_decline_rate` with ascertainment_floor_fraction = 1: the " *
                "path is then the constant level and never reads the rate, so it is unidentified. " *
                "Lower the floor fraction or remove the rate from learn_params."
        )
    end
    return nothing
end
