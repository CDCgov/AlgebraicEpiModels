# ============================================================================
# DAY-OF-WEEK OBSERVATION EFFECT (daily cadence)
# ============================================================================
#
# Daily surveillance counts carry a predictable weekly cycle (fewer ED visits recorded at the
# weekend, a Monday catch-up). With no term for it, the only thing that can absorb that cycle is
# the transmission latent, which then reads a reporting artefact as a weekly wobble in `Rt`. The
# effect belongs in the OBSERVATION model:
#
#     y_t ~ NB(mean = alpha(t) · w_{d(t)} · incidence_t,  dispersion = phi_{d(t)})
#
# `d(t)` is the calendar weekday of model time `t`. The fit scores row `k` at `t = k - 1` and the
# forecast draws horizon `h` at `t = T - 1 + h`, so calendar date = `start_date + t` on both paths
# (see `fit_forecast!` in build_inference.jl). That identity needs a gap-free daily series, and the
# daily runner checks for one.
#
# The weights are normalised to ARITHMETIC weekly mean 1. A week's expected total is then
# unchanged, and so is the meaning of the ascertainment level; the weights only redistribute counts
# across the week.
#
# The two effects represent uncertainty about the weekday pattern in different ways.
#
# - `plugin` fixes point estimates, so it has to carry the uncertainty in the observation model.
#   Write the realised weekday factor as `w_d · ε` with `E[ε] = 1` and `Var[ε] = s_d²`. Matching
#   variances, `Var(y) ≈ μ + μ²(1/φ + s_d²)` is again a negative binomial, with per-weekday
#   dispersion `φ_d = 1 / (1/φ + s_d²)`. `s_d²` is estimated from the same history, and the PF's
#   exact NB likelihood carries it without changing form.
# - `learned` adds NO extra variance. The weights are particles, and the marginal over the ensemble
#   is the approximate posterior over them, so the uncertainty is already in the predictive mixture.
#   Adding a history-estimated `s_d²` on top would count it twice.
#
# Learned weights use zero-sum HELMERT coordinates. The log-effects are `e = H z`, where `H` is a
# fixed 7×6 orthonormal basis of the sum-to-zero subspace, and `z` has six independent
# unconstrained priors N(0, σ_z²). The weights are then `w = 7 · softmax(e)`: arithmetic weekly
# mean 1, as for the plugin.
#
# Why not fix one reference day at zero? That treats the days unevenly. With the other six iid
# N(0, σ²), the reference day's log-effect relative to the weekly mean has prior sd ≈ 0.35σ, against
# ≈ 0.91σ for every other day. Liu–West's shrink-and-jitter uses the cloud's full covariance, so it
# is unaffected by a linear change of coordinates. What is NOT unaffected is the initial cloud and
# the diagonal jitter floor, so the reference day would be explored ~2.6× more narrowly. For ED
# visits that day (Sunday) is likely one of the most extreme.
#
# An isotropic prior on `z` gives every centred log-effect the same marginal sd `σ_z·√(6/7)` and
# pairwise correlation −1/6. Any orthonormal basis implies the same prior and floor. Helmert is used
# because it is closed-form and therefore reproducible: `LinearAlgebra.nullspace` returns a basis
# that depends on the LAPACK build, so identical RNG draws would give different weekday effects on
# different machines. No Julia package provides a sum-to-zero transform: EKP constraints are
# elementwise, and Bijectors.jl and TransformVariables.jl have simplex transforms but not zero-sum.
# ============================================================================

# `Dates.dayofweek` order: Monday = 1 … Sunday = 7.
const DOW_DAY_ABBREVIATIONS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# Six Helmert coordinates: softmax ignores the common level of the seven log-effects, so the seventh
# direction is one the likelihood never sees; Liu–West would only wander in it.
const DOW_LEARNED_NAMES = (:dow_z1, :dow_z2, :dow_z3, :dow_z4, :dow_z5, :dow_z6)

# Prior sd of each Helmert coordinate that gives every centred weekday log-effect marginal sd
# `sd`: `Var(e_d) = σ_z² (1 − 1/7)`.
dow_helmert_prior_sd(sd::Real) = sd * sqrt(7 / 6)

# Helmert basis, rows Monday … Sunday. Column k (k = 1…6) is `1/√(k(k+1))` in rows 1…k,
# `−k/√(k(k+1))` in row k + 1, and 0 below that. Its columns are orthonormal and each sums to zero.
const DOW_HELMERT = ntuple(
    j -> ntuple(
        k -> j <= k ? 1 / sqrt(k * (k + 1)) : j == k + 1 ? -k / sqrt(k * (k + 1)) : 0.0,
        6,
    ),
    7,
)

"""
    validate_day_of_week(cfg::DayOfWeekConfig) -> cfg

Reject an estimator window that cannot hold four usable weeks of each weekday, at config-parse time.
Which effect is in force is structural (the config's type), so there is nothing else to check.
"""
validate_day_of_week(cfg::Union{NoDayOfWeekConfig, LearnedDayOfWeekConfig}) = cfg

function validate_day_of_week(cfg::PluginDayOfWeekConfig)
    # Four usable weeks of each weekday after losing three days at each end to the centred week.
    cfg.window_days >= 7 * 5 || error(
        "day_of_week window_days must be >= 35 (five weeks); got $(cfg.window_days)"
    )
    cfg.exclude_recent_days >= 0 || error(
        "day_of_week exclude_recent_days must be >= 0; got $(cfg.exclude_recent_days)"
    )
    cfg.fit_policy in ("per_origin", "first_vintage") || error(
        "day_of_week fit_policy must be per_origin or first_vintage; got $(cfg.fit_policy)"
    )
    return cfg
end

"""
    day_of_week_learned_names(cfg::DayOfWeekConfig) -> Tuple

The θ names this weekday effect adds to the learned set: the six Helmert coordinates under
`learned`, none otherwise.
"""
day_of_week_learned_names(::DayOfWeekConfig) = ()
day_of_week_learned_names(::LearnedDayOfWeekConfig) = DOW_LEARNED_NAMES

"""
    assert_day_of_week_learnable(cfg::DayOfWeekConfig, learn_params)

The weekday coordinates in an explicit learned set must be exactly `day_of_week_learned_names(cfg)`.
A partial set leaves the weights half-learned, and a coordinate the observation never reads would be
an unidentified dimension.
"""
function assert_day_of_week_learnable(cfg::DayOfWeekConfig, learn_params)
    expected = day_of_week_learned_names(cfg)
    present = Tuple(n for n in DOW_LEARNED_NAMES if n in learn_params)
    present == expected || error(
        "learn_params weekday coordinates $(present) do not match the $(nameof(typeof(cfg))) " *
            "weekday effect, which learns $(expected). Select [epi.<submodel>.day_of_week.learned] " *
            "to learn all six, or remove them."
    )
    return nothing
end

# Calendar weekday (Monday = 1) of model time `t`, given the weekday of `t = 0`. Model times on the
# daily path are exact integers.
@inline _weekday(dow0::Int, t) = mod1(dow0 + round(Int, t), 7)

"""
    DayOfWeekModifier(inner, dow0, weights)

An observation `mean_modifier` equal to `inner(hyper, t) · w_{d(t)}`. `inner` is the ascertainment
path, and `dow0 = dayofweek(start_date)`. With `weights::NTuple{7}` it is the plugin mode. With
`weights = nothing` the weights are `day_of_week_multipliers(hyper)`, read from the learned Helmert
coordinates `hyper.dow_z1 … dow_z6`.

Like `AscertainmentPath` it supports both `(hyper, t)` and `(latent, hyper, t)`, so the same object
serves as the observation modifier and as the `alpha_at` used by the initial-state inversion.
"""
struct DayOfWeekModifier{A, W}
    inner::A
    dow0::Int
    weights::W
end

@inline _dow_weight(w::NTuple{7, Float64}, hyper, d) = w[d]

# Zero-sum log-effects `e = H z` (Monday … Sunday) from the learned Helmert coordinates.
@inline function dow_log_effects(hyper)
    z = promote(hyper.dow_z1, hyper.dow_z2, hyper.dow_z3, hyper.dow_z4, hyper.dow_z5, hyper.dow_z6)
    return ntuple(j -> sum(DOW_HELMERT[j][k] * z[k] for k in 1:6), 7)
end

"""
    day_of_week_multipliers(hyper) -> NTuple{7}

The learned weekday multipliers, Monday … Sunday: `7 · softmax(H z)`, arithmetic mean 1.

The coordinates are unconstrained, so the softmax is shifted by its maximum log-effect before
exponentiating. That leaves the weights unchanged and keeps a large coordinate from overflowing to
`Inf / Inf = NaN`, which would silently zero a particle's likelihood.
"""
@inline function day_of_week_multipliers(hyper)
    e = dow_log_effects(hyper)
    top = maximum(e)
    total = sum(x -> exp(x - top), e)
    return ntuple(d -> 7 * exp(e[d] - top) / total, 7)
end

@inline _dow_weight(::Nothing, hyper, d) = day_of_week_multipliers(hyper)[d]

"""
    day_of_week_weight(modifier, hyper, t) -> Real

The weekday factor alone, at OBSERVATION time `t`: `w_{d(t)}` for a `DayOfWeekModifier`, and `true`
(the multiplicative identity, so the no-effect path stays bit-exact) for any other modifier.
"""
@inline day_of_week_weight(modifier, hyper, t) = true
@inline day_of_week_weight(m::DayOfWeekModifier, hyper, t) =
    _dow_weight(m.weights, hyper, _weekday(m.dow0, t))

@inline (m::DayOfWeekModifier)(hyper, t) = m.inner(hyper, t) * day_of_week_weight(m, hyper, t)
# The latent is passed THROUGH to `inner`: the declining path ignores it (so this is the same
# number as the two-argument form), while the trend ascertainment reads its level from it.
@inline (m::DayOfWeekModifier)(latent, hyper, t) =
    m.inner(latent, hyper, t) * day_of_week_weight(m, hyper, t)

"""
    remove_day_of_week_effect(history, modifier, hyper) -> history

Divide each historical count by the weekday weight of its own observation date,
`history.times` on the model clock. The initial-state reconstruction then works on weekday-free
counts and inverts them with ascertainment ALONE, evaluated when the infections happened.

The two factors cannot be combined into one `alpha_at(t)`. That reconstruction evaluates
ascertainment at infection times, `t − bin/2 − reporting delay`. A weekday factor evaluated there
would belong to the wrong calendar day, and on the daily path to a half-day. Without a weekday
effect, or without a history, the history is returned unchanged.
"""
remove_day_of_week_effect(history, modifier, hyper) = history
remove_day_of_week_effect(::Nothing, ::DayOfWeekModifier, hyper) = nothing
remove_day_of_week_effect(history, m::DayOfWeekModifier, hyper) = merge(
    history,
    (; counts = [c / day_of_week_weight(m, hyper, t) for (t, c) in zip(history.times, history.counts)]),
)

"""
    HyperPhi()

The negative-binomial dispersion with no weekday widening: `(latent, hyper, t) -> hyper.phi`.
"""
struct HyperPhi end
@inline (::HyperPhi)(latent, hyper, t) = hyper.phi

"""
    DayOfWeekDispersion(dow0, extra_var)

The negative-binomial dispersion `(latent, hyper, t) -> 1 / (1/hyper.phi + extra_var[d(t)])`.
With `extra_var` all zero it returns `hyper.phi`.
"""
struct DayOfWeekDispersion
    dow0::Int
    extra_var::NTuple{7, Float64}
end

@inline (p::DayOfWeekDispersion)(latent, hyper, t) =
    inv(inv(hyper.phi) + p.extra_var[_weekday(p.dow0, t)])

const _NO_DOW_EFFECTS = (
    weights = ntuple(_ -> 1.0, 7),
    extra_var = ntuple(_ -> 0.0, 7),
    fallback = true,
    n_used = 0,
)

"""
    estimate_day_of_week_effects(dates, counts; phi, window_days = 182, exclude_recent_days = 14,
        min_weeks = 4) -> (; weights, extra_var)

A multiplicative decomposition of a gap-free daily count series, against a LEAVE-ONE-OUT baseline:

1. Drop the newest `exclude_recent_days` and keep the trailing `window_days`.
2. `b_t` is the mean of the OTHER six days of the centred week around `t`. Excluding `y_t` is what
   makes the variance step unbiased: an ordinary centred mean contains `y_t` itself, so a high
   Monday raises its own baseline. That shrinks the variance terms by `(1 − w_d/7)² ≈ 0.7`, and
   a simulated `s² = 0.02` is expected back at about 0.013. The six days contain every weekday
   except `d`, so `E[b_t] = μ_t (7 − w_d)/6` for weights of mean 1. Under growth rate `r_t` the
   weights take a first-order bias scaled by the weekday asymmetry around `d`: up to 1–3% at a
   sustained 7-day doubling (`r = 0.1`), and it cancels over waves.
3. `q_d = Σ y / Σ b` over the days with weekday `d` estimates `6 w_d / (7 − w_d)`. Invert with
   `w_d = 7 q_d / (6 + q_d)`, then normalise to mean 1. A ratio of sums is robust to zeros and small
   counts.
4. With `u_t = y_t / e_t`, where `e_t = b_t · 6 w_d / (7 − w_d)`, RECENTRE per weekday: divide by
   the weekday's mean ratio `ū_d`, so `ẽ_t = ū_d e_t` and `ũ_t = y_t / ẽ_t` average exactly 1. A
   trend biases the baseline by a factor common to a weekday's days. Under growth this is about
   `(7/3)(r² + r')`, which normalising the weights does not remove, and without recentring it would
   add its square to `s_d²` (about 5.5e-4 at a sustained `r = 0.1`). Estimating `ū_d` uses one degree
   of freedom, hence `n_d − 1`:
   `s_d² = max(0, Σ(ũ − 1)²/(n_d − 1) − mean(1/ẽ + 1/(6 b)) − (1 + 1/6)/φ)`.
   This removes the Poisson and NB variance of `y_t` (which the likelihood already carries) and the
   sampling noise of `b_t`, so nothing is counted twice. Two biases remain, both upward
   (conservative): the other weekdays' `s²` enters through `b_t` scaled by about 1/36, and growth
   that varies through the window (a wave) adds about `A_d² Var(r_t) + (7/3)² Var(r_t² + r_t')`,
   where `A_d` is the weekday asymmetry. That is at most ~2e-4 at 14-day doubling.

If any weekday has fewer than `min_weeks` usable days, or no baseline mass, it warns and returns no
effect (weights 1, extra variance 0).

The full derivation is in `docs/models/configurable-epi/day-of-week-effect.md`.
"""
function estimate_day_of_week_effects(
        dates::AbstractVector{Date}, counts::AbstractVector{<:Real};
        phi::Real, window_days::Integer = 182, exclude_recent_days::Integer = 14,
        min_weeks::Integer = 4,
    )
    length(dates) == length(counts) || throw(ArgumentError("dates and counts differ in length"))
    all(diff(dates) .== Day(1)) || throw(
        ArgumentError("day-of-week estimation needs a gap-free, ascending daily series")
    )
    last_kept = length(counts) - exclude_recent_days
    first_kept = max(1, last_kept - window_days + 1)
    if last_kept - first_kept + 1 < 7 * min_weeks + 6
        @warn "day-of-week: too little history for the estimator; using no effect" n = length(counts) exclude_recent_days
        return _NO_DOW_EFFECTS
    end
    y = Float64.(counts[first_kept:last_kept])
    days = dates[first_kept:last_kept]
    n = length(y)
    centres = 4:(n - 3)

    sum_y = zeros(7)
    sum_b = zeros(7)
    used = zeros(Int, 7)
    baseline = fill(NaN, n)
    for i in centres
        baseline[i] = (sum(@view y[(i - 3):(i + 3)]) - y[i]) / 6
        d = dayofweek(days[i])
        sum_y[d] += y[i]
        sum_b[d] += baseline[i]
        used[d] += 1
    end
    # A weekday with no counts would get weight 0: a zero observation mean on that weekday, and a
    # zero effective ascertainment for the initial-state inversion when it is the start weekday.
    if any(<(min_weeks), used) || any(<=(0.0), sum_b) || any(<=(0.0), sum_y)
        @warn "day-of-week: a weekday has too few days or no counts; using no effect" used sum_y
        return _NO_DOW_EFFECTS
    end
    q = sum_y ./ sum_b
    raw = 7 .* q ./ (6 .+ q)
    weights = raw ./ (sum(raw) / 7)

    # Expected count from the baseline, and each weekday's mean ratio: the recentring scale.
    expected = fill(NaN, n)
    ratio_sum = zeros(7)
    scored = zeros(Int, 7)
    for i in centres
        baseline[i] > 0 || continue
        d = dayofweek(days[i])
        expected[i] = baseline[i] * 6 * weights[d] / (7 - weights[d])
        expected[i] > 0 || continue
        ratio_sum[d] += y[i] / expected[i]
        scored[d] += 1
    end
    scale = ratio_sum ./ max.(scored, 1)

    sq = zeros(7)
    sampling = zeros(7)
    for i in centres
        expected[i] > 0 || continue   # also skips the NaN of an unscored day
        d = dayofweek(days[i])
        centred = scale[d] * expected[i]
        sq[d] += (y[i] / centred - 1)^2
        sampling[d] += 1 / centred + 1 / (6 * baseline[i])
    end
    extra_var = ntuple(
        d -> scored[d] < 2 || scale[d] <= 0 ? 0.0 :
            max(0.0, sq[d] / (scored[d] - 1) - sampling[d] / scored[d] - (1 + 1 / 6) / phi),
        7,
    )
    return (
        weights = Tuple(weights)::NTuple{7, Float64},
        extra_var,
        fallback = false,
        n_used = sum(scored),
    )
end

"""
    prepare_day_of_week_history(dates, counts, report_date)

Validate one reconstructed daily as-of series and return the exact history supplied to the plugin
estimator. The series must end on `report_date - 1` and be gap-free. The final assimilated
observation is deliberately omitted, matching the model's initial-state history contract.
"""
function prepare_day_of_week_history(
        dates::AbstractVector{Date}, counts::AbstractVector{<:Real}, report_date::Date,
    )
    length(dates) == length(counts) || throw(ArgumentError("dates and counts differ in length"))
    isempty(dates) && throw(ArgumentError("day-of-week history is empty for $report_date"))
    expected_last = report_date - Day(1)
    last(dates) == expected_last || throw(
        ArgumentError(
            "day-of-week as-of series for $report_date must end on $expected_last; " *
                "got $(last(dates))"
        )
    )
    expected = first(dates):Day(1):expected_last
    dates == collect(expected) || throw(
        ArgumentError(
            "day-of-week as-of series for $report_date is not gap-free through $expected_last: " *
                "expected $(length(expected)) consecutive days from $(first(dates)), " *
                "got $(length(dates))"
        )
    )
    length(dates) >= 2 || throw(
        ArgumentError("day-of-week history needs at least two observations for $report_date")
    )
    return (dates = dates[1:(end - 1)], counts = counts[1:(end - 1)])
end

"""
    build_day_of_week_observation(cfg, inner, start_date, history; phi, precomputed = nothing)
        -> (; modifier, dispersion, effects, derived_hyperparameters)

The observation pieces for weekday effect `cfg`:

- `modifier`: the `mean_modifier`. It is `inner` itself (the ascertainment path) under
  `NoDayOfWeekConfig`, which keeps the step-1 model bit-exact; otherwise a `DayOfWeekModifier`
  wrapping it.
- `dispersion`: the NB `phi` function. It is `HyperPhi()` unless the plugin widens it per weekday.
- `effects`: the plugin's history estimates to report; `nothing` otherwise.
- `derived_hyperparameters`: the learned multipliers to summarise; `nothing` unless learned.

`history` is the submodel context's `(; times, counts)` on the model clock. Only the plugin reads it.
`precomputed` supplies a validated first-vintage estimate for reuse; it is rejected for the other
modes so orchestration cannot silently attach an effect to the wrong configuration.
"""
function build_day_of_week_observation(
        ::NoDayOfWeekConfig, inner, start_date::Date, history;
        phi::Real, precomputed = nothing,
    )
    precomputed === nothing || error("precomputed weekday effects require plugin mode")
    return (
        modifier = inner, dispersion = HyperPhi(), effects = nothing,
        derived_hyperparameters = nothing,
    )
end

function build_day_of_week_observation(
        cfg::PluginDayOfWeekConfig, inner, start_date::Date, history;
        phi::Real, precomputed = nothing,
    )
    estimated = precomputed === nothing ?
        _estimate_day_of_week(cfg, history, start_date; phi) : precomputed
    estimated.fallback && cfg.fit_policy == "first_vintage" && error(
        "first-vintage day-of-week estimation fell back to no effect"
    )
    dow0 = dayofweek(start_date)
    return (
        modifier = DayOfWeekModifier(inner, dow0, estimated.weights),
        dispersion = DayOfWeekDispersion(dow0, estimated.extra_var),
        effects = estimated,
        derived_hyperparameters = nothing,
    )
end

# The weights are read from each particle's θ (`weights = nothing`), and their spread across the
# ensemble IS the uncertainty, so the dispersion stays `phi` (see the header).
function build_day_of_week_observation(
        ::LearnedDayOfWeekConfig, inner, start_date::Date, history;
        phi::Real, precomputed = nothing,
    )
    precomputed === nothing || error("precomputed weekday effects require plugin mode")
    return (
        modifier = DayOfWeekModifier(inner, dayofweek(start_date), nothing),
        dispersion = HyperPhi(),
        effects = nothing,
        derived_hyperparameters = DayOfWeekMultiplierReport(),
    )
end

function _estimate_day_of_week(::PluginDayOfWeekConfig, ::Nothing, start_date; phi)
    @warn "day-of-week: no history in the model context; the plugin uses no effect"
    return _NO_DOW_EFFECTS
end

_estimate_day_of_week(cfg::PluginDayOfWeekConfig, history, start_date; phi) =
    estimate_day_of_week_effects(
    start_date .+ Day.(round.(Int, history.times)), history.counts;
    phi, window_days = cfg.window_days, exclude_recent_days = cfg.exclude_recent_days,
)

const _DOW_MULTIPLIER_NAMES = Tuple(Symbol("dow_multiplier_", day) for day in DOW_DAY_ABBREVIATIONS)

# Helmert coordinates mean nothing one at a time, so the summary reports the multipliers they imply.
struct DayOfWeekMultiplierReport end
(::DayOfWeekMultiplierReport)(hyper) = NamedTuple{_DOW_MULTIPLIER_NAMES}(day_of_week_multipliers(hyper))

"""
    day_of_week_report_rows(effects) -> Tuple of (parameter, statistic, value)

Hyperparameter-table rows for the plugin's history estimates: `dow_multiplier_<day>` and
`dow_extra_var_<day>`, with statistic `"plugin"`. Learned multipliers are reported by the Liu–West
summary instead, under the same names but with quantile statistics.
"""
day_of_week_report_rows(::Nothing) = ()
day_of_week_report_rows(effects::NamedTuple) = (
    (
        (String(name), "plugin", effects.weights[d])
            for (d, name) in enumerate(_DOW_MULTIPLIER_NAMES)
    )...,
    (
        ("dow_extra_var_$day", "plugin", effects.extra_var[d])
            for (d, day) in enumerate(DOW_DAY_ABBREVIATIONS)
    )...,
)
