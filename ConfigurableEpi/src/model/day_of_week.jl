# Weekday observation effect for daily counts: `y_t ~ NB(alpha(t) * w_{d(t)} * incidence, phi_{d(t)})`,
# with weights of arithmetic weekly mean 1. The plugin effect fixes point estimates and carries
# their uncertainty as per-weekday extra dispersion; the learned effect carries six zero-sum
# Helmert coordinates in the particle cloud and adds no extra dispersion.

const DOW_DAY_ABBREVIATIONS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
const DOW_LEARNED_NAMES = (:dow_z1, :dow_z2, :dow_z3, :dow_z4, :dow_z5, :dow_z6)

"Prior sd of each Helmert coordinate that gives every centred weekday log-effect marginal sd `sd`."
dow_helmert_prior_sd(sd::Real) = sd * sqrt(7 / 6)

# Orthonormal zero-sum basis, rows Monday … Sunday (closed form, so reproducible across BLAS builds).
const DOW_HELMERT = ntuple(
    j -> ntuple(k -> j <= k ? 1 / sqrt(k * (k + 1)) : j == k + 1 ? -k / sqrt(k * (k + 1)) : 0.0, 6), 7
)

"""
    validate_day_of_week(cfg::DayOfWeekConfig) -> cfg

The plugin window must hold at least five weeks; other modes have nothing to check.
"""
validate_day_of_week(cfg::Union{NoDayOfWeekConfig, LearnedDayOfWeekConfig}) = cfg
function validate_day_of_week(cfg::PluginDayOfWeekConfig)
    cfg.window_days >= 35 || error("day_of_week window_days must be >= 35 (five weeks); got $(cfg.window_days)")
    cfg.exclude_recent_days >= 0 || error("day_of_week exclude_recent_days must be >= 0; got $(cfg.exclude_recent_days)")
    cfg.fit_policy in ("per_origin", "first_vintage") ||
        error("day_of_week fit_policy must be per_origin or first_vintage; got $(cfg.fit_policy)")
    return cfg
end

"The θ names this effect adds to the learned set: the six Helmert coordinates under `learned`."
day_of_week_learned_names(::DayOfWeekConfig) = ()
day_of_week_learned_names(::LearnedDayOfWeekConfig) = DOW_LEARNED_NAMES

"""
    assert_day_of_week_learnable(cfg::DayOfWeekConfig, learn_params)

The weekday coordinates in an explicit learned set must be exactly `day_of_week_learned_names(cfg)`.
"""
function assert_day_of_week_learnable(cfg::DayOfWeekConfig, learn_params)
    expected = day_of_week_learned_names(cfg)
    present = Tuple(n for n in DOW_LEARNED_NAMES if n in learn_params)
    present == expected || error(
        "learn_params weekday coordinates $present do not match the $(nameof(typeof(cfg))) effect, " *
            "which learns $expected; select [day_of_week.learned] to learn all six, or remove them",
    )
    return nothing
end

# Calendar weekday (Monday = 1) of integer model time `t`, given the weekday of `t = 0`.
@inline _weekday(dow0::Int, t) = mod1(dow0 + round(Int, t), 7)

"""
    DayOfWeekModifier(inner, dow0, weights)

An observation `mean_modifier` equal to `inner(hyper, t) * w_{d(t)}`, callable as `(hyper, t)` and
`(latent, hyper, t)`. `weights::NTuple{7}` is the plugin effect; `weights = nothing` reads the
learned multipliers from `hyper.dow_z1 … dow_z6`.
"""
struct DayOfWeekModifier{A, W}
    inner::A
    dow0::Int
    weights::W
end

@inline function dow_log_effects(hyper)
    z = promote(hyper.dow_z1, hyper.dow_z2, hyper.dow_z3, hyper.dow_z4, hyper.dow_z5, hyper.dow_z6)
    return ntuple(j -> sum(DOW_HELMERT[j][k] * z[k] for k in 1:6), 7)
end

"""
    day_of_week_multipliers(hyper) -> NTuple{7}

The learned weekday multipliers Monday … Sunday, `7 * softmax(H z)` (shifted by the maximum so a
large coordinate cannot overflow).
"""
@inline function day_of_week_multipliers(hyper)
    e = dow_log_effects(hyper)
    top = maximum(e)
    total = sum(x -> exp(x - top), e)
    return ntuple(d -> 7 * exp(e[d] - top) / total, 7)
end

@inline _dow_weight(w::NTuple{7, Float64}, hyper, d) = w[d]
@inline _dow_weight(::Nothing, hyper, d) = day_of_week_multipliers(hyper)[d]

"""
    day_of_week_weight(modifier, hyper, t)

The weekday factor alone at observation time `t`; `true` for a modifier without one.
"""
@inline day_of_week_weight(modifier, hyper, t) = true
@inline day_of_week_weight(m::DayOfWeekModifier, hyper, t) = _dow_weight(m.weights, hyper, _weekday(m.dow0, t))

@inline (m::DayOfWeekModifier)(hyper, t) = m.inner(hyper, t) * day_of_week_weight(m, hyper, t)
@inline (m::DayOfWeekModifier)(latent, hyper, t) = m.inner(latent, hyper, t) * day_of_week_weight(m, hyper, t)

"""
    remove_day_of_week_effect(history, modifier, hyper) -> history

Divide each historical count by the weekday weight of its own observation date so the
initial-state reconstruction inverts with ascertainment alone.
"""
remove_day_of_week_effect(history, modifier, hyper) = history
remove_day_of_week_effect(::Nothing, ::DayOfWeekModifier, hyper) = nothing
remove_day_of_week_effect(history, m::DayOfWeekModifier, hyper) = merge(
    history,
    (; counts = [c / day_of_week_weight(m, hyper, t) for (t, c) in zip(history.times, history.counts)]),
)

"""
    HyperPhi()                            # (latent, hyper, t) -> hyper.phi
    DayOfWeekDispersion(dow0, extra_var)  # (latent, hyper, t) -> 1 / (1/hyper.phi + extra_var[d(t)])

The negative-binomial dispersion without and with the plugin's per-weekday widening.
"""
struct HyperPhi end
@inline (::HyperPhi)(latent, hyper, t) = hyper.phi

struct DayOfWeekDispersion
    dow0::Int
    extra_var::NTuple{7, Float64}
end
@inline (p::DayOfWeekDispersion)(latent, hyper, t) = inv(inv(hyper.phi) + p.extra_var[_weekday(p.dow0, t)])

const _NO_DOW_EFFECTS = (weights = ntuple(_ -> 1.0, 7), extra_var = ntuple(_ -> 0.0, 7), fallback = true, n_used = 0)

"""
    estimate_day_of_week_effects(dates, counts; phi, window_days = 182, exclude_recent_days = 14,
                                 min_weeks = 4) -> (; weights, extra_var, fallback, n_used)

Multiplicative weekday decomposition of a gap-free daily series against a leave-one-out centred
weekly baseline: weights `w_d = 7 q_d / (6 + q_d)` from the ratio of sums `q_d = Σy / Σb`,
normalised to mean 1, and per-weekday extra variance `s_d²` after removing the Poisson/NB
sampling variance of the counts and of the baseline. Falls back to no effect (with a warning)
when any weekday has fewer than `min_weeks` usable days.
"""
function estimate_day_of_week_effects(
        dates::AbstractVector{Date}, counts::AbstractVector{<:Real};
        phi::Real, window_days::Integer = 182, exclude_recent_days::Integer = 14, min_weeks::Integer = 4,
    )
    length(dates) == length(counts) || throw(ArgumentError("dates and counts differ in length"))
    all(diff(dates) .== Day(1)) || throw(ArgumentError("day-of-week estimation needs a gap-free, ascending daily series"))
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

    sum_y, sum_b, used = zeros(7), zeros(7), zeros(Int, 7)
    baseline = fill(NaN, n)
    for i in centres
        baseline[i] = (sum(@view y[(i - 3):(i + 3)]) - y[i]) / 6
        d = dayofweek(days[i])
        sum_y[d] += y[i]
        sum_b[d] += baseline[i]
        used[d] += 1
    end
    if any(<(min_weeks), used) || any(<=(0.0), sum_b) || any(<=(0.0), sum_y)
        @warn "day-of-week: a weekday has too few days or no counts; using no effect" used sum_y
        return _NO_DOW_EFFECTS
    end
    q = sum_y ./ sum_b
    raw = 7 .* q ./ (6 .+ q)
    weights = raw ./ (sum(raw) / 7)

    expected = fill(NaN, n)
    ratio_sum, scored = zeros(7), zeros(Int, 7)
    for i in centres
        baseline[i] > 0 || continue
        d = dayofweek(days[i])
        expected[i] = baseline[i] * 6 * weights[d] / (7 - weights[d])
        expected[i] > 0 || continue
        ratio_sum[d] += y[i] / expected[i]
        scored[d] += 1
    end
    scale = ratio_sum ./ max.(scored, 1)

    sq, sampling = zeros(7), zeros(7)
    for i in centres
        expected[i] > 0 || continue
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
    return (weights = Tuple(weights)::NTuple{7, Float64}, extra_var, fallback = false, n_used = sum(scored))
end

"""
    prepare_day_of_week_history(dates, counts, report_date) -> (; dates, counts)

Check one daily as-of series ends on `report_date - 1` and is gap-free, and drop its final
observation (the model's history contract).
"""
function prepare_day_of_week_history(dates::AbstractVector{Date}, counts::AbstractVector{<:Real}, report_date::Date)
    length(dates) == length(counts) || throw(ArgumentError("dates and counts differ in length"))
    isempty(dates) && throw(ArgumentError("day-of-week history is empty for $report_date"))
    expected_last = report_date - Day(1)
    last(dates) == expected_last ||
        throw(ArgumentError("day-of-week as-of series for $report_date must end on $expected_last; got $(last(dates))"))
    dates == collect(first(dates):Day(1):expected_last) ||
        throw(ArgumentError("day-of-week as-of series for $report_date is not gap-free through $expected_last"))
    length(dates) >= 2 || throw(ArgumentError("day-of-week history needs at least two observations for $report_date"))
    return (dates = dates[1:(end - 1)], counts = counts[1:(end - 1)])
end

"""
    build_day_of_week_observation(cfg, inner, start_date, history; phi, precomputed = nothing)
        -> (; modifier, dispersion, effects, derived_hyperparameters)

The observation pieces for weekday effect `cfg`: the `mean_modifier` (`inner` itself under
`NoDayOfWeekConfig`), the NB dispersion function, the plugin's history estimates to report, and
the learned multipliers to summarise. `history` is `(; times, counts)` on the model clock;
`precomputed` reuses a first-vintage plugin estimate.
"""
function build_day_of_week_observation(::NoDayOfWeekConfig, inner, start_date::Date, history; phi::Real, precomputed = nothing)
    precomputed === nothing || error("precomputed weekday effects require plugin mode")
    return (modifier = inner, dispersion = HyperPhi(), effects = nothing, derived_hyperparameters = nothing)
end

function build_day_of_week_observation(
        cfg::PluginDayOfWeekConfig, inner, start_date::Date, history; phi::Real, precomputed = nothing,
    )
    estimated = precomputed === nothing ? _estimate_day_of_week(cfg, history, start_date; phi) : precomputed
    estimated.fallback && cfg.fit_policy == "first_vintage" &&
        error("first-vintage day-of-week estimation fell back to no effect")
    dow0 = dayofweek(start_date)
    return (
        modifier = DayOfWeekModifier(inner, dow0, estimated.weights),
        dispersion = DayOfWeekDispersion(dow0, estimated.extra_var),
        effects = estimated, derived_hyperparameters = nothing,
    )
end

function build_day_of_week_observation(::LearnedDayOfWeekConfig, inner, start_date::Date, history; phi::Real, precomputed = nothing)
    precomputed === nothing || error("precomputed weekday effects require plugin mode")
    return (
        modifier = DayOfWeekModifier(inner, dayofweek(start_date), nothing),
        dispersion = HyperPhi(), effects = nothing, derived_hyperparameters = DayOfWeekMultiplierReport(),
    )
end

function _estimate_day_of_week(::PluginDayOfWeekConfig, ::Nothing, start_date; phi)
    @warn "day-of-week: no history in the model context; the plugin uses no effect"
    return _NO_DOW_EFFECTS
end
_estimate_day_of_week(cfg::PluginDayOfWeekConfig, history, start_date; phi) = estimate_day_of_week_effects(
    start_date .+ Day.(round.(Int, history.times)), history.counts;
    phi, window_days = cfg.window_days, exclude_recent_days = cfg.exclude_recent_days,
)

const _DOW_MULTIPLIER_NAMES = Tuple(Symbol("dow_multiplier_", day) for day in DOW_DAY_ABBREVIATIONS)

"Summarises the learned Helmert coordinates as the multipliers they imply."
struct DayOfWeekMultiplierReport end
(::DayOfWeekMultiplierReport)(hyper) = NamedTuple{_DOW_MULTIPLIER_NAMES}(day_of_week_multipliers(hyper))

"""
    day_of_week_report_rows(effects) -> Tuple of (parameter, statistic, value)

Summary rows for the plugin's estimates: `dow_multiplier_<day>` and `dow_extra_var_<day>`.
"""
day_of_week_report_rows(::Nothing) = ()
day_of_week_report_rows(effects::NamedTuple) = (
    ((String(name), "plugin", effects.weights[d]) for (d, name) in enumerate(_DOW_MULTIPLIER_NAMES))...,
    (("dow_extra_var_$day", "plugin", effects.extra_var[d]) for (d, day) in enumerate(DOW_DAY_ABBREVIATIONS))...,
)
