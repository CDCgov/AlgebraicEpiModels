# ============================================================================
# Peak-anchored initialisation
# ============================================================================
#
# THE PROBLEM. One partially-ascertained time series badly under-specifies a compartmental state.
# The susceptible fraction in particular is never observed, so it has to come from somewhere, and
# the somewhere used to be an assertion: `S/N = 1/R0` at whatever calendar date the run happened
# to start. That is the endemic equilibrium, which is true on average over many years and false at
# any particular moment — measured, free-running from that seed overshot the observed level by 26x,
# and nothing downstream could correct it (`P0` gives `S` a +/-2.5pp spread, and a 12-week burn-in
# against a 26-week waning time is 0.47 e-foldings).
#
# THE FIX. There is one moment in each wave at which `S` IS identified, and identified by
# mechanism rather than by assumption:
#
#     d(E+I)/dt = beta*S*I/N - gamma*I = 0   <=>   beta*S/(gamma*N) = 1   <=>   R_eff = 1
#
# so at the peak of TOTAL INFECTED PREVALENCE, `S/N = 1/(R0 * Rt * chi)` exactly. Note where that
# is and is not true:
#
#   - It is the peak of `E + I`, NOT of incidence and NOT of `I` alone. Incidence peaks earlier
#     (at the incidence peak `sigma*E = gamma*I + beta*I^2/N > gamma*I`, so `I` is still rising).
#   - It survives time-varying transmission — the derivation uses only the instantaneous rate — so
#     seasonality and the AR(1) do not spoil it. `chi` therefore BELONGS in the seed here, which is
#     the opposite of the right answer at an arbitrary date.
#   - It is exact only with `n_I = 1` (for any positive `n_E`). With more `I` stages the peak of
#     `E+I` is where incidence equals the TERMINAL stage's removal flux, not `gamma*(E+I)`, and it
#     drifts.
#     `anchor_is_exact` reports this rather than letting it pass silently.
#
# GETTING THERE FROM WHAT WE OBSERVE. The observed signal is infection incidence delayed by the
# reporting chain. Two peak shifts take us from it to the prevalence peak, and both use the same
# argument: convolving with a kernel moves a peak by that kernel's MEAN, because the phase bias
# vanishes where the first derivative does. Only shape terms survive, and they are second order.
#
#     t_anchor = t_week_end_peak - weekly_bin/2 - reporting_delay + prevalence_peak_lag
#
# NO LEAKAGE. The anchor is derived once per run, from data inside the burn-in window, all of which
# precedes every scored origin. It is a retrospective seed, not a peek at the future.

"""
    reporting_delay_days(dur::Durations, n_obs_stages) -> Float64

Mean infection-to-report delay, `(n_obs_stages - 1) * obs_progression`.

Only `n_obs_stages - 1` stages contribute: the terminal one is the reset accumulator, which
registers a count on arrival and accrues no residence time. Under the event tap this is the WHOLE
delay — `latent` and `infectious` contribute nothing, which is what freed them to take literature
values.
"""
reporting_delay_days(dur::Durations, n_obs_stages::Integer) =
    (n_obs_stages - 1) * dur.obs_progression

"""
    prevalence_peak_lag_days(dur::Durations, n_E = 1, n_I = 1) -> Float64

How long after the incidence peak total infected prevalence peaks.

Prevalence is incidence convolved with `Q(a) = P(still in E or I at infection-age a)`, so the peak
moves by that kernel's normalised mean, `E[T^2] / (2 E[T])`. For Erlang latent and infectious
periods with `n_E` and `n_I` stages,

    E[T^2] = (T_E + T_I)^2 + T_E^2/n_E + T_I^2/n_I.

This equals the generation interval exactly for SIR (`T_E = 0`) and `0.75 x GI` when
`T_E == T_I` with one stage each — so "one generation interval" is the right shape but up to 25%
long for the shipped single-stage model.
"""
function prevalence_peak_lag_days(
        dur::Durations, n_E::Integer = 1, n_I::Integer = 1
    )
    n_E > 0 || throw(ArgumentError("n_E must be positive, got $n_E"))
    n_I > 0 || throw(ArgumentError("n_I must be positive, got $n_I"))
    total = dur.latent + dur.infectious
    variance = dur.latent^2 / n_E + dur.infectious^2 / n_I
    return (total^2 + variance) / (2 * total)
end

"""
    anchor_is_exact(n_E, n_I) -> Bool

Whether `R_eff = 1` holds exactly at the peak of `E + I` for this stage configuration. True only
for a single `I` stage; see the header.
"""
anchor_is_exact(n_E::Integer, n_I::Integer) = n_I == 1

# Centred rolling geometric mean. The series is multiplicative and strictly positive, so smoothing
# on the log scale is what keeps a wave's peak where it actually is rather than dragging it toward
# the larger flank.
function _smooth_log(values::AbstractVector{<:Real}, window::Integer = 5)
    n = length(values)
    out = similar(values, Float64)
    half = window ÷ 2
    for i in 1:n
        lo, hi = max(1, i - half), min(n, i + half)
        out[i] = exp(sum(log(max(float(values[j]), 1.0e-4)) for j in lo:hi) / (hi - lo + 1))
    end
    return out
end

"""
    find_observed_peak(counts; window = 5, min_prominence = 1.1) -> Union{Int, Nothing}

Index of the observed wave peak, or `nothing` when the history does not contain one.

Returns `nothing` rather than a best guess when the maximum sits at either end of the history or is
not strict. A series that is still rising, still falling, or has an ambiguous flat top has no
identified peak, and seeding from it would assert `R_eff = 1` at a moment we have no reason to
think it holds. The caller falls back to its non-peak seed and says so.
"""
function find_observed_peak(
        counts::AbstractVector{<:Real}; window::Integer = 5, min_prominence::Real = 1.1
    )
    length(counts) >= 3 * window || return nothing
    smoothed = _smooth_log(counts, window)
    peak = argmax(smoothed)
    # An interior, strict maximum. `argmax` returns the first point of a plateau, so checking both
    # neighbours is what makes an ambiguous flat top take the documented fallback.
    (peak <= window || peak > length(smoothed) - window) && return nothing
    (smoothed[peak] > smoothed[peak - 1] && smoothed[peak] > smoothed[peak + 1]) ||
        return nothing
    flank = max(minimum(smoothed[1:(peak - 1)]), minimum(smoothed[(peak + 1):end]))
    smoothed[peak] >= min_prominence * flank || return nothing
    return peak
end

"""
    carry_susceptible(s_anchor, t_anchor, t_target, times, daily_incidence, omega) -> Float64

Integrate `ds/dt = omega (1 - s) - i(t)` from the anchor to the model start, in either direction.

`times` and `daily_incidence` are the reconstructed per-capita infection incidence on the model
clock; values outside their range are held flat (persistence), which is the honest treatment for
the ~1 week at the end of the window where the reporting delay means no observation has landed yet.

Midpoint RK2 with steps no longer than one day — the integrand is a smooth weekly-resolution
reconstruction, so the step size is set by readability rather than by accuracy. The final step is
chosen to land exactly on `t_target`.
"""
function carry_susceptible(
        s_anchor::Real, t_anchor::Real, t_target::Real,
        times::AbstractVector{<:Real}, daily_incidence::AbstractVector{<:Real}, omega::Real
    )
    isempty(times) && return float(s_anchor)
    delta = float(t_target - t_anchor)
    iszero(delta) && return float(s_anchor)
    n = ceil(Int, abs(delta))
    step = delta / n
    s = float(s_anchor)
    t = float(t_anchor)
    interp(τ) = _interp_flat(times, daily_incidence, τ)
    for _ in 1:n
        k1 = omega * (1 - s) - interp(t)
        k2 = omega * (1 - (s + 0.5 * step * k1)) - interp(t + 0.5 * step)
        s += step * k2
        t += step
    end
    return clamp(s, 0.0, 1.0)
end

# The anchor's ascertainment is either a constant `Real` or a callable `t -> Real` giving
# observations per infection at model-clock day `t` (the same contract as `chi_at`), evaluated at
# each reconstructed infection time. A constant reproduces the historical division exactly.
@inline _ascertainment_at(a::Real, t) = a
@inline _ascertainment_at(f, t) = f(t)

# Linear interpolation, held flat outside the data range.
function _interp_flat(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, x::Real)
    x <= first(xs) && return float(first(ys))
    x >= last(xs) && return float(last(ys))
    j = searchsortedfirst(xs, x)
    x == xs[j] && return float(ys[j])
    w = (x - xs[j - 1]) / (xs[j] - xs[j - 1])
    return float(ys[j - 1]) * (1 - w) + float(ys[j]) * w
end

"""
    peak_anchored_susceptible_fraction(history, dur, n_obs_stages, ascertainment, pop,
                                       weekly_dt, chi_at, R0; n_E = 1, n_I = 1)
        -> Union{NamedTuple, Nothing}

The whole chain: find the observed peak, shift it to the prevalence peak, evaluate `R_eff = 1`
there, and carry `S/N` back to the model start (`t = 0` on the model clock).

`history` is `(; times, counts)` on the model clock in days — the burn-in window and nothing later.
Its times are week-ending labels for accumulated counts, so both the peak and reconstructed
incidence are shifted back by `weekly_dt / 2` to the bin midpoint. `chi_at(t)` returns the seasonal
multiplier at model time `t`, so the caller keeps ownership of the forcing (and of `Rt`, which is 1
at the seed by construction). `ascertainment` is likewise either a constant `Real` or a callable
`t -> Real` (observations per infection at model-clock day `t`, e.g. `t -> path(hyper, t)` for an
`AscertainmentPath`), evaluated at each reconstructed infection time — so a declining path reads
more infections per count later in the history.

Returns `nothing` when the window contains no identified peak, which is a signal to fall back to
the endemic seed, not an error. It also returns `nothing` when `n_I > 1`, where `R_eff = 1` is not
exact at the total-prevalence peak.
"""
function peak_anchored_susceptible_fraction(
        history, dur::Durations, n_obs_stages::Integer, ascertainment, pop::Real,
        weekly_dt::Real, chi_at, R0::Real; n_E::Integer = 1, n_I::Integer = 1
    )
    anchor_is_exact(n_E, n_I) || return nothing
    peak = find_observed_peak(history.counts)
    peak === nothing && return nothing

    delay = reporting_delay_days(dur, n_obs_stages)
    lag = prevalence_peak_lag_days(dur, n_E, n_I)
    bin_offset = weekly_dt / 2
    t_anchor = history.times[peak] - bin_offset - delay + lag

    # `R_eff = 1` at the prevalence peak. `Rt` is 1 at the seed by construction (the AR(1) is
    # centred there), so it drops out; `chi` does NOT, and that is the point.
    chi = chi_at(t_anchor)
    (isfinite(chi) && chi > 0) || return nothing
    s_anchor = clamp(1 / (R0 * chi), 0.0, 1.0)

    # Reconstructed per-capita daily infection incidence, dated when the infections HAPPENED —
    # and divided by the ascertainment in force WHEN THEY HAPPENED.
    incidence_times = history.times .- bin_offset .- delay
    incidence = [
        max(float(c), 0.0) /
            (_ascertainment_at(ascertainment, incidence_times[i]) * weekly_dt * pop)
            for (i, c) in enumerate(history.counts)
    ]
    s0 = carry_susceptible(
        s_anchor, t_anchor, 0.0, incidence_times, incidence, 1 / dur.immunity
    )
    return (;
        susceptible_fraction = s0, anchor_time = t_anchor, anchor_fraction = s_anchor,
        peak_index = peak, delay_days = delay, lag_days = lag,
    )
end
