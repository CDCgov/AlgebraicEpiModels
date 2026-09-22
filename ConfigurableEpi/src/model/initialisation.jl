# Initial state: the RK4 stability guard on the durations, inversion of one observed count into
# compartment occupancies, and the peak-anchored susceptible fraction.

"Classical RK4 is stable on a real negative eigenvalue only while `|lambda * h| < 2.785`."
const RK4_STABILITY_LIMIT = 2.785
"Proximity warning threshold, about 1.4x below the stability limit."
const RK4_WARN_LAMBDA_H = 2.0

"""
    max_transition_rate(dur::Durations, n_E, n_I; R_eff_max = 1.0) -> Float64

The fastest model rate in 1/day: the Erlang stage rates, observation progression, waning, and
the faster eigenvalue of the linearised `(E, I)` block at `R_eff_max`.
"""
function max_transition_rate(dur::Durations, n_E::Integer, n_I::Integer; R_eff_max::Real = 1.0)
    R_eff_max >= 0 || throw(ArgumentError("R_eff_max must be non-negative, got $R_eff_max"))
    sigma, gamma = n_E / dur.latent, n_I / dur.infectious
    half_sum = (sigma + gamma) / 2
    transmission = half_sum + sqrt(half_sum^2 + sigma * gamma * (R_eff_max - 1))
    return maximum((sigma, gamma, 1 / dur.obs_progression, 1 / dur.immunity, transmission))
end

"""
    required_supersample(dur, n_E, n_I, dt; R_eff_max = 1.0, target = RK4_WARN_LAMBDA_H) -> Int

The smallest `supersample` whose substep keeps `|lambda * h| <= target`.
"""
function required_supersample(
        dur::Durations, n_E::Integer, n_I::Integer, dt::Real; R_eff_max::Real = 1.0, target::Real = RK4_WARN_LAMBDA_H,
    )
    target > 0 || throw(ArgumentError("target must be positive, got $target"))
    return max(1, ceil(Int, max_transition_rate(dur, n_E, n_I; R_eff_max) * dt / target))
end

"""
    assert_integration_stable(dur, n_E, n_I, dt, supersample; R_eff_max = 1.0) -> lambda_h

Error beyond [`RK4_STABILITY_LIMIT`](@ref) and warn beyond [`RK4_WARN_LAMBDA_H`](@ref). Size
`R_eff_max` from the priors' upper tail ([`prior_R_eff_bound`](@ref)), not the endemic seed.
"""
function assert_integration_stable(
        dur::Durations, n_E::Integer, n_I::Integer, dt::Real, supersample::Integer; R_eff_max::Real = 1.0,
    )
    supersample >= 1 || throw(ArgumentError("supersample must be >= 1, got $supersample"))
    rate = max_transition_rate(dur, n_E, n_I; R_eff_max)
    lambda_h = rate * dt / supersample
    if lambda_h >= RK4_STABILITY_LIMIT
        needed = max(supersample + 1, required_supersample(dur, n_E, n_I, dt; R_eff_max, target = RK4_STABILITY_LIMIT))
        throw(
            ArgumentError(
                "ODE substep is unstable: the fastest rate is $(round(rate, sigdigits = 3))/day at " *
                    "R_eff <= $(round(R_eff_max, sigdigits = 3)) and |lambda*h| = $(round(lambda_h, sigdigits = 3)) " *
                    "exceeds RK4's limit $RK4_STABILITY_LIMIT. Raise `supersample` to at least $needed " *
                    "(currently $supersample) or lengthen the shortest duration.",
            )
        )
    elseif lambda_h > RK4_WARN_LAMBDA_H
        @warn "ODE substep is within 1.4x of RK4's stability limit; raise `supersample` for headroom" lambda_h supersample suggested_supersample =
            required_supersample(dur, n_E, n_I, dt; R_eff_max) R_eff_max
    end
    return lambda_h
end

"""
    initial_infection_state(y0, dur::Durations, ascertainment, accumulation_window_days)
        -> (; daily_incidence, exposed, infectious, obs_stage)

Invert one partially ascertained count into quasi-steady compartment occupancies:
`daily_incidence = y0 / (ascertainment * accumulation_window_days)` and each upstream compartment
holds `daily_incidence * mean_duration`.
"""
function initial_infection_state(y0, dur::Durations, ascertainment, accumulation_window_days)
    isfinite(ascertainment) && ascertainment > 0 ||
        throw(ArgumentError("ascertainment must be finite and > 0 to invert an observed count; got $ascertainment"))
    isfinite(accumulation_window_days) && accumulation_window_days > 0 ||
        throw(ArgumentError("accumulation_window_days must be finite and > 0; got $accumulation_window_days"))
    daily_incidence = y0 / (ascertainment * accumulation_window_days)
    return (;
        daily_incidence, exposed = daily_incidence * dur.latent,
        infectious = daily_incidence * dur.infectious, obs_stage = daily_incidence * dur.obs_progression,
    )
end

"""
    reporting_delay_days(dur::Durations, n_obs_stages) -> Float64

Mean infection-to-report delay, `(n_obs_stages - 1) * obs_progression`.
"""
reporting_delay_days(dur::Durations, n_obs_stages::Integer) = (n_obs_stages - 1) * dur.obs_progression

"""
    prevalence_peak_lag_days(dur::Durations, n_E = 1, n_I = 1) -> Float64

How long after the incidence peak total infected prevalence peaks: the normalised mean
`E[T²] / (2 E[T])` of the Erlang residence kernel.
"""
function prevalence_peak_lag_days(dur::Durations, n_E::Integer = 1, n_I::Integer = 1)
    n_E > 0 && n_I > 0 || throw(ArgumentError("stage counts must be positive, got n_E = $n_E, n_I = $n_I"))
    total = dur.latent + dur.infectious
    return (total^2 + dur.latent^2 / n_E + dur.infectious^2 / n_I) / (2 * total)
end

"Whether `R_eff = 1` holds exactly at the peak of `E + I`: only for a single `I` stage."
anchor_is_exact(n_E::Integer, n_I::Integer) = n_I == 1

# Centred rolling geometric mean, so smoothing a multiplicative series keeps a wave's peak in place.
function _smooth_log(values::AbstractVector{<:Real}, window::Integer = 5)
    n, half = length(values), window ÷ 2
    return [
        exp(sum(log(max(float(values[j]), 1.0e-4)) for j in max(1, i - half):min(n, i + half)) / (min(n, i + half) - max(1, i - half) + 1))
            for i in 1:n
    ]
end

"""
    find_observed_peak(counts; window = 5, min_prominence = 1.1) -> Union{Int, Nothing}

Index of the observed wave peak: an interior, strict maximum of the log-smoothed series at least
`min_prominence` above the higher flank. `nothing` when the history contains no identified peak.
"""
function find_observed_peak(counts::AbstractVector{<:Real}; window::Integer = 5, min_prominence::Real = 1.1)
    length(counts) >= 3 * window || return nothing
    smoothed = _smooth_log(counts, window)
    peak = argmax(smoothed)
    (peak <= window || peak > length(smoothed) - window) && return nothing
    smoothed[peak] > smoothed[peak - 1] && smoothed[peak] > smoothed[peak + 1] || return nothing
    flank = max(minimum(smoothed[1:(peak - 1)]), minimum(smoothed[(peak + 1):end]))
    return smoothed[peak] >= min_prominence * flank ? peak : nothing
end

# Linear interpolation held flat outside the data range.
function _interp_flat(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, x::Real)
    x <= first(xs) && return float(first(ys))
    x >= last(xs) && return float(last(ys))
    j = searchsortedfirst(xs, x)
    x == xs[j] && return float(ys[j])
    w = (x - xs[j - 1]) / (xs[j] - xs[j - 1])
    return float(ys[j - 1]) * (1 - w) + float(ys[j]) * w
end

"""
    carry_susceptible(s_anchor, t_anchor, t_target, times, daily_incidence, omega) -> Float64

Integrate `ds/dt = omega (1 - s) - i(t)` from the anchor to the target time (midpoint RK2, steps
of at most one day), holding the per-capita incidence flat outside its range.
"""
function carry_susceptible(
        s_anchor::Real, t_anchor::Real, t_target::Real, times::AbstractVector{<:Real},
        daily_incidence::AbstractVector{<:Real}, omega::Real,
    )
    isempty(times) && return float(s_anchor)
    delta = float(t_target - t_anchor)
    iszero(delta) && return float(s_anchor)
    n = ceil(Int, abs(delta))
    step = delta / n
    s, t = float(s_anchor), float(t_anchor)
    interp(τ) = _interp_flat(times, daily_incidence, τ)
    for _ in 1:n
        k1 = omega * (1 - s) - interp(t)
        k2 = omega * (1 - (s + 0.5 * step * k1)) - interp(t + 0.5 * step)
        s += step * k2
        t += step
    end
    return clamp(s, 0.0, 1.0)
end

@inline _ascertainment_at(a::Real, t) = a
@inline _ascertainment_at(f, t) = f(t)

"""
    peak_anchored_susceptible_fraction(history, dur, n_obs_stages, ascertainment, pop, dt, chi_at, R0;
                                       n_E = 1, n_I = 1) -> Union{NamedTuple, Nothing}

Find the observed peak in `history = (; times, counts)` (model clock, bin-end labels), shift it
back by half a bin and the reporting delay and forward by the prevalence lag, evaluate
`S/N = 1 / (R0 * chi_at(t_anchor))` there (`R_eff = 1` at the prevalence peak, `Rt = 1` by
construction), and carry `S/N` back to `t = 0` against the reconstructed incidence.
`ascertainment` is a constant or `t -> observations per infection`. Returns `nothing` when no
peak is identified or the anchor is inexact (`n_I > 1`).
"""
function peak_anchored_susceptible_fraction(
        history, dur::Durations, n_obs_stages::Integer, ascertainment, pop::Real, dt::Real, chi_at, R0::Real;
        n_E::Integer = 1, n_I::Integer = 1,
    )
    anchor_is_exact(n_E, n_I) || return nothing
    peak = find_observed_peak(history.counts)
    peak === nothing && return nothing
    delay = reporting_delay_days(dur, n_obs_stages)
    lag = prevalence_peak_lag_days(dur, n_E, n_I)
    t_anchor = history.times[peak] - dt / 2 - delay + lag
    chi = chi_at(t_anchor)
    isfinite(chi) && chi > 0 || return nothing
    s_anchor = clamp(1 / (R0 * chi), 0.0, 1.0)
    incidence_times = history.times .- dt / 2 .- delay
    incidence = [
        max(float(c), 0.0) / (_ascertainment_at(ascertainment, incidence_times[i]) * dt * pop)
            for (i, c) in enumerate(history.counts)
    ]
    s0 = carry_susceptible(s_anchor, t_anchor, 0.0, incidence_times, incidence, 1 / dur.immunity)
    return (; susceptible_fraction = s0, anchor_time = t_anchor, anchor_fraction = s_anchor, peak_index = peak, delay_days = delay, lag_days = lag)
end
