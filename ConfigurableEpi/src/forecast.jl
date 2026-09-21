# ============================================================================
# BACKTEST — vintage-aware forecasting building blocks (filter-agnostic)
# ============================================================================
#
# Pure helpers for the expanding-window backtest run scripts (run_ukf_backtest.jl,
# run_pf_backtest.jl). No IO and no scoring — the Julia run scripts emit a
# forecast-quantile table and Python (epifusion) owns WIS/coverage + plotting.
#
# The forecast is a pure-predictive roll (NO `correct!`): from the filtered state
# at an origin we propagate forward `n_ahead` steps. Two modes, matched to the filter:
#   * UKF  → `forecast_states`: deepcopy the filter, seed at the filtered Gaussian
#            (x_T, P_T) and step with `predict!`. The unscented transform propagates
#            the FULL covariance exactly, so the predictive state each horizon is a
#            Gaussian (deterministic — no sampling/`rng`); the caller maps it to the
#            observation distribution with `observation_gaussian_moments`, evaluated at
#            each horizon's own time (the ascertainment may be time-varying).
#   * PF   → `forecast_ensemble`: roll the resampled particle cloud forward with
#            `sample_state`/`sample_measurement`. Learned hyperparameters ride frozen
#            in each particle's tail (the forecast never calls the Liu-West updater),
#            so this is a pure predictive roll at fixed θ.
# Ported from the pre-restructure driver's `generate_forecast`/`optimize_hyperparams`,
# adapted to the current rates/build_petri_vf API (R1 is identity, so a hyperparameter
# re-optimization just re-runs `forward_trajectory(ukf,u,y,p)` with a new `p`).
# ============================================================================

"""
    DEFAULT_QS

Default forecast quantile levels (the standard epi-forecast set).
"""
const DEFAULT_QS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)

"""
    forecast_ensemble(filter, init_states, p; n_ahead, t0, dt=1.0, u=Float64[], latent_range=1:0)
        -> (samples::Matrix{Float64}, latent_samples::Array{Float64,3})

Roll each state in `init_states` forward `n_ahead` steps with NO measurement update
(`correct!`), drawing a predictive observation each horizon. Returns `samples[h, j]`
— the predictive observation at horizon `h` for initial draw `j` (clamped ≥ 0) — together
with `latent_samples[h, j, l]`, the `l`-th requested latent coefficient (in its
**unconstrained** chart) at that horizon. Pass `latent_range = layout.latent_range` to
capture them; the default `1:0` captures nothing and yields a `(n_ahead, n, 0)` array.

The latent block is the forecast-spread diagnostic: it is the realised, across-particle
spread of the latent coefficients (e.g. log-`Rt`) as the horizon grows, which is what the
AR(1) persistence ρ controls. Reading it off the cloud rather than from
`σ²(1−ρ^{2h})/(1−ρ²)` keeps this agnostic to which parameter means "persistence" and
captures the nonlinearity the closed form misses.

This is the PARTICLE-filter forecast path. `sample_state(filter, x, u, p, t)` advances
one step (with process noise) and `sample_measurement(filter, x, u, p, t)` draws the
predictive observation. `init_states` = the (resampled) particle cloud; the learned θ in
each particle's tail is carried frozen because the forecast never calls the Liu-West
updater. (The UKF instead uses `forecast_states` — analytic `predict!` propagation.)

`t0` is the time (days) of the origin; `dt` is the step (one week ⇒ `dt = 7.0`).
"""
function forecast_ensemble(
        filter, init_states::AbstractVector, p;
        n_ahead::Int, t0::Real, dt::Real = 1.0, u = Float64[],
        latent_range::AbstractUnitRange{<:Integer} = 1:0,
        n_obs::Int = 1,
    )
    n_obs > 0 || throw(ArgumentError("n_obs must be positive, got $n_obs"))
    n = length(init_states)
    n_latent_captured = length(latent_range)
    # `samples[h, j]` for one signal, `samples[h, j, o]` for several. The two branches issue
    # `sample_state`/`sample_measurement` in the identical order with identical arguments, so the
    # RNG consumption sequence is bit-identical and the single-signal branch reads `y[1]` exactly as
    # it always did. That is what keeps the existing UKF/PF forecasts unchanged. The small return
    # union is confined to two `build_inference` closures, not a hot loop.
    samples = n_obs == 1 ? Matrix{Float64}(undef, n_ahead, n) :
        Array{Float64, 3}(undef, n_ahead, n, n_obs)
    latent_samples = Array{Float64, 3}(undef, n_ahead, n, n_latent_captured)
    for j in 1:n
        x = init_states[j]
        t = float(t0)
        for h in 1:n_ahead
            x = sample_state(filter, x, u, p, t)        # advance one step, no correct!
            t += dt
            y = sample_measurement(filter, x, u, p, t)  # predictive observation draw
            if n_obs == 1
                samples[h, j] = max(0.0, float(y[1]))   # single-signal counts, non-negative
            else
                for o in 1:n_obs
                    samples[h, j, o] = max(0.0, float(y[o]))
                end
            end
            for (l, slot) in enumerate(latent_range)
                latent_samples[h, j, l] = float(x[slot])   # unconstrained chart
            end
        end
    end
    return samples, latent_samples
end

"""
    forecast_states(kf, x0, R0; n_ahead, t0, dt=1.0, u=Float64[], p) -> (means, covs)

Analytic Kalman forecast for the UKF: `deepcopy` `kf`, seed it at the filtered Gaussian
(`x0` = x_T, `R0` = P_T) and roll forward `n_ahead` steps with `predict!` only (NO
`correct!` — no data). The unscented transform propagates the full state covariance
exactly, so each horizon yields a Gaussian state estimate with NO Monte-Carlo sampling
(deterministic — no `rng`, no draw count). Returns `means[h]`, `covs[h]`: the predicted
state mean (Vector) and covariance (Matrix) at horizon `h`.

This is the proper Kalman forecast (vs. sampling the filtered Gaussian and propagating
draws through `forecast_ensemble`, which Monte-Carlos the state propagation). The caller
maps the per-horizon state Gaussian to the predictive observation — `observation_gaussian_moments`
does this for a NegBinomial signal: `μ_h = α(t_h)·mean[acc]` and
`σ²_h = α(t_h)²·cov[acc,acc] + μ_h + μ_h²/φ`, with `α` the spec's (possibly time-varying)
modifier at the horizon's own time `t_h = t0 + h·dt`.

`t0` is the origin time (days); `dt` is the step (one week ⇒ `dt = 7.0`); `p` is the
hyperparameter set (passed to `predict!` at each step). `kf.x`/`kf.R` are mutated on the
COPY only, so the caller's filter is untouched.
"""
function forecast_states(kf, x0, R0; n_ahead::Int, t0::Real, dt::Real = 1.0, u = Float64[], p)
    fkf = deepcopy(kf)
    fkf.x = x0          # seed at the filtered mean x_T (copy is private)
    fkf.R .= R0         # seed at the filtered covariance P_T (in-place; preserves type)
    means = Vector{Vector{Float64}}(undef, n_ahead)
    covs = Vector{Matrix{Float64}}(undef, n_ahead)
    for h in 1:n_ahead
        predict!(fkf, u, p, t0 + (h - 1) * dt)   # advance one step (one week), NO correct!
        means[h] = Vector{Float64}(fkf.x)        # snapshot (predict! reassigns x / mutates R)
        covs[h] = Matrix{Float64}(fkf.R)
    end
    return means, covs
end

"""
    forecast_quantiles(samples; qs=DEFAULT_QS) -> Matrix{Float64}  # (n_ahead × length(qs))

Per-horizon quantiles of the predictive `samples` (rows = horizons, from
`forecast_ensemble`). Column `k` holds quantile `qs[k]` across draws.
"""
function forecast_quantiles(samples::AbstractMatrix; qs = DEFAULT_QS)
    n_ahead = size(samples, 1)
    _q(col, q) = (v = filter(isfinite, col); isempty(v) ? NaN : quantile(v, q))
    return [_q(view(samples, h, :), q) for h in 1:n_ahead, q in qs]
end

"""
    forecast_quantiles(samples::AbstractArray{<:Real,3}; qs=DEFAULT_QS)
        -> Array{Float64,3}  # (n_ahead × n_observations × length(qs))

Multi-observation sibling of the matrix method: `samples[h, j, o]` (from `forecast_ensemble` with
`n_obs > 1`) reduces over draws `j` to `[horizon, observation, quantile]`. The matrix method is left
exactly as it is, so single-signal callers are unaffected.
"""
function forecast_quantiles(samples::AbstractArray{<:Real, 3}; qs = DEFAULT_QS)
    n_ahead, _, n_obs = size(samples)
    _q(col, q) = (v = filter(isfinite, col); isempty(v) ? NaN : quantile(v, q))
    return [
        _q(view(samples, h, :, o), q) for h in 1:n_ahead, o in 1:n_obs, q in qs
    ]
end

"""
    append_latent_spread!(summary, latent_names, log_sd) -> summary

Append the per-horizon predictive spread of each latent coefficient to a
`(parameter, statistic, value)` summary frame (the one written to
`<stem>_hyperparameters.csv`). `log_sd[h, l]` is the standard deviation of latent `l`'s
**unconstrained** coordinate at horizon `h`.

Two statistics per latent per horizon: `fc_log_sd_h<h>` (the unconstrained-chart sd) and
`fc_factor_h<h> = exp(sd)`, which for a log-chart latent such as `Rt` reads directly as a
multiplicative ±1sd spread. How fast these grow with `h` is exactly what the AR(1)
persistence ρ controls, so this is the readout for "is the correction's forecast budget
still too big?" — a flat profile means corrections do not persist past the origin, while
`fc_log_sd_h4 / fc_log_sd_h1 ≈ 1.7` is the ρ = 0.9 signature.
"""
function append_latent_spread!(summary, latent_names, log_sd::AbstractMatrix)
    for (l, name) in enumerate(latent_names), h in axes(log_sd, 1)
        sd = Float64(log_sd[h, l])
        push!(summary, (string(name), "fc_log_sd_h$h", sd))
        push!(summary, (string(name), "fc_factor_h$h", exp(sd)))
    end
    return summary
end

"""
    append_latent_audit!(summary, latent_names, xt, latent_range; dt) -> summary

Append the **in-sample** (filtered) behaviour of each latent coefficient to a
`(parameter, statistic, value)` summary frame.

This is the counterpart to [`append_latent_spread!`](@ref), which reports only the
FORECAST-horizon spread. Nothing else in the package looks at the filtered path itself, so
there has never been a check that the latent actually behaves like the process it is declared
to be. Four statistics per latent, all in the **unconstrained** chart:

- `is_sd` — realised sd of the filtered path. Against the process's own stationary sd this
  answers "is the correction doing more than the prior admits?".
- `is_acf1` — realised lag-1 autocorrelation.
- `is_tau_days` — the correlation time it implies, `−dt / log(acf1)`. `NaN` when `acf1 ≤ 0`,
  which means there is no correlation time to report rather than that it is zero.
- `is_drift` — mean of the second half minus mean of the first half. A mean-reverting process
  centred on `mu` should give ~0; a persistent offset is the signature of a level the
  mechanism should have supplied.

The ratios that make these falsifiers (`is_sd` against the stationary sd, `is_tau_days`
against the declared correlation time) are deliberately NOT computed here: `src/` does not
know which hyperparameter names mean "persistence" and "process noise", and the emitted CSV
already carries those estimates as `estimate` rows, so the comparison is available downstream
without teaching this function any submodel's vocabulary.

**Read `is_acf1` as a lower bound.** This is the FILTERED path, so each step absorbs a
data-driven innovation and the realised autocorrelation is biased down (and `is_sd` up)
relative to the underlying process. The smoothed path is biased the other way. Reported
honestly as one end of a bracket, not as an estimate of the process.
"""
function append_latent_audit!(
        summary, latent_names, xt::AbstractVector, latent_range::AbstractUnitRange{<:Integer};
        dt::Real
    )
    n = length(xt)
    for (l, slot) in enumerate(latent_range)
        name = string(latent_names[l])
        if n < 3
            push!(summary, (name, "is_sd", NaN))
            push!(summary, (name, "is_acf1", NaN))
            push!(summary, (name, "is_tau_days", NaN))
            push!(summary, (name, "is_drift", NaN))
            continue
        end
        path = [Float64(xt[t][slot]) for t in 1:n]
        m = sum(path) / n
        centred = path .- m
        variance = sum(abs2, centred) / n
        lag_one = sum(centred[t] * centred[t + 1] for t in 1:(n - 1)) / n
        acf1 = variance > 0 ? lag_one / variance : NaN
        tau = (isfinite(acf1) && 0 < acf1 < 1) ? -float(dt) / log(acf1) : NaN
        half = n ÷ 2
        drift = sum(path[(n - half + 1):n]) / half - sum(path[1:half]) / half
        push!(summary, (name, "is_sd", sqrt(variance)))
        push!(summary, (name, "is_acf1", acf1))
        push!(summary, (name, "is_tau_days", tau))
        push!(summary, (name, "is_drift", drift))
    end
    return summary
end


"""
    asof_series(triangle; r_k, date_col=:date, issue_col=:as_of, group_cols=()) -> DataFrame

No-leakage as-of series at report date `r_k`: keep rows with `as_of <= r_k`, then the latest
`as_of` per reference `date` (identity nowcast — the revision-prone recent tail is used as-is).
Sorted by `date`. This is the `as_of <= r_k` discipline from `.github/rolling-data-planning.md`.

`group_cols` names additional keys that identify a distinct series, so "latest issue per reference
date" becomes "latest issue per reference date **within each series**". A multi-location triangle
MUST pass `group_cols = (:location,)`: with the default the grouping key is `date` alone, so
`argmax` keeps one arbitrary row per date and every location but one is silently dropped. The
`as_of <= r_k` filter is untouched either way, so the no-leakage guarantee does not depend on this.
"""
function asof_series(
        triangle; r_k, date_col::Symbol = :date, issue_col::Symbol = :as_of,
        group_cols = (),
    )
    avail = triangle[triangle[!, issue_col] .<= r_k, :]
    isempty(avail) && return avail
    keys_cols = [date_col, group_cols...]
    latest = combine(groupby(avail, keys_cols)) do sub
        sub[[argmax(sub[!, issue_col])], :]   # 1-row frame: latest issue for this series+date
    end
    # Date-major, so the per-date blocks a multi-series caller needs are contiguous.
    return sort!(latest, keys_cols)
end

"""
    backtest_forecast_rows(qmat, origin, target_dates, truth;
                           qs=DEFAULT_QS, model_id, date_col=:date, value_col=:counts) -> DataFrame

Forecast-table rows from a quantile matrix `qmat` (`n_ahead × length(qs)`, from
`forecast_quantiles`): one row per (horizon, quantile) with columns
`origin, horizon, target_date, quantile, value, observed, model_id`. `observed` is looked up
from `truth` (the latest vintage) at each `target_date` (`missing` if absent).
"""
function backtest_forecast_rows(
        qmat::AbstractMatrix, origin, target_dates, truth;
        qs = DEFAULT_QS, model_id::AbstractString,
        date_col::Symbol = :date, value_col::Symbol = :counts
    )
    truth_lookup = Dict(truth[i, date_col] => Float64(truth[i, value_col]) for i in 1:nrow(truth))
    n_ahead, nq = size(qmat, 1), length(qs)
    N = n_ahead * nq
    # Explicit column types so successive origins' frames concatenate cleanly
    # (observed is always Union{Missing,Float64}, even when a batch is all-present).
    col_origin = fill(origin, N)
    col_h = Vector{Int}(undef, N)
    col_td = Vector{eltype(target_dates)}(undef, N)
    col_q = Vector{Float64}(undef, N)
    col_v = Vector{Float64}(undef, N)
    col_obs = Vector{Union{Missing, Float64}}(undef, N)
    col_mid = fill(String(model_id), N)
    r = 0
    for h in 1:n_ahead
        td = target_dates[h]
        obs = get(truth_lookup, td, missing)
        for (k, q) in enumerate(qs)
            r += 1
            col_h[r] = h; col_td[r] = td; col_q[r] = q; col_v[r] = qmat[h, k]; col_obs[r] = obs
        end
    end
    return DataFrame(
        origin = col_origin, horizon = col_h, target_date = col_td,
        quantile = col_q, value = col_v, observed = col_obs, model_id = col_mid,
    )
end

"""
    backtest_forecast_rows(qarr::AbstractArray{<:Real,3}, origin, target_dates, truth;
                           locations, qs=DEFAULT_QS, model_id,
                           date_col=:date, loc_col=:location, value_col=:counts) -> DataFrame

Multi-location sibling: `qarr[h, o, k]` (from the three-dimensional `forecast_quantiles`) becomes
one row per (location, horizon, quantile), with `location` prepended to the seven columns the matrix
method emits. `locations[o]` labels observation `o`, so the caller controls the emitted order, and
`observed` is joined on `(location, target_date)` rather than on the date alone.

The matrix method above is untouched, so a single-signal run's CSV is byte-identical to before.
"""
function backtest_forecast_rows(
        qarr::AbstractArray{<:Real, 3}, origin, target_dates, truth;
        locations, qs = DEFAULT_QS, model_id::AbstractString,
        date_col::Symbol = :date, loc_col::Symbol = :location, value_col::Symbol = :counts,
    )
    n_ahead, n_obs, nq = size(qarr, 1), size(qarr, 2), length(qs)
    n_obs == length(locations) || throw(
        DimensionMismatch(
            "quantile array has $n_obs observations but $(length(locations)) locations " *
                "were named",
        )
    )
    labels = [lowercase(String(loc)) for loc in locations]
    truth_lookup = Dict(
        (lowercase(String(truth[i, loc_col])), truth[i, date_col]) =>
            Float64(truth[i, value_col]) for i in 1:nrow(truth)
    )
    N = n_obs * n_ahead * nq
    col_loc = Vector{String}(undef, N)
    col_origin = fill(origin, N)
    col_h = Vector{Int}(undef, N)
    col_td = Vector{eltype(target_dates)}(undef, N)
    col_q = Vector{Float64}(undef, N)
    col_v = Vector{Float64}(undef, N)
    col_obs = Vector{Union{Missing, Float64}}(undef, N)
    col_mid = fill(String(model_id), N)
    r = 0
    for o in 1:n_obs
        label = labels[o]
        for h in 1:n_ahead
            td = target_dates[h]
            obs = get(truth_lookup, (label, td), missing)
            for (k, q) in enumerate(qs)
                r += 1
                col_loc[r] = label
                col_h[r] = h
                col_td[r] = td
                col_q[r] = q
                col_v[r] = qarr[h, o, k]
                col_obs[r] = obs
            end
        end
    end
    return DataFrame(
        location = col_loc, origin = col_origin, horizon = col_h, target_date = col_td,
        quantile = col_q, value = col_v, observed = col_obs, model_id = col_mid,
    )
end
