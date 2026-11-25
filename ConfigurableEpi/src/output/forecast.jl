# Forecast rolls, quantiles, latent diagnostics, and the vintage-aware backtest table helpers.

"Default forecast quantile levels."
const DEFAULT_QS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)

"""
    forecast_ensemble(filter, init_states, p; n_ahead, t0, dt = 1.0, u = Float64[], latent_range = 1:0,
                      n_obs = 1, observe_state = Returns(nothing)) -> (samples, latent_samples)

Roll each state forward `n_ahead` steps with `sample_state` and draw a predictive observation with
`sample_measurement` (no `correct!`). `samples[h, j, o]` is the non-negative predictive draw of
observation `o`; `latent_samples[h, j, l]` the requested slots (`latent_range`, any integer
vector) in their unconstrained chart, the forecast-spread diagnostic.
`observe_state(h, j, x)` sees draw `j`'s state at every horizon `h`, from the origin `h = 0`.
"""
function forecast_ensemble(
        filter, init_states::AbstractVector, p; n_ahead::Int, t0::Real, dt::Real = 1.0, u = Float64[],
        latent_range::AbstractVector{<:Integer} = 1:0, n_obs::Int = 1, observe_state = Returns(nothing),
    )
    n_obs > 0 || throw(ArgumentError("n_obs must be positive, got $n_obs"))
    n = length(init_states)
    samples = Array{Float64, 3}(undef, n_ahead, n, n_obs)
    latent_samples = Array{Float64, 3}(undef, n_ahead, n, length(latent_range))
    for j in 1:n
        x = init_states[j]
        t = float(t0)
        observe_state(0, j, x)
        for h in 1:n_ahead
            x = sample_state(filter, x, u, p, t)
            t += dt
            y = sample_measurement(filter, x, u, p, t)
            observe_state(h, j, x)
            for o in 1:n_obs
                samples[h, j, o] = max(0.0, float(y[o]))
            end
            for (l, slot) in enumerate(latent_range)
                latent_samples[h, j, l] = float(x[slot])
            end
        end
    end
    return samples, latent_samples
end

"""
    forecast_quantiles(samples; qs = DEFAULT_QS)

Per-horizon quantiles over draws: `[horizon, quantile]` from `samples[h, j]`, or
`[horizon, observation, quantile]` from `samples[h, j, o]`.
"""
function forecast_quantiles(samples::AbstractMatrix; qs = DEFAULT_QS)
    return [_finite_quantile(view(samples, h, :), q) for h in axes(samples, 1), q in qs]
end
function forecast_quantiles(samples::AbstractArray{<:Real, 3}; qs = DEFAULT_QS)
    return [_finite_quantile(view(samples, h, :, o), q) for h in axes(samples, 1), o in axes(samples, 3), q in qs]
end
_finite_quantile(col, q) = (v = filter(isfinite, col); isempty(v) ? NaN : quantile(v, q))

"""
    append_latent_spread!(summary, latent_names, log_sd) -> summary

Per-horizon predictive spread of each latent coefficient: `fc_log_sd_h<h>` (the unconstrained sd,
`log_sd[h, l]`) and `fc_factor_h<h> = exp(sd)`, a multiplicative ±1sd spread for a log-chart latent.
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

In-sample behaviour of each latent's filtered path (unconstrained chart): `is_sd`, the lag-1
autocorrelation `is_acf1`, the implied correlation time `is_tau_days = -dt / log(acf1)` (`NaN`
when `acf1 <= 0`) and `is_drift`, the second-half mean minus the first-half mean. Filtered paths
absorb data innovations, so read `is_acf1` as a lower bound on the process's own.
"""
function append_latent_audit!(summary, latent_names, xt::AbstractVector, latent_range::AbstractUnitRange{<:Integer}; dt::Real)
    n = length(xt)
    for (l, slot) in enumerate(latent_range)
        name = string(latent_names[l])
        if n < 3
            foreach(stat -> push!(summary, (name, stat, NaN)), ("is_sd", "is_acf1", "is_tau_days", "is_drift"))
            continue
        end
        path = [Float64(xt[t][slot]) for t in 1:n]
        centred = path .- sum(path) / n
        variance = sum(abs2, centred) / n
        acf1 = variance > 0 ? sum(centred[t] * centred[t + 1] for t in 1:(n - 1)) / n / variance : NaN
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

const _LATENT_PATH_Z = 1.6448536269514722   # the standard normal 95% quantile

"""
    latent_path_rows(stochastic, dates, latents) -> DataFrame

Filtered latents `(; mean, sd)`, each `[slot, latent]` in the unconstrained chart the filter
carries, as one row per date and latent: `unconstrained_mean`, `unconstrained_sd`, and `q05`,
`median` and `q95` on the latent's own scale, the chart's Gaussian quantiles mapped through its
constraint. `dates` are the dates of the slots. Empty for `latents === nothing`.
"""
function latent_path_rows(stochastic, dates::AbstractVector, latents)
    rows = DataFrame(
        date = eltype(dates)[], latent = String[], unconstrained_mean = Float64[], unconstrained_sd = Float64[],
        q05 = Float64[], median = Float64[], q95 = Float64[],
    )
    latents === nothing && return rows
    size(latents.mean, 1) == length(dates) ||
        throw(DimensionMismatch("$(length(dates)) dates for latents over $(size(latents.mean, 1)) slots"))
    for (l, spec) in enumerate(stochastic.coefficients)
        constraint = ScalarConstraint(spec.init)
        quantile_at(m, sd, z) = _to_constrained(constraint, m + z * sd)
        for (k, date) in enumerate(dates)
            m, sd = latents.mean[k, l], latents.sd[k, l]
            push!(
                rows, (
                    date, string(spec.name), m, sd, quantile_at(m, sd, -_LATENT_PATH_Z),
                    quantile_at(m, sd, 0.0), quantile_at(m, sd, _LATENT_PATH_Z),
                ),
            )
        end
    end
    return rows
end

"""
    asof_series(triangle; r_k, date_col = :date, issue_col = :as_of, group_cols = ()) -> DataFrame

No-leakage as-of series at report date `r_k`: rows with `as_of <= r_k`, then the latest issue per
reference date (within each `group_cols` series, e.g. `(:location,)`), sorted date-major.
"""
function asof_series(triangle; r_k, date_col::Symbol = :date, issue_col::Symbol = :as_of, group_cols = ())
    available = triangle[triangle[!, issue_col] .<= r_k, :]
    isempty(available) && return available
    keys_cols = [date_col, group_cols...]
    latest = combine(groupby(available, keys_cols)) do sub
        sub[[argmax(sub[!, issue_col])], :]
    end
    return sort!(latest, keys_cols)
end

"""
    backtest_forecast_rows(quantiles, origin, target_dates, truth; qs = DEFAULT_QS, model_id,
                           date_col = :date, value_col = :counts[, locations, loc_col = :location]) -> DataFrame

One row per (horizon, quantile) with columns `origin, horizon, target_date, quantile, value,
observed, model_id`, `observed` joined from `truth` at each target date. A
`[horizon, observation, quantile]` array with `locations` labelling the observations prepends a
`location` column and joins on `(location, date)`.
"""
function backtest_forecast_rows(
        qmat::AbstractMatrix, origin, target_dates, truth;
        qs = DEFAULT_QS, model_id::AbstractString, date_col::Symbol = :date, value_col::Symbol = :counts,
    )
    n_ahead, nq = size(qmat)
    lookup = Dict(truth[i, date_col] => Float64(truth[i, value_col]) for i in 1:nrow(truth))
    return DataFrame(
        origin = fill(origin, n_ahead * nq),
        horizon = repeat(1:n_ahead; inner = nq),
        target_date = repeat(collect(target_dates); inner = nq),
        quantile = repeat(collect(Float64, qs), n_ahead),
        value = vec(permutedims(qmat)),
        observed = Union{Missing, Float64}[get(lookup, target_dates[h], missing) for h in 1:n_ahead for _ in 1:nq],
        model_id = fill(String(model_id), n_ahead * nq),
    )
end

function backtest_forecast_rows(
        qarr::AbstractArray{<:Real, 3}, origin, target_dates, truth;
        locations, qs = DEFAULT_QS, model_id::AbstractString,
        date_col::Symbol = :date, loc_col::Symbol = :location, value_col::Symbol = :counts,
    )
    n_ahead, n_obs, nq = size(qarr)
    n_obs == length(locations) ||
        throw(DimensionMismatch("quantile array has $n_obs observations but $(length(locations)) locations were named"))
    labels = [lowercase(String(loc)) for loc in locations]
    lookup = Dict(
        (lowercase(String(truth[i, loc_col])), truth[i, date_col]) => Float64(truth[i, value_col]) for i in 1:nrow(truth)
    )
    N = n_obs * n_ahead * nq
    return DataFrame(
        location = repeat(labels; inner = n_ahead * nq),
        origin = fill(origin, N),
        horizon = repeat(repeat(1:n_ahead; inner = nq), n_obs),
        target_date = repeat(repeat(collect(target_dates); inner = nq), n_obs),
        quantile = repeat(collect(Float64, qs), n_obs * n_ahead),
        value = [qarr[h, o, k] for o in 1:n_obs for h in 1:n_ahead for k in 1:nq],
        observed = Union{Missing, Float64}[
            get(lookup, (labels[o], target_dates[h]), missing) for o in 1:n_obs for h in 1:n_ahead for _ in 1:nq
        ],
        model_id = fill(String(model_id), N),
    )
end
