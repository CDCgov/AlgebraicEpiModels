# Initial state: the stability guard on the model's rates, inversion of observed counts into
# compartment occupancies, the peak-anchored susceptible fraction, the seeding strategies and the
# deterministic spin-up.

"""
    stability_limit(integrator::Type{<:AbstractIntegrator}) -> Union{Float64, Nothing}

The largest `|lambda * h|` on the negative real axis at which a SeeToDee integrator is stable: 2
for `ForwardEuler` and `Heun`, 2.513 for `Rk3` and 2.785 for `Rk4`; `nothing` for any other.
"""
function stability_limit end
stability_limit(::Type{<:Rk4}) = 2.785
stability_limit(::Type{<:Rk3}) = 2.513
stability_limit(::Union{Type{<:Heun}, Type{<:ForwardEuler}}) = 2.0
stability_limit(::Type{<:AbstractIntegrator}) = nothing

"How far below its stability limit a substep is warned about."
const STABILITY_HEADROOM = 1.4

function _stability_limit(integrator, limit)
    limit === nothing && throw(ArgumentError("no stability limit is known for $integrator; pass `limit`"))
    limit > 0 || throw(ArgumentError("the stability limit must be positive, got $limit"))
    return float(limit)
end

"""
    IntegrationBounds(durations; n_E = 1, n_I = 1, R_eff_max = 1.0, extra_rates = ())

The rates an ODE substep must integrate: the Erlang stages of `durations`, transmission at up to
`R_eff_max` (size it from the priors' upper tail with [`prior_R_eff_bound`](@ref), not the endemic
seed), and `extra_rates` a model adds outside the shared durations, such as a second observation
chain's stage rate.
"""
struct IntegrationBounds
    durations::Durations
    n_E::Int
    n_I::Int
    R_eff_max::Float64
    extra_rates::Vector{Float64}
    function IntegrationBounds(durations::Durations, n_E::Integer, n_I::Integer, R_eff_max::Real, extra_rates)
        n_E > 0 && n_I > 0 || throw(ArgumentError("stage counts must be positive, got n_E = $n_E, n_I = $n_I"))
        R_eff_max >= 0 || throw(ArgumentError("R_eff_max must be non-negative, got $R_eff_max"))
        all(r -> isfinite(r) && r >= 0, extra_rates) ||
            throw(ArgumentError("extra_rates must be finite and non-negative, got $extra_rates"))
        return new(durations, n_E, n_I, R_eff_max, collect(Float64, extra_rates))
    end
end

IntegrationBounds(durations::Durations; n_E::Integer = 1, n_I::Integer = 1, R_eff_max::Real = 1.0, extra_rates = ()) =
    IntegrationBounds(durations, n_E, n_I, R_eff_max, extra_rates)

"""
    max_transition_rate(bounds::IntegrationBounds) -> Float64

The fastest rate in 1/day: the Erlang stage rates, observation progression, waning, the faster
eigenvalue of the linearised `(E, I)` block at `R_eff_max`, and the extra rates.
"""
function max_transition_rate(b::IntegrationBounds)
    dur = b.durations
    sigma, gamma = b.n_E / dur.latent, b.n_I / dur.infectious
    half_sum = (sigma + gamma) / 2
    transmission = half_sum + sqrt(half_sum^2 + sigma * gamma * (b.R_eff_max - 1))
    return maximum((sigma, gamma, 1 / dur.obs_progression, 1 / dur.immunity, transmission, b.extra_rates...))
end

"""
    required_supersample(bounds::IntegrationBounds, dt; integrator = Rk4, limit = stability_limit(integrator),
                         target = limit / STABILITY_HEADROOM) -> Int

The smallest `supersample` whose substep keeps `|lambda * h| <= target`.
"""
function required_supersample(
        b::IntegrationBounds, dt::Real; integrator::Type{<:AbstractIntegrator} = Rk4, limit = stability_limit(integrator),
        target::Real = _stability_limit(integrator, limit) / STABILITY_HEADROOM,
    )
    target > 0 || throw(ArgumentError("target must be positive, got $target"))
    return max(1, ceil(Int, max_transition_rate(b) * dt / target))
end

"""
    assert_integration_stable(bounds::IntegrationBounds, dt, supersample; integrator = Rk4,
                              limit = stability_limit(integrator)) -> lambda_h

Error at the integrator's [`stability_limit`](@ref) and warn within [`STABILITY_HEADROOM`](@ref)
of it. An integrator without a known limit needs `limit`.
"""
function assert_integration_stable(
        b::IntegrationBounds, dt::Real, supersample::Integer; integrator::Type{<:AbstractIntegrator} = Rk4,
        limit = stability_limit(integrator),
    )
    supersample >= 1 || throw(ArgumentError("supersample must be >= 1, got $supersample"))
    limit = _stability_limit(integrator, limit)
    rate = max_transition_rate(b)
    lambda_h = rate * dt / supersample
    if lambda_h >= limit
        needed = max(supersample + 1, required_supersample(b, dt; integrator, limit, target = limit))
        throw(
            ArgumentError(
                "ODE substep is unstable: the fastest rate is $(round(rate, sigdigits = 3))/day at " *
                    "R_eff <= $(round(b.R_eff_max, sigdigits = 3)) and |lambda*h| = $(round(lambda_h, sigdigits = 3)) " *
                    "exceeds the integrator's limit $limit. Raise `supersample` to at least $needed " *
                    "(currently $supersample) or lengthen the shortest duration.",
            )
        )
    elseif lambda_h > limit / STABILITY_HEADROOM
        @warn "ODE substep is within $(STABILITY_HEADROOM)x of the integrator's stability limit; raise `supersample` for headroom" lambda_h supersample suggested_supersample =
            required_supersample(b, dt; integrator, limit) b.R_eff_max
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

"""
    chain_stage_seed(inflow, growth_rate, rate, n_stages) -> Vector{Float64}

Occupancy of an `n_stages` Erlang chain progressing at `rate` and fed by `inflow · exp(g t)` for
all `t ≤ 0`: stage `k` holds `inflow / (rate + g) · (rate / (rate + g))^(k − 1)`, which at `g = 0`
is the quasi-steady fill `inflow / rate`. An empty chain constrains nothing.
"""
function chain_stage_seed(inflow::Real, growth_rate::Real, rate::Real, n_stages::Integer)
    n_stages >= 0 || throw(ArgumentError("n_stages must be non-negative, got $n_stages"))
    n_stages == 0 && return Float64[]
    isfinite(rate) && rate > 0 || throw(ArgumentError("the stage rate must be finite and positive, got $rate"))
    growth_rate > -rate ||
        throw(ArgumentError("a decline at rate $(-growth_rate) is too fast for a chain progressing at $rate"))
    return [inflow / (rate + growth_rate) * (rate / (rate + growth_rate))^(k - 1) for k in 1:n_stages]
end

"""
    SeedContext(; durations, population, bin_days, ascertainment, history = nothing, first_count = nothing,
                R0 = nothing, chi_at = Returns(1.0), n_E = 1, n_I = 1, n_obs = 1)

What [`initial_seed`](@ref) reads. `history = (; times, counts)` is on the model clock with each
count labelled by the end of its `bin_days` bin; `first_count` is the first fitted count.
`ascertainment` is a constant or `t -> observations per infection`, `chi_at(t)` the seasonal
forcing, and `n_obs` counts the observation stages including the reset accumulator.
"""
struct SeedContext{H, F, R, A, C}
    durations::Durations
    population::Float64
    bin_days::Float64
    ascertainment::A
    history::H
    first_count::F
    R0::R
    chi_at::C
    n_E::Int
    n_I::Int
    n_obs::Int
end

function SeedContext(;
        durations::Durations, population::Real, bin_days::Real, ascertainment, history = nothing,
        first_count = nothing, R0 = nothing, chi_at = Returns(1.0), n_E::Integer = 1, n_I::Integer = 1, n_obs::Integer = 1,
    )
    population > 0 && bin_days > 0 || throw(ArgumentError("population and bin_days must be positive"))
    n_E > 0 && n_I > 0 && n_obs >= 1 ||
        throw(ArgumentError("n_E and n_I must be positive and n_obs at least 1, got $n_E, $n_I, $n_obs"))
    return SeedContext(durations, float(population), float(bin_days), ascertainment, history, first_count, R0, chi_at, n_E, n_I, n_obs)
end

"""
    initial_seed(strategy::InitialisationStrategy, ctx::SeedContext) -> NamedTuple

The state at `t = 0` as fractions and stage fills: `susceptible_fraction`, the stage occupancies
`exposed`, `infectious` and `observation` (the in-transit reporting stages),
`first_observation_accumulator`, and the infection stream `infection_rate · exp(growth_rate t)`
they were filled from, which seeds any further chain through [`chain_stage_seed`](@ref).

The peak strategies invert `first_count` into a flat stream and prime the accumulator with that
count's infections. `SlowModeInitialisation` fits the stream to the last three bins of a history
that ends before `t = 0`, primes the accumulator with the extrapolated bin before it, and also returns
`susceptible_variance` and the terms of its susceptible fraction, `boundary_survival` and `kernel`.
`PeakFromHistoryInitialisation` also returns its `anchor`.
"""
function initial_seed end

function initial_seed(::PeakAtStartInitialisation, ctx::SeedContext)
    R0 = _seed_input(ctx.R0, :R0)
    return (; susceptible_fraction = clamp(1 / max(R0 * ctx.chi_at(0.0), 1.0e-6), 0.0, 1.0), _count_seed(ctx)...)
end

function initial_seed(::PeakFromHistoryInitialisation, ctx::SeedContext)
    R0 = _seed_input(ctx.R0, :R0)
    anchor = ctx.history === nothing ? nothing : peak_anchored_susceptible_fraction(
        ctx.history, ctx.durations, ctx.n_obs, ctx.ascertainment, ctx.population, ctx.bin_days, ctx.chi_at, R0;
        ctx.n_E, ctx.n_I,
    )
    susceptible_fraction = anchor === nothing ? 1 / max(R0, 1.0) : anchor.susceptible_fraction
    return (; susceptible_fraction, _count_seed(ctx)..., anchor)
end

# Integrated against waning survival, the per-capita report rate fixes the susceptible fraction:
# each bin's rate is held over the bin and through any gap after it, and the last up to `t = 0`.
function initial_seed(s::SlowModeInitialisation, ctx::SeedContext)
    h = _seed_input(ctx.history, :history)
    isempty(h.times) && throw(ArgumentError("slow_mode needs pre-start seed history"))
    length(h.times) == length(h.counts) || throw(DimensionMismatch("seed times and counts differ"))
    issorted(h.times; lt = <=) || throw(ArgumentError("seed times must increase"))
    last(h.times) < 0 || throw(ArgumentError("slow_mode seed history must end before the fit start"))
    alpha0 = _ascertainment_at(ctx.ascertainment, 0.0)
    0 < alpha0 <= 1 || throw(ArgumentError("initial ascertainment must lie in (0, 1], got $alpha0"))
    bin = ctx.bin_days
    rates = max.(float.(h.counts), 0.0) ./ (bin * ctx.population)

    omega = inv(ctx.durations.immunity)
    hold(kernel, rate, span) = kernel * exp(-omega * span) - rate * expm1(-omega * span) / omega
    start = first(h.times) - bin
    kernel, t, previous = 0.0, start, first(rates)
    for (stop, rate) in zip(h.times, rates)
        stop - bin >= t || throw(ArgumentError("seed bins of $bin days overlap"))
        kernel = hold(hold(kernel, previous, stop - bin - t), rate, bin)
        t, previous = stop, rate
    end
    kernel = hold(kernel, last(rates), -t)
    q = exp(omega * start)
    logsd = s.historical_multiplier_logsd
    boundary_mean = (s.boundary_low + s.boundary_high) / 2
    boundary_variance = (s.boundary_high - s.boundary_low)^2 / 12
    susceptible_fraction = 1 + (boundary_mean - 1) * q - kernel / alpha0 * exp(logsd^2 / 2)
    susceptible_variance = q^2 * boundary_variance + (kernel / alpha0)^2 * (exp(2 * logsd^2) - exp(logsd^2))

    # The growth rate is the log-linear slope over the last three positive bins. A bin total is an
    # integral, so undo it at the last bin end, advance to `t = 0`, then invert the reporting
    # chain's transfer function `(rho / (rho + g))^(n_obs - 1)`.
    recent = filter(j -> rates[j] > 0, max(1, length(rates) - 2):length(rates))
    g = length(recent) < 2 ? 0.0 : clamp(_slope(h.times[recent], log.(rates[recent])), -0.1, 0.1)
    rho = inv(ctx.durations.obs_progression)
    bin_integral = iszero(g) ? bin : -expm1(-g * bin) / g
    report_rate = last(rates) * ctx.population * bin / bin_integral * exp(-g * last(h.times))
    infection_rate = report_rate / alpha0 * (1 + g / rho)^(ctx.n_obs - 1)
    # The first correction precedes a predict step, so the reset accumulator holds the extrapolated
    # bin before the fit start.
    seed = _stream_seed(ctx, infection_rate, g, report_rate / alpha0 * bin_integral)
    return (; susceptible_fraction, seed..., susceptible_variance, boundary_survival = q, kernel)
end

_seed_input(value, name) = value === nothing ? throw(ArgumentError("this initialisation needs `$name` in its SeedContext")) : value

_slope(xs, ys) = (dx = xs .- mean(xs); sum(dx .* (ys .- mean(ys))) / sum(abs2, dx))

# The stage fills of the infection stream `infection_rate · exp(g t)` through E, I and the
# in-transit reporting stages.
function _stream_seed(ctx::SeedContext, infection_rate, g, first_observation_accumulator)
    dur = ctx.durations
    sigma, gamma, rho = ctx.n_E / dur.latent, ctx.n_I / dur.infectious, inv(dur.obs_progression)
    exposed = chain_stage_seed(infection_rate, g, sigma, ctx.n_E)
    infectious = chain_stage_seed(infection_rate * (sigma / (sigma + g))^ctx.n_E, g, gamma, ctx.n_I)
    observation = chain_stage_seed(infection_rate, g, rho, ctx.n_obs - 1)
    return (; exposed, infectious, observation, first_observation_accumulator, infection_rate, growth_rate = g)
end

# The peak strategies' flat stream: `first_count` inverted into a daily infection rate. The first
# correction precedes a predict step, so the accumulator holds that count's infections.
function _count_seed(ctx::SeedContext)
    count = _seed_input(ctx.first_count, :first_count)
    incidence = initial_infection_state(count, ctx.durations, _ascertainment_at(ctx.ascertainment, 0.0), ctx.bin_days).daily_incidence
    return _stream_seed(ctx, incidence, 0.0, incidence * ctx.bin_days)
end

"""
    deterministic_spinup(petri_vf!, layout, x_ode, hyper, latent_at, t_from, t_to;
                         dt = 1.0, supersample = 2, integrator = Rk4, keep_accumulators = false) -> Vector

Integrate the ODE state from `t_from` to `t_to` in `supersample` substeps per `dt` of a SeeToDee
explicit `integrator`, the latents held on the path `latent_at(t)` rather than filtered. Reset accumulators are zeroed each `dt` step, as in a filter
step, and at the end unless `keep_accumulators`, which leaves them holding the last step's flow.
States are floored at zero, not `STATE_FLOOR`, so an empty compartment stays empty.
"""
function deterministic_spinup(
        petri_vf!, layout::StateLayout, x_ode::AbstractVector, hyper, latent_at, t_from::Real, t_to::Real;
        dt::Real = 1.0, supersample::Integer = 2, integrator::Type{<:AbstractIntegrator} = Rk4, keep_accumulators::Bool = false,
    )
    unified = build_unified_vf(petri_vf!, layout)
    f(x, u, p, t) = unified(x, u, (p, latent_at(t)), t)
    return _integrate(f, layout, x_ode, hyper, t_from, t_to; dt, supersample, integrator, keep_accumulators)
end

# The integrator behind `deterministic_spinup` and `replay_state`: `before_substep(x, t, h)` runs
# before each of the `supersample` substeps of a `dt` step.
function _integrate(
        f, layout::StateLayout, x0::AbstractVector, hyper, t_from::Real, t_to::Real;
        dt::Real, supersample::Integer, integrator::Type{<:AbstractIntegrator}, keep_accumulators::Bool,
        before_substep = Returns(nothing),
    )
    supersample >= 1 || throw(ArgumentError("supersample must be at least 1, got $supersample"))
    t_to >= t_from || throw(ArgumentError("the integration ends at $t_to before it starts at $t_from"))
    n_steps = round(Int, (t_to - t_from) / dt)
    isapprox(t_from + n_steps * dt, t_to; atol = 1.0e-9) ||
        throw(ArgumentError("the span $(t_to - t_from) is not a whole number of $dt-day steps"))
    T = promote_type(eltype(x0), typeof(float(t_from)))
    h = float(dt) / supersample
    step = integrator(f, h)
    function reset!(x)
        @inbounds for i in layout.accumulator_indices
            x[i] = zero(T)
        end
        return x
    end
    x = T.(x0)
    for k in 0:(n_steps - 1)
        x = reset!(max.(x, zero(T)))
        for s in 0:(supersample - 1)
            t = t_from + k * dt + s * h
            before_substep(x, t, h)
            x = step(x, nothing, hyper, t)
        end
    end
    x = Vector{T}(max.(x, zero(T)))
    return keep_accumulators ? x : reset!(x)
end
