# Warm start: condition the fast part of a starting state on the signal observed just before it.
# The slow part (who is immune) comes from a prior, a replay or a spin-up; the fast part is the
# recent infections the signal implies and the transmission multiplier that continues them.

"""
    log_growth_rate(counts; interval = 1.0) -> Float64

Least-squares slope of `log(counts)` per unit time over counts `interval` apart; zero with fewer
than two counts or any not positive.
"""
function log_growth_rate(counts::AbstractVector; interval::Real = 1.0)
    length(counts) >= 2 && all(c -> isfinite(c) && c > 0, counts) || return 0.0
    return _slope(eachindex(counts) .* float(interval), log.(counts))
end

"""
    signal_incidence(ends, counts, per_infection, split; delay, interval = 7.0, growth_points = 4,
                     until = 0.0) -> BinnedIncidence

The infections a signal implies at the model's observation level: `counts[k]`, reported over the
`interval` ending at `ends[k]` and observed `delay` after infection, is
`Σ_c per_infection[c] split[c, k]` per infection over the interval ending at `ends[k] - delay`,
shared over cells as `split[:, k]` (a vector serves every count). Unobserved infections continue
to `until` at the [`log_growth_rate`](@ref) of the last `growth_points` counts; fewer than two hold
the last rate.
"""
function signal_incidence(
        ends::AbstractVector{<:Real}, counts::AbstractVector{<:Real}, per_infection::AbstractVector{<:Real},
        split::AbstractVecOrMat{<:Real}; delay::Real, interval::Real = 7.0, growth_points::Integer = 4, until::Real = 0.0,
    )
    n, n_cells = length(ends), length(per_infection)
    n >= 1 && length(counts) == n || throw(ArgumentError("signal_incidence needs one count per interval end"))
    all(d -> isapprox(d, interval; atol = 1.0e-6), diff(ends)) ||
        throw(ArgumentError("the signal's interval ends must be $interval apart"))
    all(c -> isfinite(c) && c >= 0, counts) || throw(ArgumentError("signal counts must be finite and non-negative"))
    size(split, 1) == n_cells && size(split, 2) in (1, n) ||
        throw(DimensionMismatch("split needs $n_cells rows and 1 or $n columns"))
    edges = [Float64(ends[1] - delay - interval); Float64.(ends .- delay)]
    rates = zeros(n_cells, n)
    for k in 1:n
        s = view(split, :, min(k, size(split, 2)))
        level = sum(per_infection .* s)
        level > 0 || throw(ArgumentError("the split of interval $k puts no infection in an observed cell"))
        rates[:, k] .= counts[k] / interval .* s ./ level
    end
    growth = growth_points >= 2 ? log_growth_rate(@view(counts[max(1, n - growth_points + 1):n]); interval) : 0.0
    last_mid = edges[end] - interval / 2
    while edges[end] < until - 1.0e-9
        upper = min(edges[end] + interval, Float64(until))
        rates = hcat(rates, rates[:, n] .* exp(growth * ((edges[end] + upper) / 2 - last_mid)))
        push!(edges, upper)
    end
    return BinnedIncidence(edges, rates)
end

"""
    splice_incidence(history, recent) -> BinnedIncidence

`history` up to the first edge of `recent`, then `recent`.
"""
function splice_incidence(history::BinnedIncidence, recent::BinnedIncidence)
    size(history.rates, 1) == size(recent.rates, 1) ||
        throw(DimensionMismatch("the two incidence histories have different cells"))
    cut = first(recent.edges)
    cut > first(history.edges) || return recent
    keep = findall(<(cut), @view(history.edges[1:(end - 1)]))
    return BinnedIncidence([history.edges[keep]; recent.edges], hcat(history.rates[:, keep], recent.rates))
end

function _incidence_between(incidence::BinnedIncidence, c::Integer, from::Real, to::Real)
    (; edges, rates) = incidence
    return sum(max(min(to, edges[b + 1]) - max(from, edges[b]), 0.0) * rates[c, b] for b in 1:(length(edges) - 1))
end

"""
    closing_incidence(incidence; per_infection = nothing) -> (; rate, growth)

Infections per day over all cells at the history's last edge, and their growth per day: the
growth between the last two bins, continued from the last bin's midpoint. With `per_infection` by
cell the rate is in the signal's units, which the signal fixes whatever the split over cells.
"""
function closing_incidence(incidence::BinnedIncidence; per_infection = nothing)
    (; edges, rates) = incidence
    n = size(rates, 2)
    w = per_infection === nothing ? ones(size(rates, 1)) : per_infection
    length(w) == size(rates, 1) || throw(DimensionMismatch("per_infection needs one value per incidence cell"))
    last_rate = sum(w .* @view(rates[:, n]))
    previous = n >= 2 ? sum(w .* @view(rates[:, n - 1])) : 0.0
    last_rate > 0 && previous > 0 || return (; rate = last_rate, growth = 0.0)
    mid = (edges[n] + edges[n + 1]) / 2
    growth = log(last_rate / previous) / (mid - (edges[n - 1] + edges[n]) / 2)
    return (; rate = last_rate * exp(growth * (edges[n + 1] - mid)), growth)
end

"""
    history_closure(incidence, per_infection, ends, counts; delay, interval = 7.0) -> (; reported, implied, ratio)

A reconstructed history pushed back through the observation level: `implied[k]` is
`Σ_c per_infection[c] ×` its infections over the interval ending at `ends[k] - delay`, and `ratio`
is `sum(counts) / sum(implied)`. A ratio far from 1 says the history's level and the observation
level disagree, which a fit would resolve silently by infecting a different number of people.
"""
function history_closure(
        incidence::BinnedIncidence, per_infection::AbstractVector{<:Real}, ends::AbstractVector{<:Real},
        counts::AbstractVector{<:Real}; delay::Real, interval::Real = 7.0,
    )
    length(per_infection) == size(incidence.rates, 1) ||
        throw(DimensionMismatch("per_infection needs one value per incidence cell"))
    length(ends) == length(counts) || throw(DimensionMismatch("history_closure needs one count per interval end"))
    implied = [
        sum(per_infection[c] * _incidence_between(incidence, c, e - delay - interval, e - delay) for c in eachindex(per_infection))
            for e in ends
    ]
    reported = Float64.(counts)
    return (; reported, implied, ratio = sum(reported) / sum(implied))
end

"""
    total_infection_rate(model::ReplayModel, x, hyper, latent, t; per_infection = nothing) -> Real

Hosts entering infection per day at ODE state `x`: what `model.vf!` takes from the routes' source
compartments beyond what `model.replay_vf!` does. With `per_infection` by cell the rate is in the
signal's units, so a source compartment must feed cells of one level.
"""
function total_infection_rate(model::ReplayModel, x, hyper, latent, t; per_infection = nothing)
    xT = promote_type(eltype(x), _value_type(hyper), _value_type(latent)).(x)
    with = build_unified_vf(model.vf!, model.layout)(xT, nothing, (hyper, latent), t)
    without = build_unified_vf(model.replay_vf!, model.layout)(xT, nothing, (hyper, latent), t)
    return -sum(w * (with[i] - without[i]) for (i, w) in _source_weights(model.routes, per_infection))
end

function _source_weights(routes::InfectionRoutes, per_infection)
    per_infection === nothing && return Dict(i => 1.0 for i in routes.source)
    length(per_infection) == length(routes.cells) ||
        throw(DimensionMismatch("per_infection needs one value per cell of the routes"))
    weight = Dict{Int, Float64}()
    for k in eachindex(routes.source)
        w = Float64(per_infection[routes.cell[k]])
        get!(weight, routes.source[k], w) == w ||
            throw(ArgumentError("a source compartment feeds cells of different per-infection levels"))
    end
    return weight
end

_value_type(values::NamedTuple) = promote_type(Float64, (typeof(v) for v in values if v isa Real)...)
_value_type(::Any) = Float64

"""
    matched_transmission_multiplier(rate_at, target; lower = 0.1, upper = 10.0) -> Real

The multiplier `m` at which `rate_at(m)`, affine in `m`, equals `target`, clamped to
`[lower, upper]`; 1 when transmission contributes nothing.
"""
function matched_transmission_multiplier(rate_at, target::Real; lower::Real = 0.1, upper::Real = 10.0)
    base = rate_at(0.0)
    slope = rate_at(1.0) - base
    slope > 0 || return one(slope)
    return clamp((target - base) / slope, lower, upper)
end

"""
    closed_loop_start(model::ReplayModel, x_from, hyper, latent_of, closing; window, accumulate_days, fast,
                      conserve = identity, per_infection = nothing, supersample = 2, integrator = Rk4,
                      first_guess = 1.0, refinements = 2, lower = 0.1, upper = 10.0) -> (; state, multiplier, scale, growth)

Run the model itself over the last `window` days from `x_from`, with the latents held at
`latent_of(m)`, and set two things against `closing` ([`closing_incidence`](@ref)):

- the transmission multiplier `m`, so the infection rate grows at `closing.growth` over the last
  `accumulate_days`, by `refinements` secant steps;
- the `fast` compartments, scaled together by `scale` so the infection rate at the start is
  `closing.rate`; `conserve(state)` returns the difference to the slow compartments.

The reset accumulators hold the flow over the last `accumulate_days`.
"""
function closed_loop_start(
        model::ReplayModel, x_from::AbstractVector, hyper, latent_of, closing; window::Real, accumulate_days::Real,
        fast::AbstractVector{<:Integer}, conserve = identity, per_infection = nothing, supersample::Integer = 2,
        integrator::Type{<:AbstractIntegrator} = Rk4, first_guess::Real = 1.0, refinements::Integer = 2, lower::Real = 0.1, upper::Real = 10.0,
    )
    window > accumulate_days > 0 ||
        throw(ArgumentError("the closed-loop window ($window) must exceed the accumulation interval ($accumulate_days)"))
    rate_at(x, m, t) = total_infection_rate(model, x, hyper, latent_of(m), t; per_infection)
    function run(m)
        latent = Returns(latent_of(m))
        x = promote_type(eltype(x_from), _value_type(hyper), _value_type(latent_of(m))).(x_from)
        x = deterministic_spinup(model.vf!, model.layout, x, hyper, latent, -window, -accumulate_days; supersample, integrator)
        before = rate_at(x, m, -accumulate_days)
        x = deterministic_spinup(
            model.vf!, model.layout, x, hyper, latent, -accumulate_days, 0.0;
            dt = accumulate_days, supersample = round(Int, accumulate_days * supersample), integrator,
            keep_accumulators = true,
        )
        rate = rate_at(x, m, 0.0)
        growth = before > 0 && rate > 0 ? log(rate / before) / accumulate_days : zero(rate)
        return (; state = x, rate, growth)
    end
    m = clamp(float(first_guess), lower, upper)
    at = run(m)
    miss = at.growth - closing.growth
    # The growth rate moves by about one over the generation time per unit log multiplier, so a
    # four-day generation makes the first move a short one.
    next = clamp(m * exp(-clamp(4 * miss, -0.5, 0.5)), lower, upper)
    for _ in 0:refinements
        next == m && break
        previous_m, previous_miss = m, miss
        m = next
        at = run(m)
        miss = at.growth - closing.growth
        miss == previous_miss && break
        next = clamp(m - miss * (m - previous_m) / (miss - previous_miss), lower, upper)
    end
    # Only the rate the fast compartments drive scales with them, and conserving moves it a
    # little, so the scaling is done twice.
    state, rate, scale = at.state, at.rate, one(at.rate)
    for pass in 1:2
        unforced = rate_at(state, zero(m), 0.0)
        closing.rate > unforced && rate > unforced || break
        factor = (closing.rate - unforced) / (rate - unforced)
        state = state .* one(factor)
        state[fast] .*= factor
        state = conserve(state)
        scale *= factor
        pass == 1 && (rate = rate_at(state, m, 0.0))
    end
    return (; state, multiplier = m, scale, at.growth)
end
