# A joint prior on the Rt scale and the negative-binomial dispersion, through the relative
# variance they add between them and the share of it that transmission carries.

"""
    RtNBVariancePrior(; transmission_weight, window_days = 30, log_sd_mean = log(0.2),
                      log_sd_std = log(2) / 1.6448536269514722, logit_rt_mean = log(9),
                      logit_rt_std = 1, rt_scale = :Rt_sigma_stat, dispersion = :phi)

A [`PriorCoupling`](@ref) with independent normal priors on `log(sqrt(T))` and `logit(f)`, where
`T = transmission_weight * rt_scale^2 + 1/dispersion` is a local relative variance and
`f = transmission_weight * rt_scale^2 / T` the share of it from transmission. `rt_scale` and
`dispersion` name the learned Rt scale and negative-binomial dispersion, whose scalar priors it
replaces; their constraints must be logs.

`transmission_weight` is the conditional response variance at the horizon of interest per unit
squared Rt scale, and the dispersion adds the relative variance of the negative binomial's gamma
multiplier. `T` is a proxy: it leaves out Poisson variation and ascertainment drift.
"""
struct RtNBVariancePrior <: PriorCoupling
    transmission_weight::Float64
    window_days::Float64
    log_sd_mean::Float64
    log_sd_std::Float64
    logit_rt_mean::Float64
    logit_rt_std::Float64
    rt_scale::Symbol
    dispersion::Symbol
    function RtNBVariancePrior(;
            transmission_weight, window_days = 30.0, log_sd_mean = log(0.2),
            log_sd_std = log(2.0) / 1.6448536269514722, logit_rt_mean = log(9.0), logit_rt_std = 1.0,
            rt_scale = :Rt_sigma_stat, dispersion = :phi,
        )
        rt_scale, dispersion = Symbol(rt_scale), Symbol(dispersion)
        rt_scale != dispersion || throw(ArgumentError("Rt/NB prior needs two distinct parameters"))
        all(isfinite, (transmission_weight, window_days, log_sd_mean, log_sd_std, logit_rt_mean, logit_rt_std)) ||
            throw(ArgumentError("Rt/NB prior settings must be finite"))
        all(>(0), (transmission_weight, window_days, log_sd_std, logit_rt_std)) ||
            throw(ArgumentError("Rt/NB prior weights, window and standard deviations must be positive"))
        return new(
            transmission_weight, window_days, log_sd_mean, log_sd_std, logit_rt_mean, logit_rt_std, rt_scale, dispersion,
        )
    end
end

joint_prior_parameters(p::RtNBVariancePrior) = (p.rt_scale, p.dispersion)

_softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))
_rtnb_indices(p::RtNBVariancePrior, b) = map(n -> findfirst(==(n), b.names), joint_prior_parameters(p))

function _validate_coupling(p::RtNBVariancePrior, b)
    pair = joint_prior_parameters(p)
    all(in(b.names), pair) || throw(ArgumentError("Rt/NB joint prior requires learned $(pair[1]) and $(pair[2])"))
    for z in (0.0, 0.7)
        hp = constrained_values(b, fill(z, length(b.names)))
        all(n -> isapprox(hp[n], exp(z); rtol = 1.0e-12), pair) ||
            throw(ArgumentError("Rt/NB joint prior requires log-positive $(pair[1]) and $(pair[2]) constraints"))
    end
    return nothing
end

# `(log T, logit f)` from `(log rt_scale, log dispersion)`, and back.
function _rtnb_summary(p::RtNBVariancePrior, x, y)
    a, b = log(p.transmission_weight) + 2x, -y
    return max(a, b) + log1p(exp(-abs(a - b))), a - b
end
_rtnb_inverse(p::RtNBVariancePrior, s, r) = ((s - _softplus(-r) - log(p.transmission_weight)) / 2, -s + _softplus(r))

function _prior_logpdf(p::RtNBVariancePrior, b, u)
    length(u) == length(b.names) || throw(DimensionMismatch("wrong parameter vector length"))
    all(isfinite, u) || return -Inf
    i, j = _rtnb_indices(p, b)
    s, r = _rtnb_summary(p, u[i], u[j])
    # |d(log T, logit f) / d(log rt_scale, log dispersion)| = 2.
    lp = logpdf(Normal(2p.log_sd_mean, 2p.log_sd_std), s) + logpdf(Normal(p.logit_rt_mean, p.logit_rt_std), r) + log(2.0)
    for k in eachindex(u)
        k in (i, j) && continue
        value = logpdf(b.scalars[k], [u[k]])
        lp += value isa Real ? value : sum(value)
    end
    return lp
end

function _sample_coupled!(rng, p::RtNBVariancePrior, b, u)
    s = 2(p.log_sd_mean + p.log_sd_std * randn(rng))
    r = p.logit_rt_mean + p.logit_rt_std * randn(rng)
    i, j = _rtnb_indices(p, b)
    u[i], u[j] = _rtnb_inverse(p, s, r)
    return u
end

# Gauss-Hermite quadrature over `logit f`; `log T` is independent of it and enters both
# coordinates linearly. Deterministic, so building an engine draws no random numbers.
function _prior_moments(p::RtNBVariancePrior, b)
    m, V = _scalar_moments(b.scalars)
    rule = eigen(SymTridiagonal(zeros(80), sqrt.(Float64.(1:79))))
    weights = rule.vectors[1, :] .^ 2
    r = p.logit_rt_mean .+ p.logit_rt_std .* rule.values
    x, y = -0.5 .* _softplus.(-r), _softplus.(r)
    mx, my = dot(weights, x), dot(weights, y)
    v = (2p.log_sd_std)^2
    i, j = _rtnb_indices(p, b)
    m[i], m[j] = p.log_sd_mean - 0.5log(p.transmission_weight) + mx, -2p.log_sd_mean + my
    V[i, i] = v / 4 + dot(weights, (x .- mx) .^ 2)
    V[j, j] = v + dot(weights, (y .- my) .^ 2)
    V[i, j] = V[j, i] = -v / 2 + dot(weights, (x .- mx) .* (y .- my))
    return m, V
end
