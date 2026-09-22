# Scalar EKP priors and the bundle mapping between unconstrained vectors and constrained values.

"""
    prior_name(prior::ParameterDistribution) -> Symbol

Name of a scalar EKP prior. Errors on a combined (multi-name) distribution.
"""
function prior_name(prior::ParameterDistribution)
    names = get_name(prior)
    names isa AbstractVector && length(names) != 1 &&
        throw(ArgumentError("expected a scalar ParameterDistribution, got names=$names"))
    return Symbol(names isa AbstractVector ? only(names) : names)
end

"""
    unconstrained_gaussian(name, mean, sd)
    positive_gaussian(name, mean, sd)
    unit_interval_gaussian(name, mean, sd)

Scalar EKP `constrained_gaussian` priors on `(-Inf, Inf)`, `(0, Inf)` and `(0, 1)`.
"""
unconstrained_gaussian(name, mean::Real, sd::Real) =
    constrained_gaussian(string(name), mean, sd, -Inf, Inf)
positive_gaussian(name, mean::Real, sd::Real) =
    constrained_gaussian(string(name), mean, sd, 0.0, Inf)
unit_interval_gaussian(name, mean::Real, sd::Real) =
    constrained_gaussian(string(name), mean, sd, 0.0, 1.0)

"""
    prior_unconstrained_mean(prior) -> Float64
    prior_unconstrained_variance(prior) -> Float64

Mean and variance of a scalar prior in its unconstrained coordinate, the chart that learned
hyperparameters and latent coefficients are stored in.
"""
prior_unconstrained_mean(prior) = (m = mean(prior); Float64(m isa Number ? m : only(m)))
prior_unconstrained_variance(prior) = (v = var(prior); Float64(v isa Number ? v : only(v)))

"""
    ParameterPriorBundle(priors::NamedTuple)
    ParameterPriorBundle(priors::ParameterDistribution...)

Ordered scalar priors combined into one EKP distribution. [`constrained_values`](@ref),
[`unconstrained_values`](@ref) and [`prior_logpdf`](@ref) map between the optimiser's
unconstrained vector and constrained `NamedTuple`s. NamedTuple keys must equal the prior names.
"""
struct ParameterPriorBundle{N, P}
    names::NTuple{N, Symbol}
    prior::P
end

function ParameterPriorBundle(priors::NamedTuple)
    isempty(priors) && throw(ArgumentError("at least one prior is required"))
    for (key, prior) in pairs(priors)
        prior_name(prior) == key || throw(
            ArgumentError("prior key $key does not match its distribution's name $(prior_name(prior))")
        )
    end
    return ParameterPriorBundle(keys(priors), combine_distributions(collect(values(priors))))
end

ParameterPriorBundle(priors::ParameterDistribution...) =
    ParameterPriorBundle(NamedTuple{map(prior_name, priors)}(priors))

constrained_values(b::ParameterPriorBundle, unconstrained::AbstractVector) =
    NamedTuple{b.names}(Tuple(transform_unconstrained_to_constrained(b.prior, collect(unconstrained))))

function unconstrained_values(b::ParameterPriorBundle, constrained::NamedTuple)
    u = transform_constrained_to_unconstrained(b.prior, [constrained[n] for n in b.names])
    all(isfinite, u) || throw(
        ArgumentError("non-finite unconstrained value: a bounded parameter cannot start on its boundary")
    )
    return u
end

function prior_logpdf(b::ParameterPriorBundle, unconstrained::AbstractVector)
    lp = logpdf(b.prior, collect(unconstrained))
    return lp isa Real ? lp : sum(lp)
end
