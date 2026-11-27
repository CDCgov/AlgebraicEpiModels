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
    PriorCoupling
    Independent()

How a [`ParameterPriorBundle`](@ref) combines its scalar priors. `Independent()` takes their
product; a coupling such as [`RtNBVariancePrior`](@ref) replaces the scalar densities of the
parameters it names with a joint one.
"""
abstract type PriorCoupling end
struct Independent <: PriorCoupling end

"The learned parameters whose scalar priors a coupling replaces."
joint_prior_parameters(::Independent) = ()

"""
    ParameterPriorBundle(priors::NamedTuple; joint = Independent())
    ParameterPriorBundle(priors::ParameterDistribution...)

Ordered scalar priors combined into one EKP distribution, coupled by `joint`
([`PriorCoupling`](@ref)). [`constrained_values`](@ref), [`unconstrained_values`](@ref) and
[`prior_logpdf`](@ref) map between the optimiser's unconstrained vector and constrained
`NamedTuple`s. NamedTuple keys must equal the prior names.
"""
struct ParameterPriorBundle{N, P, S <: NamedTuple, J <: PriorCoupling}
    names::NTuple{N, Symbol}
    prior::P
    scalars::S
    joint::J
end

function ParameterPriorBundle(priors::NamedTuple; joint::PriorCoupling = Independent())
    isempty(priors) && throw(ArgumentError("at least one prior is required"))
    for (key, prior) in pairs(priors)
        prior_name(prior) == key || throw(
            ArgumentError("prior key $key does not match its distribution's name $(prior_name(prior))")
        )
    end
    bundle = ParameterPriorBundle(keys(priors), combine_distributions(collect(values(priors))), priors, joint)
    _validate_coupling(joint, bundle)
    return bundle
end

ParameterPriorBundle(priors::ParameterDistribution...) =
    ParameterPriorBundle(NamedTuple{map(prior_name, priors)}(priors))

"""
    constrained_values(bundle, unconstrained) -> NamedTuple

Transform an unconstrained optimiser vector to named values on their model scales.
"""
constrained_values(b::ParameterPriorBundle, unconstrained::AbstractVector) =
    NamedTuple{b.names}(Tuple(transform_unconstrained_to_constrained(b.prior, collect(unconstrained))))

"""
    unconstrained_values(bundle, constrained) -> Vector

Transform named model-scale parameter values to the optimiser's unconstrained coordinates.
"""
function unconstrained_values(b::ParameterPriorBundle, constrained::NamedTuple)
    u = transform_constrained_to_unconstrained(b.prior, [constrained[n] for n in b.names])
    all(isfinite, u) || throw(
        ArgumentError("non-finite unconstrained value: a bounded parameter cannot start on its boundary")
    )
    return u
end

"""
    prior_logpdf(bundle, unconstrained) -> Real

Evaluate the combined prior density in the optimiser's unconstrained coordinates.
"""
prior_logpdf(b::ParameterPriorBundle, unconstrained::AbstractVector) = _prior_logpdf(b.joint, b, unconstrained)

"""
    prior_moments(bundle) -> (mean, covariance)

Mean and covariance of the bundle's prior in the optimiser's unconstrained coordinates.
"""
prior_moments(b::ParameterPriorBundle) = _prior_moments(b.joint, b)

"""
    sample_prior(rng, bundle) -> Vector

A draw from the bundle's prior in the optimiser's unconstrained coordinates.
"""
sample_prior(rng::Random.AbstractRNG, b::ParameterPriorBundle) =
    _sample_coupled!(rng, b.joint, b, vec(sample(rng, b.prior, 1)))

_validate_coupling(::Independent, _) = nothing
function _prior_logpdf(::Independent, b, unconstrained)
    lp = logpdf(b.prior, collect(unconstrained))
    return lp isa Real ? lp : sum(lp)
end
_prior_moments(::Independent, b) = _scalar_moments(b.scalars)
_sample_coupled!(_, ::Independent, _, u) = u

_scalar_moments(priors) =
    ([prior_unconstrained_mean(p) for p in priors], Matrix(Diagonal([prior_unconstrained_variance(p) for p in priors])))

# A draw from the bundle's prior, for a particle filter's initial learned slots.
struct _PriorSampler{B <: ParameterPriorBundle}
    bundle::B
end
Base.length(d::_PriorSampler) = length(d.bundle.names)
Base.rand(rng::Random.AbstractRNG, d::_PriorSampler) = sample_prior(rng, d.bundle)
