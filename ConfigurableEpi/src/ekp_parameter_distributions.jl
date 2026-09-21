# ============================================================================
# EKP PARAMETER DISTRIBUTION BRIDGE
# ============================================================================

"""
    prior_name(prior::ParameterDistribution) -> Symbol

Return the scalar parameter name stored in an EKP `ParameterDistribution`.

This is intended for single-parameter distributions only. It throws an
`ArgumentError` if the distribution carries multiple names, such as a combined
distribution built with `combine_distributions`.
"""
function prior_name(prior::ParameterDistribution)
    names = get_name(prior)
    if names isa AbstractVector
        length(names) == 1 ||
            throw(ArgumentError("Expected a scalar ParameterDistribution, got names=$names"))
        return Symbol(only(names))
    end
    return Symbol(names)
end

"""
    unconstrained_gaussian(name, mean, sd) -> ParameterDistribution

Construct a scalar Gaussian prior with no constraint.

This is a convenience wrapper around EKP's `constrained_gaussian` using
`(-Inf, Inf)` bounds.
"""
function unconstrained_gaussian(name::Union{Symbol, AbstractString}, mean::Real, sd::Real)
    return constrained_gaussian(string(name), mean, sd, -Inf, Inf)
end

"""
    positive_gaussian(name, mean, sd) -> ParameterDistribution

Construct a scalar Gaussian prior constrained to `(0, Inf)`.

This is primarily used for scales and other positive-valued parameters.
"""
function positive_gaussian(name::Union{Symbol, AbstractString}, mean::Real, sd::Real)
    return constrained_gaussian(string(name), mean, sd, 0.0, Inf)
end

"""
    unit_interval_gaussian(name, mean, sd) -> ParameterDistribution

Construct a scalar Gaussian prior constrained to `(0, 1)`.

This is primarily used for persistence or reporting-rate parameters that must
stay in the unit interval.
"""
function unit_interval_gaussian(name::Union{Symbol, AbstractString}, mean::Real, sd::Real)
    return constrained_gaussian(string(name), mean, sd, 0.0, 1.0)
end

"""
    ParameterPriorBundle

Container for a combined EKP prior together with the ordered parameter names
used to map between unconstrained vectors and constrained `NamedTuple`s.

# Fields
- `names::NTuple{N, Symbol}`: Parameter names in bundle order
- `prior`: Combined EKP distribution from `combine_distributions`
"""
struct ParameterPriorBundle{N, P}
    names::NTuple{N, Symbol}
    prior::P
end

"""
    ParameterPriorBundle(priors::Vararg{ParameterDistribution})
    ParameterPriorBundle(priors::NamedTuple)

Combine scalar EKP priors into a single bundle for optimization.

The bundle preserves the prior order and exposes helper methods for converting
between unconstrained optimizer vectors and constrained `NamedTuple`s.
"""
function ParameterPriorBundle(priors::Vararg{ParameterDistribution, N}) where {N}
    names = Tuple(prior_name(prior) for prior in priors)
    combined = combine_distributions(collect(priors))
    return ParameterPriorBundle{N, typeof(combined)}(names, combined)
end

function ParameterPriorBundle(priors::NamedTuple)
    return ParameterPriorBundle(values(priors)...)
end

"""
    constrained_values(bundle, unconstrained) -> NamedTuple

Transform an unconstrained optimizer vector into constrained parameter values
using the bundle's combined EKP prior.
"""
function constrained_values(bundle::ParameterPriorBundle, unconstrained::AbstractVector)
    constrained = transform_unconstrained_to_constrained(bundle.prior, collect(unconstrained))
    return NamedTuple{bundle.names}(Tuple(constrained))
end

"""
    unconstrained_values(bundle, constrained) -> Vector

Transform constrained parameter values into the bundle's unconstrained EKP
coordinates.

Throws an `ArgumentError` if any transformed value is non-finite, which occurs
when bounded parameters are initialized exactly on a boundary.
"""
function unconstrained_values(bundle::ParameterPriorBundle, constrained::NamedTuple)
    ordered = [constrained[name] for name in bundle.names]
    unconstrained = transform_constrained_to_unconstrained(bundle.prior, ordered)
    all(isfinite, unconstrained) || throw(
        ArgumentError(
            "Constraint transform produced a non-finite unconstrained vector. " *
                "Bounded latent states cannot be initialized exactly at the boundary."
        )
    )
    return unconstrained
end

"""
    prior_logpdf(bundle, unconstrained) -> Real

Evaluate the log-density of an unconstrained optimizer vector under a combined
EKP prior bundle.

The helper always returns a scalar, summing component-wise values when EKP
returns a vector of log-densities.
"""
function prior_logpdf(bundle::ParameterPriorBundle, unconstrained::AbstractVector)
    values = logpdf(bundle.prior, collect(unconstrained))
    return values isa Real ? values : sum(values)
end
