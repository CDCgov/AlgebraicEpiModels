# ============================================================================
# PARAMETER RESOLUTION - Resolving ParamSpecs to values
# ============================================================================
#
# Neither transition rates nor measurement parameters are resolved through a
# ParamSpec DSL any more — both use plain `(latent, hyperparams, t)` functions
# (see build_petri_vf and the measurement model). What remains here is value
# resolution for the specs still used by latent dynamics and priors.

"""
    get_param_value(spec::ParamSpec, params::NamedTuple)

Resolve a ParamSpec to its actual value from a merged parameter NamedTuple.

The params NamedTuple should contain both hyperparameters (static) and latent parameters
(time-varying), typically created via `merge(hyperparams, latent_constrained)`.

- FixedParam: return stored value
- HyperParam/LatentParam/ProcessParamSpec: lookup by name in params
- DerivedParam: evaluate formula with params
"""
get_param_value(spec::FixedParam, ::NamedTuple) = spec.value
get_param_value(spec::HyperParam, params::NamedTuple) = params[spec.name]
get_param_value(spec::HyperParamRW, params::NamedTuple) = params[spec.name]
get_param_value(spec::LatentParam, params::NamedTuple) = params[spec.name]
get_param_value(spec::ProcessParamSpec, params::NamedTuple) = params[spec.name]

function get_param_value(spec::DerivedParam, params::NamedTuple)
    return spec.formula(params)
end
