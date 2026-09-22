# The full model definition handed to `build_inference`.

"""
    EpiModel(; vectorfield!, layout, stochastic, observation, hyperparams, priors, initial_state,
             initial_latent_variance = (;), initial_learned_variance = (;),
             initial_accumulator_variance = (;), forgetting_memory_days = (;),
             derived_hyperparameters = nothing)

Everything [`build_inference`](@ref) needs to fit and forecast one model:

- `vectorfield!`: the Petri vector field from [`build_petri_vf`](@ref).
- `layout`, `stochastic`: the [`StateLayout`](@ref) and [`StochasticUpdate`](@ref).
- `observation`: a tuple of observation specs.
- `hyperparams`: every hyperparameter the rates, drivers and observation read, including the
  starting value of each learned one.
- `priors`: `name => ParameterDistribution` for the hyperparameters to learn (a subset of
  `hyperparams`, at least one).
- `initial_state`: the model-state vector at `t = 0`, or a function `hyperparams -> vector` when
  the seed depends on the parameters (an equilibrium `S(0) = N / R0`, say).
- `initial_*_variance`: by-name overrides of the initial variance of latent coefficient slots,
  learned slots (PF) and reset accumulators (EnKF), in unconstrained space.
- `forgetting_memory_days`: the model's default Liu-West forgetting memories (parameter => days),
  overridden per key by `LiuWest.forgetting_memory_days`.
- `derived_hyperparameters`: optional `hyper -> NamedTuple` summarised alongside the learned
  parameters by the particle filter.
"""
struct EpiModel{VF, L <: StateLayout, SU <: StochasticUpdate, O <: Tuple, H <: NamedTuple, P <: NamedTuple, X, D}
    vectorfield!::VF
    layout::L
    stochastic::SU
    observation::O
    hyperparams::H
    priors::P
    build_x0::X
    initial_latent_variance::NamedTuple
    initial_learned_variance::NamedTuple
    initial_accumulator_variance::NamedTuple
    forgetting_memory_days::NamedTuple
    derived_hyperparameters::D
end

function EpiModel(;
        vectorfield!, layout::StateLayout, stochastic::StochasticUpdate,
        observation::Tuple{Vararg{ObservationSpec}}, hyperparams::NamedTuple, priors::NamedTuple,
        initial_state, initial_latent_variance::NamedTuple = (;), initial_learned_variance::NamedTuple = (;),
        initial_accumulator_variance::NamedTuple = (;), forgetting_memory_days::NamedTuple = (;),
        derived_hyperparameters = nothing,
    )
    isempty(observation) && throw(ArgumentError("observation must hold at least one spec"))
    isempty(priors) && throw(ArgumentError("priors must name at least one hyperparameter to learn"))
    for (name, prior) in pairs(priors)
        haskey(hyperparams, name) ||
            throw(ArgumentError("learned parameter $name has no starting value in hyperparams"))
        prior_name(prior) == name ||
            throw(ArgumentError("prior for $name is named $(prior_name(prior))"))
    end
    build_x0 = initial_state isa AbstractVector ? Returns(collect(float.(initial_state))) : initial_state
    x0 = build_x0(hyperparams)
    length(x0) == layout.total_dim ||
        throw(DimensionMismatch("initial state has length $(length(x0)); the layout has $(layout.total_dim) slots"))
    return EpiModel(
        vectorfield!, layout, stochastic, observation, hyperparams, priors, build_x0,
        initial_latent_variance, initial_learned_variance, initial_accumulator_variance,
        forgetting_memory_days, derived_hyperparameters,
    )
end

"""
    initial_state(model::EpiModel, hyperparams = model.hyperparams) -> Vector

The model-state vector at `t = 0` under `hyperparams`.
"""
initial_state(model::EpiModel, hyperparams = model.hyperparams) = model.build_x0(hyperparams)

"The names of the learned hyperparameters."
learned_names(model::EpiModel) = keys(model.priors)
