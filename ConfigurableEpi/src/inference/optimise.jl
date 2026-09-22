# Hyperparameter optimisation over the prior bundle's unconstrained coordinates.

"""
    DEFAULT_OPTIMISER_STAGES

Adam first (its step is bounded by the learning rate however badly the filter log-posterior is
scaled), then LBFGS to polish; each stage is `remake`d from the previous solution.
"""
const DEFAULT_OPTIMISER_STAGES = ((Adam(0.05), (maxiters = 300,)), (LBFGS(), (maxiters = 50,)))

"""
    optimiser_stages(method) -> Tuple of (optimiser, options)

Normalise a single optimiser, a tuple of optimisers, or a tuple of `(optimiser, options)` pairs.
"""
optimiser_stages(method) = ((method, (;)),)
optimiser_stages(method::Tuple) = map(stage -> stage isa Tuple ? stage : (stage, (;)), method)

"""
    optimize_hyperparams(neg_logposterior, initial::NamedTuple, bundle::ParameterPriorBundle;
                         stages = DEFAULT_OPTIMISER_STAGES, adtype = AutoForwardDiff(), options = (;))
        -> (; θ, ll, retcode, unconstrained)

Minimise `neg_logposterior(u, _)` over the bundle's unconstrained coordinates from the constrained
`initial` values. `options` (such as `maxiters`) are merged over each stage's own; a stage whose
result is non-finite or worse than the incumbent is discarded.
"""
function optimize_hyperparams(
        neg_ll, initial::NamedTuple, bundle::ParameterPriorBundle;
        stages = DEFAULT_OPTIMISER_STAGES, adtype = AutoForwardDiff(), options = (;),
    )
    θ0 = unconstrained_values(bundle, initial)
    prob = OptimizationProblem(OptimizationFunction(neg_ll, adtype), θ0)
    best_u, best_objective, retcode = θ0, neg_ll(θ0, nothing), nothing
    for (optimiser, stage_options) in optimiser_stages(stages)
        sol = solve(remake(prob; u0 = best_u), optimiser; merge(stage_options, options)...)
        retcode = sol.retcode
        if all(isfinite, sol.u) && isfinite(sol.objective) && sol.objective <= best_objective
            best_u, best_objective = collect(sol.u), sol.objective
        end
    end
    return (θ = constrained_values(bundle, best_u), ll = -best_objective, retcode, unconstrained = collect(best_u))
end
