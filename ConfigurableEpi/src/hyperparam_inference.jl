# ============================================================================
# Hyperparameter Inference Method Types
# ============================================================================

"""
    HyperparamInferenceMethod

Abstract base type for methods used to infer or fix model hyperparameters.
"""
abstract type HyperparamInferenceMethod end

"""
    DEFAULT_OPTIMISER_STAGES

Default hyperparameter-optimizer schedule: **Adam, then LBFGS to polish.**

Each entry is an `(optimiser, options)` pair, run in sequence by
`optimize_hyperparams` via `remake` (the SciML composition pattern), each stage
starting from the previous stage's solution.

Why staged rather than LBFGS alone: the filter marginal log-posterior is badly *scaled*, not
merely nonlinear. Measured on `basic_seir` at its start point, the objective is ~1e11 while its
gradient components are ~1e3, and the objective has a cliff — where the filter diverges it jumps
to a flat `_DIVERGED_PENALTY` whose likelihood-gradient is exactly zero. A quasi-Newton
method takes an essentially steepest-descent first step before it has any curvature estimate, so
that scaling makes it overshoot, and the flat cliff corrupts the inverse-Hessian approximation
if it lands there. Adam normalises each coordinate by that coordinate's own running gradient
RMS, so its step is bounded by the learning rate however the objective is scaled — it walks in
robustly, and LBFGS then converges quickly from a sane starting point.

The gradient itself is exact and cheap (forward-mode AD through the filter costs about one
primal evaluation), and is *more* accurate than finite differences here: at this scale central
FD suffers catastrophic cancellation and returns 0.0 for components AD resolves correctly.
"""
const DEFAULT_OPTIMISER_STAGES = (
    (Adam(0.05), (maxiters = 300,)),
    (LBFGS(), (maxiters = 50,)),
)

"""
    OptimiseHyperparams(bundle; method=DEFAULT_OPTIMISER_STAGES, adtype=AutoForwardDiff(), options=(;))
    OptimiseHyperparams(priors::NamedTuple; method=..., adtype=..., options=(;))

Configure optimization-based hyperparameter inference. `bundle` owns the
ordered parameter names, constraint transformations, and joint prior used by
the optimizer. Passing a `NamedTuple` of scalar priors constructs the
`ParameterPriorBundle` automatically.

`method` accepts any of:

- a **tuple of `(optimiser, options)` pairs** — a staged schedule, chained with `remake`
  (the default; see [`DEFAULT_OPTIMISER_STAGES`](@ref));
- a **tuple of optimisers** — the same, with `options` applied to every stage;
- a **single optimiser** — one stage, e.g. `NelderMead()`.

`options` is merged **over** each stage's own options, so a run config's `opt_maxiters` caps
every stage (gradient stages stop early on convergence anyway).

The unconstrained chart the priors define means no box constraints are needed, so unconstrained
optimizers apply directly. `adtype = AutoForwardDiff()` supplies exact gradients; pass a
derivative-free `method` together with `adtype = SciMLBase.NoAD()` to skip building gradients
that will not be used.
"""
struct OptimiseHyperparams{B <: ParameterPriorBundle, M, A, O} <: HyperparamInferenceMethod
    priors::B
    method::M
    adtype::A
    options::O
end

function OptimiseHyperparams(
        priors::ParameterPriorBundle;
        method = DEFAULT_OPTIMISER_STAGES,
        adtype = AutoForwardDiff(),
        options = (;)
    )
    return OptimiseHyperparams(priors, method, adtype, options)
end

function OptimiseHyperparams(
        priors::NamedTuple;
        method = DEFAULT_OPTIMISER_STAGES,
        adtype = AutoForwardDiff(),
        options = (;)
    )
    return OptimiseHyperparams(ParameterPriorBundle(priors); method, adtype, options)
end

"""
    EKPCalibration(priors; n_ensemble, iterations, burnin_iterations=iterations,
                   inflation=0.0, threads=false)

Outer ensemble-Kalman-inversion calibration of the static parameters, the derivative-free
counterpart of [`OptimiseHyperparams`](@ref).

`OptimiseHyperparams` differentiates the filter's marginal log-posterior with ForwardDiff, which
requires the filter to be rebuilt at the AD element type on every evaluation and confines it to
paths where every operation is Dual-safe. An ensemble filter is not: it resamples from `d0`, so its
likelihood is a Monte-Carlo estimate with no usable derivative. Calibration therefore moves outward
— an ensemble of candidate parameter vectors, each scored by one complete inner filter replay,
updated by `EnsembleKalmanProcesses.TransformInversion`.

The loss handed to EKP is `sqrt(max(-2·ll, eps()))` against a target of `[0.0]`, i.e. EKP performs
its Gauss-Newton-flavoured step on a least-squares reading of the deviance.
`TransformInversion(prior)` imposes the prior itself, so — unlike the optimiser path — the loss
carries **no** `prior_logpdf` term.

`priors` accepts a [`ParameterPriorBundle`](@ref) or a `NamedTuple` of `ParameterDistribution`s;
the field name `priors` is shared with `OptimiseHyperparams` because `build_inference` reads
`.priors.names` to fix the learned set.

`threads = true` evaluates independent outer candidates concurrently. It cannot be combined with
the inner EnKF's own threading.
"""
struct EKPCalibration{B <: ParameterPriorBundle} <: HyperparamInferenceMethod
    priors::B
    n_ensemble::Int
    iterations::Int
    burnin_iterations::Int
    inflation::Float64
    threads::Bool

    function EKPCalibration(
            priors::B, n_ensemble::Integer, iterations::Integer,
            burnin_iterations::Integer, inflation::Real, threads::Bool,
        ) where {B <: ParameterPriorBundle}
        # EKP itself warns below 10; two is the hard floor for a sample covariance.
        n_ensemble > 1 || throw(
            ArgumentError("n_ensemble must be at least 2, got $n_ensemble")
        )
        iterations > 0 || throw(
            ArgumentError("iterations must be positive, got $iterations")
        )
        burnin_iterations > 0 || throw(
            ArgumentError("burnin_iterations must be positive, got $burnin_iterations")
        )
        inflation >= 0 || throw(
            ArgumentError("inflation must be non-negative, got $inflation")
        )
        return new{B}(
            priors, Int(n_ensemble), Int(iterations), Int(burnin_iterations),
            Float64(inflation), threads,
        )
    end
end

EKPCalibration(
    priors::ParameterPriorBundle, n_ensemble::Integer, iterations::Integer,
    burnin_iterations::Integer, inflation::Real,
) = EKPCalibration(
    priors, n_ensemble, iterations, burnin_iterations, inflation, false,
)

EKPCalibration(
    priors::ParameterPriorBundle; n_ensemble::Integer, iterations::Integer,
    burnin_iterations::Integer = iterations, inflation::Real = 0.0,
    threads::Bool = false,
) = EKPCalibration(priors, n_ensemble, iterations, burnin_iterations, inflation, threads)

EKPCalibration(priors::NamedTuple; kwargs...) =
    EKPCalibration(ParameterPriorBundle(priors); kwargs...)

"""
    optimiser_stages(method, options) -> Tuple of (optimiser, options)

Normalise the three accepted `method` spellings into a uniform stage list, with `options`
merged over each stage's own options. See [`OptimiseHyperparams`](@ref).
"""
optimiser_stages(method, options) = ((method, options),)
function optimiser_stages(method::Tuple, options)
    return map(method) do stage
        stage isa Tuple ? (stage[1], merge(stage[2], options)) : (stage, options)
    end
end

"""
    optimize_hyperparams(neg_logposterior, initial_values, method)

Minimize `neg_logposterior` in the unconstrained coordinates defined by
`method.priors`, starting from the constrained `NamedTuple` `initial_values`.
The optimizer schedule, its AD backend, and the splatted keyword options come from `method`.
Return the constrained estimate as `θ` together with the maximized log-posterior,
termination code, and unconstrained optimizer solution.

A multi-stage `method` runs its stages in sequence, each `remake`d from the previous stage's
solution — Adam to walk in robustly, then LBFGS to converge. The reported `retcode` is the last
stage's. If a stage returns a non-finite objective or a non-finite point its result is discarded
and the next stage restarts from the last good point, so one bad stage cannot poison the run.
"""
function optimize_hyperparams(
        neg_ll, init_constrained::NamedTuple, optim_opts::OptimiseHyperparams
    )
    bundle = optim_opts.priors
    θ0 = unconstrained_values(bundle, init_constrained)
    prob = OptimizationProblem(
        OptimizationFunction(neg_ll, optim_opts.adtype), θ0
    )
    stages = optimiser_stages(optim_opts.method, optim_opts.options)
    best_u, best_objective, retcode = θ0, neg_ll(θ0, nothing), nothing
    for (optimiser, options) in stages
        sol = solve(remake(prob; u0 = best_u), optimiser; options...)
        retcode = sol.retcode
        # Keep a stage's result only if it is finite and did not make things worse. Adam with a
        # fixed learning rate can step out of the feasible region on a cliffed objective, and a
        # silently-worse handoff would then be polished into a bad answer by the next stage.
        if all(isfinite, sol.u) && isfinite(sol.objective) && sol.objective <= best_objective
            best_u, best_objective = collect(sol.u), sol.objective
        end
    end
    return (
        θ = constrained_values(bundle, best_u),
        ll = -best_objective,
        retcode = retcode,
        unconstrained = collect(best_u),
    )
end
