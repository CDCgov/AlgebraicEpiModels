# ============================================================================
# Inference engine construction — multiple dispatch over the two config axes
# ============================================================================
#
# Turns the two typed config axes (`filter` × `hyper`, from src/config_schema.jl) into the
# concrete method objects and the `(filter, fit_forecast!)` engine. `build_filter`/
# `build_hyper` construct the axis objects; `setup_inference` dispatches on the *pair* to
# supply the method-specific `build_inference` wiring. Valid combinations are exactly those
# with a `setup_inference` method — an invalid pairing (e.g. UKF + Liu–West, which has no
# particle cloud to carry hyperparameters) is a `MethodError`, not a silent wrong run.

# --- Axis A: construct the state filter -------------------------------------------
build_filter(::UKFFilterConfig; weekly_dt, supersample) =
    UKF(dt = weekly_dt, supersample = supersample, obs_jitter = 1.0)
build_filter(fc::PFFilterConfig; weekly_dt, supersample) =
    PF(
    fc.n_particles; dt = weekly_dt, supersample = supersample, obs_jitter = 0.0,
    threads = fc.threads,
)
# `obs_jitter = 0.0`, as for the PF: the accumulator whisker keeps the UKF's Gaussian covariance
# full-rank for the RTS smoother, and an ensemble filter has no such covariance to regularize.
build_filter(fc::EnKFFilterConfig; weekly_dt, supersample) =
    EnKF(
    fc.n_ensemble; dt = weekly_dt, supersample = supersample, obs_jitter = 0.0,
    inflation = fc.inflation, threads = fc.threads,
)

# --- Axis B: construct the hyperparameter-inference method -------------------------
build_hyper(hc::OptimiseConfig, priors) =
    OptimiseHyperparams(priors; options = (maxiters = hc.opt_maxiters,))
# `forgetting_defaults` are the submodel's own (model science, carried on its bundle); the run
# config's `forgetting_memory_days` is merged over them, so it can override or add per parameter.
build_hyper(hc::LiuWestConfig, priors; forgetting_defaults = (;)) =
    LiuWest(
    priors; discount = hc.discount,
    jitter_floor_fraction = hc.jitter_floor_fraction,
    forgetting_memory_days = merge(
        Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in pairs(forgetting_defaults)),
        Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in hc.forgetting_memory_days),
    ),
)
build_hyper(hc::EKPConfig, priors) =
    EKPCalibration(
    priors; n_ensemble = hc.n_ensemble, iterations = hc.iterations,
    burnin_iterations = hc.burnin_iterations, inflation = hc.inflation,
    threads = hc.threads,
)

"""
    setup_inference(filter_cfg, hyper_cfg, bundle, priors; weekly_dt, supersample,
                    n_ahead, n_draws, rng) -> (filter, fit_forecast!)

Build the inference engine for one `(filter, hyper)` pairing. Dispatches on the pair so each
valid combination supplies its own `build_inference` wiring (the UKF/optimise path threads
the re-optimization cadence; the PF/Liu–West path threads the forecast draws + the initial
θ-cloud variance). Both thread `bundle.initial_latent_variance`, which is a property of the
model's latent processes rather than of either inference method. The keyword set is uniform
across methods so the call site is method-agnostic.
"""
function setup_inference(
        fc::UKFFilterConfig, hc::OptimiseConfig, bundle, priors;
        weekly_dt, supersample, n_ahead, n_draws, rng
    )
    filter_method = build_filter(fc; weekly_dt, supersample)
    hyper_method = build_hyper(hc, priors)
    return build_inference(
        filter_method, hyper_method,
        bundle.petri_vf!, bundle.layout, bundle.stochastic, bundle.obs_model,
        bundle.base_hp, bundle.x0_state;
        n_ahead,
        reopt_interval = hc.reopt_interval,
        window_length = hc.window_length,
        warm_start = hc.warm_start,
        initial_optim_options = (maxiters = hc.opt_maxiters_burnin,),
        initial_latent_variance = bundle.initial_latent_variance,
        build_x0 = bundle.build_x0,
    )
end

# Both ensembles on this axis span a subspace of dimension (members - 1), and both are used as
# if they spanned the whole space. Check it at CONSTRUCTION, because the failure mode downstream
# is not a rank error: a singular sample covariance produces spurious cross-covariances, and the
# first thing the user sees is the ODE diverging. That already happened once at L=6 — a smoke test
# at n_ensemble=20 against nx=42 was rank-19-in-42 and was diagnosed through ODE divergence before
# being fixed at 60 (test/test_run_model.jl). A 52-location run would hit the same thing after
# hours of compute.
#
# These two are what actually cap the geographic model: at 7 slots per location and the shipped
# defaults they bind at L <= 28 (inner, N=200) and L <= 26 (outer, ekp_n=32), so the honest ceiling
# for this code is L ~ 26, not 52. Raising `n_ensemble` is the wrong fix past that point —
# localization is, since it removes the long-range sample covariances the rank deficiency corrupts
# rather than paying O(nx) members to estimate them.
function _check_ensemble_rank(fc::EnKFFilterConfig, hc::EKPConfig, bundle, priors)
    nx = bundle.layout.total_dim
    fc.n_ensemble > nx || error(
        "EnKF sample covariance is rank-deficient: n_ensemble = $(fc.n_ensemble) spans " *
            "$(fc.n_ensemble - 1) dimensions but the state has nx = $nx. Set " *
            "`filter.enkf.n_ensemble` above $nx, or — past a few tens of locations, where that " *
            "stops being affordable — add EnKF localization, which is the real fix.",
    )
    n_learned = length(priors)
    hc.n_ensemble > n_learned || error(
        "EKP outer ensemble is rank-deficient: hyper.ekp.n_ensemble = $(hc.n_ensemble) spans " *
            "$(hc.n_ensemble - 1) dimensions but $(n_learned) parameters are learned " *
            "($(join(keys(priors), ", "))). Set `hyper.ekp.n_ensemble` above $(n_learned); the " *
            "outer ensemble can only move within its own span, so a deficient one silently " *
            "freezes some parameter combinations at their initial draw.",
    )
    return nothing
end

function setup_inference(
        fc::EnKFFilterConfig, hc::EKPConfig, bundle, priors;
        weekly_dt, supersample, n_ahead, n_draws, rng
    )
    _check_ensemble_threading(fc.threads, hc.threads)
    _check_ensemble_rank(fc, hc, bundle, priors)
    filter_method = build_filter(fc; weekly_dt, supersample)
    hyper_method = build_hyper(hc, priors)
    return build_inference(
        filter_method, hyper_method,
        bundle.petri_vf!, bundle.layout, bundle.stochastic, bundle.obs_model,
        bundle.base_hp, bundle.x0_state;
        n_ahead,
        reopt_interval = hc.reopt_interval,
        window_length = hc.window_length,
        warm_start = hc.warm_start,
        n_draws = n_draws,
        rng,
        initial_latent_variance = bundle.initial_latent_variance,
        # Ensemble-only: widens `P0` on the reset accumulators so the first correction is not a
        # near-singular innovation. `get` rather than a field access so a submodel that never runs
        # under the EnKF need not carry the field.
        initial_accumulator_variance = get(
            bundle, :initial_accumulator_variance, NamedTuple()
        ),
        build_x0 = bundle.build_x0,
    )
end

function setup_inference(
        fc::PFFilterConfig, hc::LiuWestConfig, bundle, priors;
        weekly_dt, supersample, n_ahead, n_draws, rng
    )
    filter_method = build_filter(fc; weekly_dt, supersample)
    hyper_method = build_hyper(
        hc, priors;
        forgetting_defaults = get(bundle, :learned_forgetting_memory_days, (;)),
    )
    return build_inference(
        filter_method, hyper_method,
        bundle.petri_vf!, bundle.layout, bundle.stochastic, bundle.obs_model,
        bundle.base_hp, bundle.x0_state;
        n_ahead,
        n_draws = n_draws,
        rng,
        initial_learned_variance = bundle.initial_learned_variance,
        initial_latent_variance = bundle.initial_latent_variance,
        build_x0 = bundle.build_x0,
        # Optional: interpretable functions of the learned θ, summarised like learned parameters.
        derived_hyperparameters = get(bundle, :derived_hyperparameters, nothing),
    )
end
