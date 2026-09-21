module ConfigurableEpi

using LinearAlgebra: BLAS, Diagonal, I, diag, cholesky, cholesky!, Symmetric,
    dot, issuccess, logdet, mul!
using Dates, Random
import TOML

# Configuration schema (typed, validated, TOML-native @option structs)
using Configurations: @option, from_dict, from_toml, from_kwargs

# Algebraic dependencies
using AlgebraicPetri, Catlab, AlgebraicEpiMech

# Array/parameter handling
# The outer ensemble-Kalman-inversion lifecycle (src/ekp_calibration.jl). Distinct from the
# `.ParameterDistributions` submodule below, which supplies only priors and constraint transforms.
using EnsembleKalmanProcesses:
    DataMisfitController,
    EnsembleKalmanProcess,
    SampleSuccGauss,
    TransformInversion,
    construct_initial_ensemble,
    get_u_final,
    get_Δt,
    get_ϕ_final,
    update_ensemble!

using EnsembleKalmanProcesses.ParameterDistributions:
    ParameterDistribution,
    bounded,
    bounded_above,
    bounded_below,
    combine_distributions,
    constrained_gaussian,
    get_all_constraints,
    get_name,
    no_constraint,
    transform_constrained_to_unconstrained,
    transform_unconstrained_to_constrained
using Distributions: logpdf, Poisson, NegativeBinomial, LogNormal, Beta
using LabelledArrays
using StaticArrays: SVector, SMatrix, MMatrix, setindex

# Distributions
using Distributions: MvNormal, Normal

# Special functions
using LogExpFunctions

# State estimation, filtering and numerical integration
using LowLevelParticleFilters: AbstractKalmanFilter, predict!, correct!, reset!,
    state, covariance, parameters, UnscentedKalmanFilter,
    KalmanFilteringSolution, forward_trajectory, smooth,
    sample_state, sample_measurement, AdvancedParticleFilter, TrivialParams,
    particles, expweights, SignalNames
# Qualified handle for src/augmented_enkf.jl, which ADDS METHODS to LLPF's generic functions
# (`LLPF.predict!(::AugmentedEnsembleKalmanFilter, …)` and friends). Extending needs `import`
# or a qualified definition, and the qualified form keeps names like `dynamics`/`measurement`
# out of this namespace — they would collide with the local closures of the same name that
# `build_dynamics_and_measurement` and `build_full_dynamics` define.
import LowLevelParticleFilters as LLPF

using PositiveFactorizations: Positive, ldlt!, default_tol, default_blocksize, floattype
using SeeToDee: Rk4

# Seasonality climatology interpolation (src/seasonality.jl)
using DataInterpolations: CubicSpline
import DataInterpolations

# Hyperparameter optimization
using Statistics: quantile, var, std, mean
using Optimization: OptimizationFunction, OptimizationProblem, solve, remake, AutoForwardDiff
using OptimizationOptimJL: NelderMead, LBFGS, BFGS
using OptimizationOptimisers: Adam
import ForwardDiff

# Data handling

using DataFramesMeta: DataFrame, groupby, combine, sort!, select, transform!,
    nrow, names, Not, ByRow
using DBInterface: close, connect, execute
using DuckDB: DB, register_data_frame, unregister_data_frame

# Useful constants

# Compartmental constructors selectable by `model_type` (all S,E,I-bearing).
const COMPARTMENTAL_MODELS = Dict(:SEIRS => SEIRS, :SEIR => SEIR, :SEIS => SEIS, :SEI => SEI)

"""
    DAILY_NSSP_SUBMODELS

Submodels that run on the daily NSSP count path: one independent, from-scratch fit per Wednesday
report vintage (`origin_mode = "independent"`), dispatched to the daily runner by the entrypoint.
"""
const DAILY_NSSP_SUBMODELS = ("basic_seir_daily_nssp", "basic_seir_daily_nssp_trend")

# Defined here, ahead of the includes, because `config_schema.jl` uses it as an @option
# default and is included before `learned_hyperparams.jl`.
"""
    DEFAULT_JITTER_FLOOR_FRACTION

Minimum Liu-West jitter variance per parameter, as a fraction of that parameter's own **prior**
variance in unconstrained space.

Liu-West's jitter is proportional to the cloud's current variance, so without a floor a
degenerate cloud can never re-expand — collapse is an absorbing state (see
`_build_hyperparam_updater` in learned_hyperparams.jl for the measured evidence). This sets the
smallest step the
kernel will ever take. At 1e-3 the floor standard deviation is ~3% of the prior's, so a frozen
cloud diffuses back out over tens of steps while a healthy cloud is untouched (the floor is
applied with `max`, not added).

Too small and collapse stays effectively absorbing; too large and θ random-walks instead of
converging. Tune via `LiuWest(priors; jitter_floor_fraction = …)`.
"""
const DEFAULT_JITTER_FLOOR_FRACTION = 1.0e-3

export COMPARTMENTAL_MODELS

# Export state filter method types
export StateFilterMethod, UKF, EnKF, PF

# Export the augmented-noise ensemble filter (src/augmented_enkf.jl)
export AugmentedEnsembleKalmanFilter

# Export hyperparameter inference method types
export HyperparamInferenceMethod, OptimiseHyperparams,
    EKPCalibration,
    DEFAULT_OPTIMISER_STAGES, optimiser_stages, marginal_loglik, positive_cholesky!

# Export ParamSpec types
export ParamSpec, ProcessParamSpec, FixedParam, HyperParam, HyperParamRW,
    AR1ParamSpec, RWParamSpec, IntegratedParamSpec, DerivedParam, LatentParam

# Export EKP parameter helpers
export ParameterDistribution, bounded, bounded_above, bounded_below, no_constraint,
    constrained_gaussian, unconstrained_gaussian, positive_gaussian,
    unit_interval_gaussian,
    prior_logpdf, ParameterPriorBundle,
    constrained_values, unconstrained_values

# Export the submodel-agnostic configuration schema (@option types + loaders).
# Submodel-specific epi configs live in their own submodels/<name>.jl file.
export PriorSpec, Durations, initial_infection_state, max_transition_rate,
    assert_integration_stable, required_supersample,
    RK4_STABILITY_LIMIT, RK4_WARN_LAMBDA_H,
    SeasonalityConfig, DayOfWeekConfig, NoDayOfWeekConfig, PluginDayOfWeekConfig,
    LearnedDayOfWeekConfig, RunConfig, RunIO,
    InputConfig, WeeklyNSSPPercentInputConfig, DailyNSSPCountInputConfig,
    UKFFilterConfig, PFFilterConfig, EnKFFilterConfig,
    OptimiseConfig, LiuWestConfig, EKPConfig,
    filter_name, hyper_name, filter_symbol,
    build_filter, build_hyper, setup_inference,
    build_prior, build_priors, resolve_priors, resolve_prior_specs,
    prior_upper, prior_R_eff_bound,
    assert_no_retired_priors, RETIRED_PRIOR_NAMES,
    load_prior_specs, load_priors,
    submodel_name, resolved_input, resolved_step_days,
    resolved_burnin_observations, resolved_drop_recent_observations,
    validate_run_semantics, from_dict, from_toml, from_kwargs, DAILY_NSSP_SUBMODELS

# Export dynamics API types
export StateLayout

# Export dynamics API functions
export all_names, ode_names, n_ode_states, n_latent, n_signals, extract_latent,
    make_lvector_constructor, make_slvector_constructor,
    build_unified_vf

# Export the stochastic-driver builder (the Lévy noise driver: diffusion coefficients + jumps)
export get_param_value, update_single, ou_step, build_stochastic_update,
    StochasticUpdate
export build_petri_vf

# Export the shared transmission-seasonality forcing (all submodels use this one implementation)
export reporting_delay_days, prevalence_peak_lag_days, anchor_is_exact,
    find_observed_peak, carry_susceptible, peak_anchored_susceptible_fraction

export SeasonalForcing, UnitForcing, CosineForcing, IndoorActivityForcing,
    build_seasonal_forcing, build_seasonal_forcings,
    validate_seasonality, build_periodic_curve, year_fraction,
    load_indoor_activity_climatology, validate_indoor_activity_climatology, N_SEASON_KNOTS,
    default_seasonal_learned, assert_seasonal_learnable, seasonal_forcing_upper_bound

# Export the declining ascertainment path (src/ascertainment.jl): the report's log-linear decline
# with an asserted floor, anchored on the calendar the way the seasonal forcing is
export AscertainmentPath, ascertainment_at, build_ascertainment_path,
    parse_ascertainment_reference_date, validate_ascertainment, assert_ascertainment_learnable,
    ASCERTAINMENT_DECLINE_RATE_PRIOR_MEAN, ASCERTAINMENT_DECLINE_RATE_PRIOR_SD,
    DEFAULT_ASCERTAINMENT_DECLINE_RATE, DEFAULT_ASCERTAINMENT_FLOOR_FRACTION,
    DEFAULT_ASCERTAINMENT_REFERENCE_DATE, DEFAULT_ASCERTAINMENT_RATE_BOUND

# Export the weekday observation effect (src/day_of_week.jl): a multiplier on the observation
# mean and a per-weekday widening of its dispersion, so a predictable weekly cycle is not read as
# transmission
export DOW_LEARNED_NAMES, DOW_DAY_ABBREVIATIONS, DOW_HELMERT,
    dow_helmert_prior_sd, dow_log_effects, day_of_week_multipliers, day_of_week_weight,
    remove_day_of_week_effect,
    DayOfWeekModifier, DayOfWeekDispersion, HyperPhi, validate_day_of_week,
    estimate_day_of_week_effects, prepare_day_of_week_history, build_day_of_week_observation,
    day_of_week_learned_names, assert_day_of_week_learnable, day_of_week_report_rows

# Export the radiation-model cross-location contact mixing (src/radiation_mixing.jl)
export load_radiation_matrix, contact_matrix, RADIATION_ARTIFACT

# Export arrival process (marked point process; particle-filter only)
export ArrivalProcess, step_arrival_probability, beta_mark, seed_transition,
    pool_redistribute!, pro_rata_move!,
    carries_state, supports_gaussian_filter, assert_gaussian_filter_compatible

# Export measurement model (UKF correct step)
export ObservationNoiseSpec, NegBinomialNoise, PoissonNoise, LogNormalNoise
export SignalObservationSpec, AggregatedSignalSpec, ObservationSpec, observation_mean,
    observation_scale, observation_baseline, observation_gaussian_moments,
    resolve_signal_indices
export n_noise_terms, apply_noise, build_measurement_model

# Export observation likelihood + sampling (particle filter)
export observation_logpdf, sample_observation, build_measurement_logpdf

# Export full dynamics (UKF predict step)
export build_full_dynamics, build_R1

# Export particle-filter builders (reuse the UKF dynamics & measurement model)
export build_pf_dynamics, build_pf_measurement

# Export backtest / forecasting building blocks (vintage-aware backtest run scripts)
export forecast_ensemble, forecast_states, forecast_quantiles, append_latent_spread!,
    append_latent_audit!,
    optimize_hyperparams, asof_series, backtest_forecast_rows, DEFAULT_QS, build_inference,
    prior_unconstrained_variance, prior_unconstrained_mean, DEFAULT_LATENT_VARIANCE,
    DEFAULT_MODEL_RELATIVE_SD
export forecast_sample_rows, write_forecast_samples, FORECAST_SAMPLE_COLUMNS

# Export the trend (integrated Brownian motion) ascertainment model
export TrendAscertainment, TrendAscertainmentSeed, ascertainment_trend_specs,
    ascertainment_trend_sigma_rate, ASCERTAINMENT_TREND_LEVEL, ASCERTAINMENT_TREND_RATE,
    ASCERTAINMENT_TREND_WANDER, DEFAULT_ASCERTAINMENT_TREND_WANDER,
    DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS,
    ASCERTAINMENT_TREND_LEVEL_LOG_SD

# Export online hyperparameter learning (particles carry static hyperparams)
export HyperUpdater, LiuWest, DEFAULT_JITTER_FLOOR_FRACTION,
    LearnedHyperparams, build_learned_hyperparams, build_hyperparam_updater,
    n_learned, hyper_range

# Export data linkage API
export ObservationSchema, n_observations, observation_names, build_observation_schema,
    DataLink, pivot_to_wide, require_complete_grid, build_observations,
    validate_observations, summarize_observations, observation_as_matrix

# Include state filter method types
include("filters.jl")

# Include the augmented-noise ensemble filter (adds methods to LLPF's generics; the ensemble
# analogue of the augmented UKF the UKF path already uses)
include("augmented_enkf.jl")

# Include EKP parameter distribution bridge
include("ekp_parameter_distributions.jl")

# Include configuration schema (@option types; uses the *_gaussian constructors above)
include("config_schema.jl")

# Include the shared transmission-seasonality forcing (uses SeasonalityConfig)
include("seasonality.jl")

# The declining ascertainment path (observations per infection vs model time; all submodels)
include("ascertainment.jl")

# The weekday observation effect (daily cadence; wraps the ascertainment path)
include("day_of_week.jl")

# Peak-anchored initialisation: turns the observed wave peak into an identified `S/N`.
include("initialisation.jl")

# Radiation-model cross-location contact mixing (uses CSV; independent of the rest)
include("radiation_mixing.jl")

# Include hyperparameter inference method types (uses ParameterPriorBundle)
include("hyperparam_inference.jl")

# Include ParamSpec types
include("ParamSpec.jl")

# Include StateLayout schema
include("StateLayout.jl")

# Parameter resolution (get_param_value)
include("param_resolution.jl")

# Build dynamics for latent processes
include("stochastic_dynamics.jl")

# Ascertainment as an integrated-Brownian-motion latent (level + decline-rate states)
include("ascertainment_trend.jl")

# Online hyperparameter learning (Liu-West); reuses latent_dynamics helpers
include("learned_hyperparams.jl")

# Build measurement model (UKF correct step)
include("measurement_model.jl")

# Include core dynamics API
include("core_dynamics.jl")

# Include full dynamics for UKF predict step
include("full_dynamics.jl")

# Include particle-filter builders (reuse UKF dynamics & measurement model)
include("pf_builders.jl")

# Include vintage-aware forecasting building blocks (forecast roll, reopt, as-of)
include("forecast.jl")

# Routine-forecasting-compatible draw-level output (shared by every submodel)
include("forecast_output.jl")

# builds the combined inference engine from the two axes (filter × hyper) and
# the model bundle
include("build_inference.jl")

# Ensemble filter + outer EKP calibration: another `build_inference` method, so it must follow
# the file that defines the function and its shared helpers.
include("ekp_calibration.jl")

# Inference engine construction: multiple dispatch over the two config axes
# (needs the filter/hyper method types + build_inference, all included above).
include("inference_setup.jl")

# Include data linkage API (connect DataFrames to UKF)
include("data_linkage.jl")

end
