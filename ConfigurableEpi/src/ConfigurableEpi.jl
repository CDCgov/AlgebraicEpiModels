"""
    ConfigurableEpi

Configurable compartmental forecasting on AlgebraicEpiMech Petri nets. Four layers:

- `config.jl`: the TOML-backed `RunConfig` and the two inference axes (`filter` × `hyper`).
- `model/`: everything that defines a model — priors, latent processes, the state layout,
  vector fields, the observation model, seasonality, ascertainment, weekday effects,
  initialisation — grouped into an [`EpiModel`](@ref).
- `inference/`: the three engines built by [`build_inference`](@ref) and driven by
  [`fit_forecast!`](@ref): UKF + Optimise, PF + Liu-West, EnKF + EKP.
- `output/`: forecast rolls, quantiles, backtest tables, routine sample output and data linkage.
"""
module ConfigurableEpi

using LinearAlgebra: BLAS, Diagonal, I, diag, cholesky, Symmetric, dot, isdiag, issuccess, logdet, mul!
using Dates, Random
import TOML
import Configurations
using Configurations: @option, from_dict, from_toml, from_kwargs
using AlgebraicPetri, AlgebraicEpiMech
using EnsembleKalmanProcesses: DataMisfitController, EnsembleKalmanProcess, SampleSuccGauss,
    TransformInversion, construct_initial_ensemble, get_u_final, get_Δt, get_ϕ_final, update_ensemble!
using EnsembleKalmanProcesses.ParameterDistributions: ParameterDistribution, combine_distributions,
    constrained_gaussian, get_all_constraints, get_name,
    transform_constrained_to_unconstrained, transform_unconstrained_to_constrained
using Distributions: logpdf, Beta, LogNormal, MvNormal, NegativeBinomial, Normal, Poisson
using LabelledArrays: LArray
using StaticArrays: SVector, SMatrix, setindex
using LowLevelParticleFilters: AbstractKalmanFilter, AdvancedParticleFilter, SignalNames, TrivialParams,
    UnscentedKalmanFilter, correct!, covariance, expweights, forward_trajectory, particles, predict!,
    reset!, sample_measurement, sample_state, state
import LowLevelParticleFilters as LLPF   # this package adds methods to its generics
using PositiveFactorizations: Positive, ldlt!, default_tol, default_blocksize, floattype
using SeeToDee: Rk4
using DataInterpolations: CubicSpline
import DataInterpolations
using Statistics: mean, quantile, std, var
using Optimization: AutoForwardDiff, OptimizationFunction, OptimizationProblem, remake, solve
using OptimizationOptimJL: LBFGS
using OptimizationOptimisers: Adam
import ForwardDiff   # loads Optimization's ForwardDiff extension for AutoForwardDiff
using DataFramesMeta: DataFrame, combine, groupby, nrow, sort!
using DBInterface: close, connect, execute
using DuckDB: DB, register_data_frame, unregister_data_frame

# Config
export @option, from_dict, from_toml, from_kwargs, option_alias
export PriorSpec, build_prior, build_priors, load_prior_specs, load_priors, prior_upper, prior_R_eff_bound
export Durations, SeasonalityConfig, DayOfWeekConfig, NoDayOfWeekConfig, PluginDayOfWeekConfig,
    LearnedDayOfWeekConfig, InputConfig, CountInput, PercentInput
export StateFilter, UKF, PF, EnKF, HyperMethod, Optimise, LiuWest, EKP, DEFAULT_JITTER_FLOOR_FRACTION
export RunIO, RunConfig, validate_run_config, submodel_name, resolve_priors, resolve_prior_specs

# Model
export ParameterDistribution, ParameterPriorBundle, prior_name, positive_gaussian, unit_interval_gaussian,
    unconstrained_gaussian, constrained_values, unconstrained_values, prior_logpdf,
    prior_unconstrained_mean, prior_unconstrained_variance
export ParamSpec, ProcessParamSpec, FixedParam, HyperParam, DerivedParam, AR1ParamSpec, RWParamSpec,
    IntegratedParamSpec, ArrivalProcess, get_param_value, step_arrival_probability, beta_mark,
    seed_transition, pool_redistribute!, pro_rata_move!, carries_state, supports_gaussian_filter,
    assert_gaussian_filter_compatible, advance_arrival!, ou_step, update_single, StochasticUpdate,
    build_stochastic_update
export StateLayout, all_names, ode_names, n_ode_states, n_latent, n_signals, extract_latent
export build_petri_vf, build_unified_vf, make_lvector_constructor, build_R1, build_full_dynamics
export ObservationNoiseSpec, NegBinomialNoise, PoissonNoise, LogNormalNoise, SignalObservationSpec,
    AggregatedSignalSpec, ObservationSpec, n_noise_terms, apply_noise, observation_logpdf,
    sample_observation, observation_mean, observation_scale, observation_baseline,
    observation_gaussian_moments, resolve_signal_indices, build_measurement_model, build_measurement_logpdf
export SeasonalForcing, UnitForcing, CosineForcing, IndoorActivityForcing, N_SEASON_KNOTS,
    build_periodic_curve, year_fraction, load_indoor_activity_climatology,
    validate_indoor_activity_climatology, validate_seasonality, default_seasonal_learned,
    seasonal_forcing_upper_bound, assert_seasonal_learnable, build_seasonal_forcing, build_seasonal_forcings
export AscertainmentPath, ascertainment_at, parse_ascertainment_reference_date, validate_ascertainment,
    build_ascertainment_path, assert_ascertainment_learnable
export TrendAscertainment, TrendAscertainmentSeed, ascertainment_trend_specs, ascertainment_trend_sigma_rate,
    ASCERTAINMENT_TREND_LEVEL, ASCERTAINMENT_TREND_RATE, ASCERTAINMENT_TREND_WANDER,
    DEFAULT_ASCERTAINMENT_TREND_WANDER, DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS, ASCERTAINMENT_TREND_LEVEL_LOG_SD
export DOW_LEARNED_NAMES, DOW_DAY_ABBREVIATIONS, DOW_HELMERT, dow_helmert_prior_sd, dow_log_effects,
    day_of_week_multipliers, day_of_week_weight, remove_day_of_week_effect, DayOfWeekModifier,
    DayOfWeekDispersion, HyperPhi, validate_day_of_week, estimate_day_of_week_effects,
    prepare_day_of_week_history, build_day_of_week_observation, day_of_week_learned_names,
    assert_day_of_week_learnable, day_of_week_report_rows
export RK4_STABILITY_LIMIT, RK4_WARN_LAMBDA_H, max_transition_rate, required_supersample,
    assert_integration_stable, initial_infection_state, reporting_delay_days, prevalence_peak_lag_days,
    anchor_is_exact, find_observed_peak, carry_susceptible, peak_anchored_susceptible_fraction
export load_radiation_matrix, contact_matrix
export COMPARTMENTAL_MODELS, EpiModel, initial_state, learned_names

# Inference
export AugmentedEnsembleKalmanFilter
export LearnedHyperparams, build_learned_hyperparams, build_hyperparam_updater, n_learned, hyper_range
export DEFAULT_OPTIMISER_STAGES, optimiser_stages, optimize_hyperparams, positive_cholesky!, marginal_loglik
export InferenceEngine, EngineSettings, UKFOptimiseEngine, PFLiuWestEngine, EnKFEKPEngine,
    build_inference, fit_forecast!, build_pf_dynamics, build_pf_measurement,
    DEFAULT_LATENT_VARIANCE, DEFAULT_MODEL_RELATIVE_SD

# Output
export DEFAULT_QS, forecast_ensemble, forecast_states, forecast_quantiles, append_latent_spread!,
    append_latent_audit!, asof_series, backtest_forecast_rows
export FORECAST_SAMPLE_COLUMNS, forecast_sample_rows, write_forecast_samples
export ObservationSchema, n_observations, observation_names, build_observation_schema, DataLink,
    pivot_to_wide, require_complete_grid, build_observations

"Compartmental constructors selectable by name (all carry S, E and I)."
const COMPARTMENTAL_MODELS = Dict(:SEIRS => SEIRS, :SEIR => SEIR, :SEIS => SEIS, :SEI => SEI)

include("model/priors.jl")
include("config.jl")
include("model/layout.jl")
include("model/processes.jl")
include("model/dynamics.jl")
include("model/observation.jl")
include("model/seasonality.jl")
include("model/ascertainment.jl")
include("model/day_of_week.jl")
include("model/initialisation.jl")
include("model/radiation_mixing.jl")
include("model/epi_model.jl")

include("inference/augmented_enkf.jl")
include("inference/learned_hyperparams.jl")
include("inference/optimise.jl")
include("inference/engine.jl")
include("inference/ukf.jl")
include("inference/particle_filter.jl")
include("inference/ensemble.jl")

include("output/forecast.jl")
include("output/samples.jl")
include("output/data_linkage.jl")

end
