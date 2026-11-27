# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- ConfigurableEpi: initialisation strategies `SlowModeInitialisation`, `PeakAtStartInitialisation` and `PeakFromHistoryInitialisation` (subtypes of `InitialisationStrategy`; `InitialisationConfig` is the `Union` for `@option` fields), checked by `validate_initialisation`.
  `initial_seed(strategy, ::SeedContext)` returns each strategy's susceptible fraction and stage fills, and `chain_stage_seed` fills any further Erlang chain from the same infection stream.
- ConfigurableEpi: replay initialisation.
  `replay_state` integrates a `ReplayModel` (the vector field, the same field with infection off from `build_replay_vf`, the layout and the `InfectionRoutes` from `infection_routes`) over a `BinnedIncidence` history, and `replay_scale_moments` gives its moments under an uncertain incidence level.
  `deterministic_spinup` integrates a model along a given latent path.
- ConfigurableEpi: warm start from the signal before the fit.
  `signal_incidence`, `splice_incidence`, `closing_incidence`, `history_closure` and `log_growth_rate` build and check the recent incidence; `total_infection_rate`, `matched_transmission_multiplier` and `closed_loop_start` set the transmission multiplier and infected compartments that continue it.
- ConfigurableEpi: age structure.
  `AgeMixing` holds a population on `AgeBand`s (checked by `validate_age_bands`) and its contact matrices by setting; `load_age_mixing` regroups base-grid tables onto a model's bands and `load_setting_contacts` reads layered contact matrices, and `contact_spectral_radius` is the radius over all settings.
  `SchoolCalendar`, `load_school_calendar`, `AnchoredCalendar` and `assert_calendar_covers` switch a school layer on model time.
- ConfigurableEpi: joint priors.
  A `PriorCoupling` (`Independent()` by default) sets how a `ParameterPriorBundle` combines its scalar priors, and `RtNBVariancePrior` couples the Rt scale and the negative-binomial dispersion.
  `EpiModel(; joint_prior)` and `with_joint_prior` attach one; `prior_moments` and `sample_prior` give the bundle's moments and draws.
  PF + Liu-West starts its learned slots from the prior and its kernel from `prior_moments`; EKP refuses a coupled prior.
- ConfigurableEpi: `normalise_config_dict` gives an option table built in Julia string keys for `from_dict`.

### Changed

- ConfigurableEpi: `InferenceConfig` replaces `RunConfig`.
  It holds only the settings the package reads (`filter`, `hyper`, `step_days`, `supersample`, `n_ahead`, `n_draws`, `learn_params`, `priors`), and a run script embeds it in its own config.
  `build_inference`, `resolve_priors` and `resolve_prior_specs` take an `InferenceConfig`.
- ConfigurableEpi: `StateFilter` and `HyperMethod` are abstract supertypes of the option structs; `StateFilterConfig` and `HyperMethodConfig` are the `Union`s for `@option` fields.
- ConfigurableEpi: `config.jl` moved to `src/model/config.jl`.
- ConfigurableEpi: `radiation_mixing.jl` became `src/model/mixing.jl`, which also holds the age loaders, and `load_radiation_matrix` reads its table with the same reader.
- ConfigurableEpi: `forecast_ensemble` always returns `samples[horizon, draw, observation]`, keeping the observation axis for one observation.
  `ForecastResult` still drops it for a one-observation model.
- ConfigurableEpi: UKF + Optimise forecasts by drawing `n_draws` states jointly from the filtered origin Gaussian and simulating each through the model's own process and observation noise, as the PF and EnKF engines do.
  `samples` is no longer `nothing` for the UKF, the quantiles are those of the draws, and any observation kind can be forecast.
  The `rng` keyword now seeds these draws.
- ConfigurableEpi: the integrator is a choice, `Rk4` by default.
  `build_full_dynamics`, `EngineSettings`, `build_inference`, `deterministic_spinup`, `replay_state` and `closed_loop_start` take an `integrator::Type{<:SeeToDee.AbstractIntegrator}`, built as `integrator(f, dt; supersample)`.
  `InferenceConfig` names one with `integrator = "Heun"`, say, resolved by `integrator_type`.
- ConfigurableEpi: `assert_integration_stable` and `required_supersample` take an `integrator` and check it against its `stability_limit`, or a `limit` passed for an integrator without a known one.
  `stability_limit(Rk4)` and `STABILITY_HEADROOM` replace `RK4_STABILITY_LIMIT` and `RK4_WARN_LAMBDA_H`, so RK4's warning starts at `2.785 / 1.4 ≈ 1.99` rather than 2.
- ConfigurableEpi: the EnKF engine runs LowLevelParticleFilters' augmented `EnsembleKalmanFilter{true, true}`, which needs LowLevelParticleFilters 3.32.
  A slot with absent components still corrects on the observed ones only.
- ConfigurableEpi: `max_transition_rate`, `required_supersample` and `assert_integration_stable` take an `IntegrationBounds` (durations, stage counts, `R_eff_max` and any `extra_rates` outside the durations) in place of `(dur, n_E, n_I; R_eff_max)`.

### Removed

- ConfigurableEpi: `observation_gaussian_moments` and `forecast_states`, the analytic UKF forecast.
- ConfigurableEpi: `AugmentedEnsembleKalmanFilter`; use LowLevelParticleFilters' `EnsembleKalmanFilter{true, true}`.
- ConfigurableEpi: `RunConfig`, `RunIO`, `InputConfig`, `CountInput`, `PercentInput`, `validate_run_config` and `submodel_name`.
  Use `validate_inference_config` for the inference settings; run settings (data paths, input scale, burn-in, origin mode, schedule) belong to the run script.

## v0.3.0

### Added

- Conditional Python and R formatter and linter configuration in generated `panache.toml` files
- Mustache templating for generated GitHub pre-commit action and workflow resources
- Conditional R cache configuration and `setup-r` workflow setup for R projects

### Changed

- Added reusable templated-file handling to the CLI resource tasks
- Preserved GitHub Actions expressions while rendering Mustache templates
- Rename from "actions" to "tasks"
- Move from RepoCLI to Run
- Tasks are not "done" vs. not; tasks are now done, ready, or blocked

### Removed

- Interactive option (instead allow for patching `._choose_repo_root()`, `._choose_features()`, `._choose_tasks()`)
