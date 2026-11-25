# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- ConfigurableEpi: `InferenceConfig` replaces `RunConfig`.
  It holds only the settings the package reads (`filter`, `hyper`, `step_days`, `supersample`, `n_ahead`, `n_draws`, `learn_params`, `priors`), and a run script embeds it in its own config.
  `build_inference`, `resolve_priors` and `resolve_prior_specs` take an `InferenceConfig`.
- ConfigurableEpi: `StateFilter` and `HyperMethod` are abstract supertypes of the option structs; `StateFilterConfig` and `HyperMethodConfig` are the `Union`s for `@option` fields.
- ConfigurableEpi: `config.jl` moved to `src/model/config.jl`.
- ConfigurableEpi: `forecast_ensemble` always returns `samples[horizon, draw, observation]`, keeping the observation axis for one observation.
  `ForecastResult` still drops it for a one-observation model.
- ConfigurableEpi: UKF + Optimise forecasts by drawing `n_draws` states jointly from the filtered origin Gaussian and simulating each through the model's own process and observation noise, as the PF and EnKF engines do.
  `samples` is no longer `nothing` for the UKF, the quantiles are those of the draws, and any observation kind can be forecast.
  The `rng` keyword now seeds these draws.

### Removed

- ConfigurableEpi: `observation_gaussian_moments` and `forecast_states`, the analytic UKF forecast.
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
