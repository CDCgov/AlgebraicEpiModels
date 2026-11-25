# ConfigurableEpi

ConfigurableEpi does configurable inference and forecasting for compartmental models over data streams.
For ConfigurableEpi, a model is a vectorfield for some dynamical model combined with a set of stochastic latent drivers and a link between observable components and data.
The inference "engine" is defined as a combination of two choices:

- A filtering approach.
  This defines how the distribution over the latent state of the model updates conditional on the new data arriving in the stream and the current static hyperparameter estimate.
- A static hyperparameter estimation approach.
  This defines how a filtering pass, or filtering passes, are used to infer hyperparameters.

## Connection to AlgebraicEpiMech

At the moment, ConfigurableEpi is designed to work with AlgebraicEpiMech insofar as a petri net representation of the dynamics is the entry point to constructing a valid model.
In principle, other petri net or vectorfield functions could be used as an entry point, however, AlgebraicEpiMech comes with an enforced low level naming convention for rate parameters which simplifies model construction.

## Engines

Three engines are currently supported:

  | Filter             | Hyperparameters | Engine                                                                                                  |
  | ------------------ | --------------- | ------------------------------------------------------------------------------------------------------- |
  | `UKF()`            | `Optimise()`    | Unscented Kalman filter; hyperparameters by maximising the differentiable marginal log-posterior.       |
  | `PF(n_particles)`  | `LiuWest()`     | Bootstrap particle filter; hyperparameters learned online in the particle cloud by the Liu-West kernel. |
  | `EnKF(n_ensemble)` | `EKP(...)`      | Augmented ensemble Kalman filter; hyperparameters by outer ensemble Kalman inversion.                   |

Other pairings are currently rejected.
Only the particle filter handles jump drivers such as `ArrivalProcess`.

Every engine forecasts by simulation.
The particle filter and the EnKF roll their own members forward; the UKF draws `n_draws` states jointly from its filtered Gaussian at the forecast origin.
Each draw is then simulated through the model's process noise and its exact observation distribution, so the forecast carries the latent's uncertainty and its covariance with the compartments.
The UKF's `obs_jitter` only keeps the filter's covariance full-rank, so the forecast leaves it out.
Two repairs keep every UKF draw valid without changing its distribution:

- negative eigenvalues of the origin covariance, usually rounding error, are set to zero, which adds no variance;
- a compartment drawn below zero is raised to `STATE_FLOOR`, the floor the dynamics applies before every step anyway.

Rejecting or truncating draws instead would change the distribution, and could stall a forecast at a trough where many draws fall below zero.
The summary reports `origin_cov_min_eigenvalue`, `origin_cov_clamped_dims` and `origin_draws_floored_frac` under `ukf`.

## The model contract

An `EpiModel` bundles everything an engine needs:

```julia
using ConfigurableEpi, AlgebraicEpiMech, Catlab

pn = attach_observation(dom(create_model(OnePopulationTyping(), SEIR())), AtEvent(:transmission); n_stages = 2)
rates(latent, hyper, t) = (transmission_S_I = hyper.gamma * hyper.R0 * latent.Rt / hyper.N,)
vf! = build_petri_vf(pn, rates; defaults = (E_to_I = 1 / 2, I_to_R = 1 / 1.5, O_transmission_1_to_O_transmission_2 = 1 / 6.6))

drivers = (AR1ParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.05), mu = 1.0, tau = 30.0, sigma = 0.05),)
layout = StateLayout(pn, drivers; signal_names = (:reports,))
stochastic = build_stochastic_update(layout, drivers)

N = 1.0e7
seed = (S = N - 200.0, E = 100.0, I = 100.0, R = 0.0, O_transmission_1 = 0.0, O_transmission_2 = 0.0)
x0 = vcat([seed[n] for n in ode_names(layout)], collect(stochastic.to_unconstrained((Rt = 1.0,))))

model = EpiModel(;
    vectorfield! = vf!, layout, stochastic,
    observation = (SignalObservationSpec(1, NegBinomialNoise(phi = (l, h, t) -> h.phi); mean_modifier = 0.005),),
    hyperparams = (gamma = 1 / 1.5, R0 = 2.0, N = N, phi = 140.0),
    priors = (R0 = positive_gaussian(:R0, 2.0, 0.75),),
    initial_state = x0,  # a vector, or hyperparams -> vector
)

engine = build_inference(UKF(), Optimise(), model; dt = 7.0, n_ahead = 4)
result = fit_forecast!(engine, weekly_counts, 1)
result.quantiles     # [horizon, quantile] at DEFAULT_QS
result.samples       # [horizon, draw], the predictive draws behind the quantiles
result.fitted_means  # fitted observation means over the series
result.summary       # (parameter, statistic, value) estimates and diagnostics
```

- **Rates.** `build_petri_vf(pn, rates; defaults)` gives the vector field.
  `rates(latent, hyperparams, t)` returns the time-varying transition rates by name; `defaults` fixes the rest.
- **Drivers.** `AR1ParamSpec`, `RWParamSpec` and `IntegratedParamSpec` are continuous latents, and `ArrivalProcess` is a marked jump process.
  `StateLayout` places compartments, observation accumulators and latents in one state vector, and `build_stochastic_update` builds their noise update.
- **Observation.** `SignalObservationSpec` reads a signal's accumulator, scales it by `mean_modifier` and adds noise (`NegBinomialNoise`, `PoissonNoise` and others).
  Ascertainment trends, day-of-week effects and seasonal forcing plug in here.
  With several observations each slot is a vector, and a `missing` component (a signal absent that slot) is skipped by every filter.
- **Hyperparameters.** `hyperparams` holds every fixed value and the starting value of each learned one; `priors` names the learned subset.

[Inference engines](../examples/inference_engines.md) runs all three engines on simulated data, and [Low-level filtering](../examples/low_level_filtering.md) wires the same building blocks into LowLevelParticleFilters directly.

## Inference configuration

The engine pairing and its cadence can be read from TOML: `build_inference(validate_inference_config(from_toml(InferenceConfig, "inference.toml")), model)`.

```toml
n_ahead = 4
step_days = 7
learn_params = ["R0_baseline"]

[filter.pf]
n_particles = 2000

[hyper.liu_west]
discount = 0.97

[priors.phi]
mean = 50.0
std = 10.0
```

The `[filter.<name>]` and `[hyper.<name>]` tables select the engine (`ukf`, `pf`, `enkf`; `optimise`, `liu_west`, `ekp`) and set its options, and `[priors]` overrides a model's default priors by name.

`InferenceConfig` holds only what the package reads.
Run settings such as data paths, the observation scale, burn-in and the forecast schedule belong to the run script, which embeds the config as one field of its own `@option` type:

```julia
@option struct MyRun
    data::String
    burnin_observations::Int
    inference::InferenceConfig   # an [inference] table in the run's TOML
end
```

## Conventions

- Latent coefficients live in the state vector in the **unconstrained** chart of their `init` prior and are read back constrained through `stochastic.extract`; every observation spec sees them constrained.
- Model time `t` is days since the first observation.
  Seasonality and ascertainment are anchored to the calendar once, at build time.
- Observations are given on the regular `step_days` grid, with `missing` where there is no observation.
  The filter predicts through a missing slot without correcting, so a reporting gap does not compress time.
- A rate function receives `(latent, hyperparams, t)`; a noise or mean parameter is a `Real` or a function of the same three arguments.

## Source layout

```
src/
  ConfigurableEpi.jl      module, imports, exports
  model/                  everything that defines a model: TOML-backed options and InferenceConfig,
                          priors, state layout, latent processes,
                          dynamics, observation, seasonality, ascertainment, day-of-week effects,
                          initialisation, radiation mixing, and EpiModel itself
  inference/              the shared engine machinery (build_inference / fit_forecast!) and the
                          UKF, particle filter and ensemble engines
  output/                 forecast rolls and quantiles, backtest tables, sample output, data linkage
```

The full list of exports is in the [API reference](../api/configurableepi.md).
