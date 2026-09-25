# ConfigurableEpi

ConfigurableEpi does configurable compartmental forecasting on [AlgebraicEpiMech](algebraicepimech.md) Petri nets.
A model is a Petri net with a rate function, a set of stochastic latent drivers and an observation model.
An inference engine fits it to a count series and forecasts ahead.

## Engines

Three engines are supported, each a pairing of a state filter with a method for the static hyperparameters:

  | Filter             | Hyperparameters | Engine                                                                                                  |
  | ------------------ | --------------- | ------------------------------------------------------------------------------------------------------- |
  | `UKF()`            | `Optimise()`    | Unscented Kalman filter; hyperparameters by maximising the differentiable marginal log-posterior.       |
  | `PF(n_particles)`  | `LiuWest()`     | Bootstrap particle filter; hyperparameters learned online in the particle cloud by the Liu-West kernel. |
  | `EnKF(n_ensemble)` | `EKP(...)`      | Augmented ensemble Kalman filter; hyperparameters by outer ensemble Kalman inversion.                   |

Other pairings are rejected.
Only the particle filter handles jump drivers such as `ArrivalProcess`.

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
result.fitted_means  # fitted observation means over the series
result.summary       # (parameter, statistic, value) estimates and diagnostics
```

- **Rates.** `build_petri_vf(pn, rates; defaults)` gives the vector field.
  `rates(latent, hyperparams, t)` returns the time-varying transition rates by name; `defaults` fixes the rest.
- **Drivers.** `AR1ParamSpec`, `RWParamSpec` and `IntegratedParamSpec` are continuous latents, and `ArrivalProcess` is a marked jump process.
  `StateLayout` places compartments, observation accumulators and latents in one state vector, and `build_stochastic_update` builds their noise update.
- **Observation.** `SignalObservationSpec` reads a signal's accumulator, scales it by `mean_modifier` and adds noise (`NegBinomialNoise`, `PoissonNoise` and others).
  Ascertainment trends, day-of-week effects and seasonal forcing plug in here.
- **Hyperparameters.** `hyperparams` holds every fixed value and the starting value of each learned one; `priors` names the learned subset.

[Inference engines](../examples/inference_engines.md) runs all three engines on simulated data, and [Low-level filtering](../examples/low_level_filtering.md) wires the same building blocks into LowLevelParticleFilters directly.

## Run configuration

A run can be configured from TOML: `build_inference(validate_run_config(from_toml(RunConfig, "run.toml")), model)` reads the engine pairing and cadence from the file.

```toml
n_ahead = 4
step_days = 7
burnin_observations = 12

[io]
data = "triangle.csv"
model_id = "seir"
forecast_df = "forecast.csv"
loc = "ny"

[input.counts]

[filter.pf]
n_particles = 2000

[hyper.liu_west]
discount = 0.97

[epi.basic_seir]  # parsed by the model's own @option type
```

The `[filter.<name>]` and `[hyper.<name>]` tables select the engine (`ukf`, `pf`, `enkf`; `optimise`, `liu_west`, `ekp`) and set its options.
`[input.counts]` or `[input.percent]` sets the observation scale, and `[priors]` overrides a model's default priors by name.

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
  config.jl               TOML-backed RunConfig and the @option types for both axes
  model/                  everything that defines a model: priors, state layout, latent processes,
                          dynamics, observation, seasonality, ascertainment, day-of-week effects,
                          initialisation, radiation mixing, and EpiModel itself
  inference/              the shared engine machinery (build_inference / fit_forecast!) and the
                          UKF, particle filter and ensemble engines
  output/                 forecast rolls and quantiles, backtest tables, sample output, data linkage
```

The full list of exports is in the [API reference](../api/configurableepi.md).
