# ConfigurableEpi

Configurable compartmental forecasting on [AlgebraicEpiMech](../AlgebraicEpiMech) Petri nets.
A model is a Petri net with a rate function, a set of stochastic latent drivers, and an observation model; an inference engine fits it to a count series and forecasts ahead.
Three engines are supported, selected by two run-config axes:

  | `[filter.<name>]` | `[hyper.<name>]` | Engine                                                                                             |
  | ----------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
  | `ukf`             | `optimise`       | Unscented Kalman filter; static parameters by maximising the differentiable marginal log-posterior |
  | `pf`              | `liu_west`       | Bootstrap particle filter; static parameters learned online in the cloud (Liu-West)                |
  | `enkf`            | `ekp`            | Augmented ensemble Kalman filter; static parameters by outer ensemble Kalman inversion             |

## Layout

```
src/
  ConfigurableEpi.jl      module, imports, exports
  config.jl               TOML-backed RunConfig and the @option types for both axes
  model/                  everything that defines a model
    priors.jl             scalar EKP priors, ParameterPriorBundle
    layout.jl             StateLayout: where compartments, observation states and latents live
    processes.jl          AR1 / random-walk / integrated latents, ArrivalProcess, StochasticUpdate
    dynamics.jl           build_petri_vf, build_full_dynamics (one filter step)
    observation.jl        noise families, observation specs, measurement / likelihood builders
    seasonality.jl        transmission forcings (cosine, indoor-activity climatology, none)
    ascertainment.jl      declining ascertainment path and the trend (integrated BM) variant
    day_of_week.jl        weekday observation effect for daily counts
    initialisation.jl     RK4 stability guard, count inversion, peak-anchored susceptibles
    radiation_mixing.jl   cross-location contact matrix
    epi_model.jl          EpiModel: the bundle handed to build_inference
  inference/
    engine.jl             shared machinery, build_inference / fit_forecast! contract
    ukf.jl, particle_filter.jl, ensemble.jl   the three engines
    learned_hyperparams.jl, optimise.jl, augmented_enkf.jl
  output/
    forecast.jl           forecast rolls, quantiles, latent diagnostics, backtest tables
    samples.jl            routine-forecasting samples.parquet output
    data_linkage.jl       DataFrame -> observation vectors
```

## Sketch

```julia
using ConfigurableEpi, AlgebraicEpiMech, Catlab

pn = attach_observation(dom(create_model(OnePopulationTyping(), SEIR())), AtEvent(:transmission); n_stages = 2)
rates(latent, hyper, t) = (transmission_S_I = hyper.gamma * hyper.R0 * latent.Rt / hyper.N,)
vf! = build_petri_vf(pn, rates; defaults = (E_to_I = 1 / 2, I_to_R = 1 / 1.5, O_transmission_1_to_O_transmission_2 = 1 / 6.6))

drivers = (AR1ParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.05), mu = 1.0, tau = 30.0, sigma = 0.05),)
layout = StateLayout(pn, drivers; signal_names = (:reports,))
stochastic = build_stochastic_update(layout, drivers)

model = EpiModel(;
    vectorfield! = vf!, layout, stochastic,
    observation = (SignalObservationSpec(1, NegBinomialNoise(phi = (l, h, t) -> h.phi); mean_modifier = 0.005),),
    hyperparams = (gamma = 1 / 1.5, R0 = 2.0, N = 1.0e7, phi = 140.0),
    priors = (R0 = positive_gaussian(:R0, 2.0, 0.75),),
    initial_state = x0,   # a vector, or hyperparams -> vector
)

engine = build_inference(UKF(), Optimise(), model; dt = 7.0, n_ahead = 4)
result = fit_forecast!(engine, weekly_counts, 1)
result.quantiles       # [horizon, quantile] at DEFAULT_QS
result.fitted_means    # fitted observation means over the series
result.summary         # (parameter, statistic, value) estimates and diagnostics
```

With a run config, `build_inference(validate_run_config(from_toml(RunConfig, "run.toml")), model)` reads the axes and the cadence from the TOML:

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

[epi.basic_seir]          # parsed by the submodel's own @option type
```

## Conventions

- Latent coefficients live in the state vector in the **unconstrained** chart of their `init` prior and are read back constrained through `stochastic.extract`; every observation spec sees them constrained.
- Model time `t` is days since the first observation.
  Seasonality and ascertainment are anchored to the calendar once, at build time.
- Observations are given on the regular `step_days` grid, with `missing` where there is no observation.
  The filter predicts through a missing slot without correcting, so a reporting gap does not compress time.
- A rate function receives `(latent, hyperparams, t)`; a noise or mean parameter is a `Real` or a function of the same three arguments.
