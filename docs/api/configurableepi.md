# ConfigurableEpi API {#ConfigurableEpi-API}

Reference for the exported API of `ConfigurableEpi`, generated from docstrings and grouped by layer.

<a id='ConfigurableEpi.ConfigurableEpi'></a> <a id='ConfigurableEpi.ConfigurableEpi-1'></a> **`ConfigurableEpi.ConfigurableEpi`** &mdash; *Module*.

```julia
ConfigurableEpi
```

Configurable compartmental forecasting on AlgebraicEpiMech Petri nets.
Four layers:

- `config.jl`: the TOML-backed `RunConfig` and the two inference axes (`filter` × `hyper`).
- `model/`: everything that defines a model — priors, latent processes, the state layout, vector fields, the observation model, seasonality, ascertainment, weekday effects, initialisation — grouped into an [`EpiModel`](configurableepi.md#ConfigurableEpi.EpiModel).
- `inference/`: the three engines built by \[`build_inference`\](configurableepi.md#ConfigurableEpi.build_inference-Tuple{StateFilter, HyperMethod, EpiModel}) and driven by [`fit_forecast!`](configurableepi.md#ConfigurableEpi.fit_forecast!): UKF + Optimise, PF + Liu-West, EnKF + EKP.
- `output/`: forecast rolls, quantiles, backtest tables, routine sample output and data linkage.

## Configuration {#Configuration}

<a id='ConfigurableEpi.DEFAULT_JITTER_FLOOR_FRACTION'></a> <a id='ConfigurableEpi.DEFAULT_JITTER_FLOOR_FRACTION-1'></a> **`ConfigurableEpi.DEFAULT_JITTER_FLOOR_FRACTION`** &mdash; *Constant*.

```julia
DEFAULT_JITTER_FLOOR_FRACTION
```

Minimum Liu-West jitter variance per parameter as a fraction of the prior's unconstrained variance, so a collapsed cloud can re-expand rather than freeze.

<a id='ConfigurableEpi.CountInput'></a> <a id='ConfigurableEpi.CountInput-1'></a> **`ConfigurableEpi.CountInput`** &mdash; *Type*.

```julia
CountInput()                                          # [input.counts]
PercentInput(; annual_rate_per_100)                   # [input.percent]
```

The observation scale.
`counts` are already on the model's count scale.
`percent` observations are percentages of a denominator series (all visits, say) whose annual volume is `annual_rate_per_100` per 100 population; the runner converts them to counts with the location's population and the observation interval.

<a id='ConfigurableEpi.Durations'></a> <a id='ConfigurableEpi.Durations-1'></a> **`ConfigurableEpi.Durations`** &mdash; *Type*.

```julia
Durations(; latent = 2.0, infectious = 1.5, immunity = 180.0, obs_progression = 6.6)
```

Mean durations in days.
Erlang stage rates are `n_stages / mean`.
Because observation is attached to the infection event, the reporting delay is the observation chain alone: `(n_obs_stages - 1) * obs_progression` (the terminal stage is the reset accumulator).

<a id='ConfigurableEpi.EKP'></a> <a id='ConfigurableEpi.EKP-1'></a> **`ConfigurableEpi.EKP`** &mdash; *Type*.

```julia
EKP(; n_ensemble, iterations, burnin_iterations, reopt_interval = 1, inflation = 0.0,
    window_length = nothing, warm_start = true, threads = false)   # [hyper.ekp]
```

Outer ensemble Kalman inversion of the static parameters, each candidate scored by a complete inner filter replay.
`burnin_iterations` at the first origin, `iterations` when warm-starting from the previous ensemble.
`threads` parallelises candidate replays (not together with `filter.enkf.threads`).

<a id='ConfigurableEpi.EnKF'></a> <a id='ConfigurableEpi.EnKF-1'></a> **`ConfigurableEpi.EnKF`** &mdash; *Type*.

```julia
EnKF(; n_ensemble, inflation = 1.0, threads = false)  # [filter.enkf]
```

Augmented ensemble Kalman filter.
`inflation >= 1` multiplies the ensemble spread after each propagation; `threads` parallelises member propagation.

<a id='ConfigurableEpi.LiuWest'></a> <a id='ConfigurableEpi.LiuWest-1'></a> **`ConfigurableEpi.LiuWest`** &mdash; *Type*.

```julia
LiuWest(; discount = 0.95, jitter_floor_fraction = DEFAULT_JITTER_FLOOR_FRACTION,
        forgetting_memory_days = Dict(), replay_on_revision = true)   # [hyper.liu_west]
```

Learn static hyperparameters online in the particle cloud by the Liu-West shrink-jitter kernel.
`forgetting_memory_days` (parameter => days) adds Kulhavý forgetting toward the prior, merged over the model's own defaults.
`replay_on_revision` tells a backtest runner whether to rebuild the filter when already-assimilated data are revised.

<a id='ConfigurableEpi.NoDayOfWeekConfig'></a> <a id='ConfigurableEpi.NoDayOfWeekConfig-1'></a> **`ConfigurableEpi.NoDayOfWeekConfig`** &mdash; *Type*.

```julia
NoDayOfWeekConfig()                                   # [day_of_week.none]
PluginDayOfWeekConfig(; window_days = 182, exclude_recent_days = 14, fit_policy = "per_origin")
LearnedDayOfWeekConfig()                              # [day_of_week.learned]
```

Weekday observation effect for daily counts: none, multipliers and per-weekday extra dispersion estimated from history (`fit_policy` is `per_origin` or `first_vintage`), or six zero-sum Helmert coordinates learned by Liu-West.

<a id='ConfigurableEpi.Optimise'></a> <a id='ConfigurableEpi.Optimise-1'></a> **`ConfigurableEpi.Optimise`** &mdash; *Type*.

```julia
Optimise(; reopt_interval = 1, maxiters = 50, maxiters_burnin = 300, window_length = nothing,
         warm_start = true)                          # [hyper.optimise]
```

Maximise the filter's marginal log-posterior over data replays every `reopt_interval` origins.
`maxiters_burnin` caps every optimiser stage on the first (and any cold-started) optimisation, `maxiters` on warm-started ones.
`window_length` scores only the most recent observations from a rolling filter checkpoint.
`warm_start = false` restarts from the configured values each time.

<a id='ConfigurableEpi.PF'></a> <a id='ConfigurableEpi.PF-1'></a> **`ConfigurableEpi.PF`** &mdash; *Type*.

```julia
PF(; n_particles, threads = true)                    # [filter.pf]
```

Bootstrap particle filter.
`threads` parallelises particle propagation; a seeded run reproduces exactly for a fixed thread count.

<a id='ConfigurableEpi.PriorSpec'></a> <a id='ConfigurableEpi.PriorSpec-1'></a> **`ConfigurableEpi.PriorSpec`** &mdash; *Type*.

```julia
PriorSpec(; mean, std, constraint = "positive")
```

One parameter's prior as written in TOML; `constraint` is `positive`, `unit_interval` or `unconstrained`.
`build_prior` turns it into an EKP `ParameterDistribution`.

<a id='ConfigurableEpi.RunConfig'></a> <a id='ConfigurableEpi.RunConfig-1'></a> **`ConfigurableEpi.RunConfig`** &mdash; *Type*.

```julia
RunConfig
```

One run's TOML: the `[io]` block, the forecast horizon and draw count, the observation cadence (`step_days`, `burnin_observations`, `drop_recent_observations`), the `[input.<name>]` scale, the inference axes `[filter.<name>]` and `[hyper.<name>]`, the `[epi.<submodel>]` block (kept opaque for the submodel to parse with its own `@option` type) and `[priors]` overriding the submodel's defaults per key.
`origin_mode` is `sequential`, `independent` or `forked` (PF + Liu-West only).

<a id='ConfigurableEpi.RunIO'></a> <a id='ConfigurableEpi.RunIO-1'></a> **`ConfigurableEpi.RunIO`** &mdash; *Type*.

```julia
RunIO(; data, model_id, forecast_df, loc, locations = String[])   # [io]
```

Per-run I/O labels: the reporting-triangle `data` path, the `model_id` and `forecast_df` output labels, the run `loc`, and the ordered `locations` of a joint multi-location model.

<a id='ConfigurableEpi.SeasonalityConfig'></a> <a id='ConfigurableEpi.SeasonalityConfig-1'></a> **`ConfigurableEpi.SeasonalityConfig`** &mdash; *Type*.

```julia
SeasonalityConfig(; mode = "indoor_activity", kappa = 0.8, fallback = "us")
```

Transmission seasonality: `mode` is `indoor_activity` (empirical location curve scaled by `kappa ∈ [0, 1]`, the seasonally forced share of transmission), `cosine` (learned annual harmonic) or `none`.
`fallback` (`us`, `none`, `error`) covers a location the climatology lacks.

<a id='ConfigurableEpi.UKF'></a> <a id='ConfigurableEpi.UKF-1'></a> **`ConfigurableEpi.UKF`** &mdash; *Type*.

```julia
UKF(; obs_jitter = 1.0)                               # [filter.ukf]
```

Unscented Kalman filter.
`obs_jitter` scales the accumulator process-noise whisker that keeps the smoother covariance full-rank.

<a id='ConfigurableEpi.build_priors-Tuple{Any}'></a> <a id='ConfigurableEpi.build_priors-Tuple{Any}-1'></a> **`ConfigurableEpi.build_priors`** &mdash; *Method*.

```julia
build_priors(specs) -> NamedTuple
```

`name => PriorSpec` pairs to a `name => ParameterDistribution` NamedTuple.

<a id='ConfigurableEpi.load_prior_specs-Tuple{Any}'></a> <a id='ConfigurableEpi.load_prior_specs-Tuple{Any}-1'></a> **`ConfigurableEpi.load_prior_specs`** &mdash; *Method*.

```julia
load_prior_specs(path) -> Dict{String, PriorSpec}
load_priors(path) -> NamedTuple
```

Read a flat priors TOML (one `[name]` table per parameter).

<a id='ConfigurableEpi.option_alias-Tuple{Any}'></a> <a id='ConfigurableEpi.option_alias-Tuple{Any}-1'></a> **`ConfigurableEpi.option_alias`** &mdash; *Method*.

```julia
option_alias(x) -> String
```

The TOML alias of an `@option` value, e.g. `option_alias(UKF()) == "ukf"`.

<a id='ConfigurableEpi.prior_R_eff_bound-Tuple{AbstractDict{String, PriorSpec}}'></a> <a id='ConfigurableEpi.prior_R_eff_bound-Tuple{AbstractDict{String, PriorSpec}}-1'></a> **`ConfigurableEpi.prior_R_eff_bound`** &mdash; *Method*.

```julia
prior_R_eff_bound(priors; chi_max = 1.55, probability = 0.95) -> Float64
```

Conservative bound on `R_eff = R0_baseline * Rt * chi * S/N` from marginal prior quantiles (`S/N <= 1`; a missing prior contributes 1).
Pass `chi_max = seasonal_forcing_upper_bound(...)`.

<a id='ConfigurableEpi.prior_upper-Tuple{PriorSpec}'></a> <a id='ConfigurableEpi.prior_upper-Tuple{PriorSpec}-1'></a> **`ConfigurableEpi.prior_upper`** &mdash; *Method*.

```julia
prior_upper(spec::PriorSpec; probability = 0.99) -> Float64
```

Upper quantile of a prior on its constrained scale (moment-matched lognormal for `positive`).

<a id='ConfigurableEpi.resolve_prior_specs-Tuple{RunConfig, Any}'></a> <a id='ConfigurableEpi.resolve_prior_specs-Tuple{RunConfig, Any}-1'></a> **`ConfigurableEpi.resolve_prior_specs`** &mdash; *Method*.

```julia
resolve_prior_specs(cfg::RunConfig, default_priors) -> Dict{String, PriorSpec}
resolve_priors(cfg::RunConfig, default_priors) -> NamedTuple
```

The submodel's `default_priors()` with the run config's `[priors]` merged over them per key, as specs or as built distributions.

<a id='ConfigurableEpi.submodel_name-Tuple{RunConfig}'></a> <a id='ConfigurableEpi.submodel_name-Tuple{RunConfig}-1'></a> **`ConfigurableEpi.submodel_name`** &mdash; *Method*.

```julia
submodel_name(cfg::RunConfig) -> String
```

The single `[epi.<name>]` key.

<a id='ConfigurableEpi.validate_run_config-Tuple{RunConfig}'></a> <a id='ConfigurableEpi.validate_run_config-Tuple{RunConfig}-1'></a> **`ConfigurableEpi.validate_run_config`** &mdash; *Method*.

```julia
validate_run_config(cfg::RunConfig) -> cfg
```

Check what the schema cannot: a positive whole number of `step_days`, the cadence counts, `origin_mode`, the input scale and both inference axes.

## Model {#Model}

<a id='ConfigurableEpi.ParameterPriorBundle'></a> <a id='ConfigurableEpi.ParameterPriorBundle-1'></a> **`ConfigurableEpi.ParameterPriorBundle`** &mdash; *Type*.

```julia
ParameterPriorBundle(priors::NamedTuple)
ParameterPriorBundle(priors::ParameterDistribution...)
```

Ordered scalar priors combined into one EKP distribution.
`constrained_values`, `unconstrained_values` and `prior_logpdf` map between the optimiser's unconstrained vector and constrained `NamedTuple`s.
NamedTuple keys must equal the prior names.

<a id='ConfigurableEpi.prior_name-Tuple{ParameterDistribution}'></a> <a id='ConfigurableEpi.prior_name-Tuple{ParameterDistribution}-1'></a> **`ConfigurableEpi.prior_name`** &mdash; *Method*.

```julia
prior_name(prior::ParameterDistribution) -> Symbol
```

Name of a scalar EKP prior.
Errors on a combined (multi-name) distribution.

<a id='ConfigurableEpi.prior_unconstrained_mean-Tuple{Any}'></a> <a id='ConfigurableEpi.prior_unconstrained_mean-Tuple{Any}-1'></a> **`ConfigurableEpi.prior_unconstrained_mean`** &mdash; *Method*.

```julia
prior_unconstrained_mean(prior) -> Float64
prior_unconstrained_variance(prior) -> Float64
```

Mean and variance of a scalar prior in its unconstrained coordinate, the chart that learned hyperparameters and latent coefficients are stored in.

<a id='ConfigurableEpi.unconstrained_gaussian-Tuple{Any, Real, Real}'></a> <a id='ConfigurableEpi.unconstrained_gaussian-Tuple{Any, Real, Real}-1'></a> **`ConfigurableEpi.unconstrained_gaussian`** &mdash; *Method*.

```julia
unconstrained_gaussian(name, mean, sd)
positive_gaussian(name, mean, sd)
unit_interval_gaussian(name, mean, sd)
```

Scalar EKP `constrained_gaussian` priors on `(-Inf, Inf)`, `(0, Inf)` and `(0, 1)`.

<a id='ConfigurableEpi.StateLayout'></a> <a id='ConfigurableEpi.StateLayout-1'></a> **`ConfigurableEpi.StateLayout`** &mdash; *Type*.

```julia
StateLayout{N, M, L, S}
```

Layout of the state vector `[core compartments (N); observation states (M); latent coefficients (L)]` observed through `S` signals.
`accumulator_indices` are the absolute slots of the reset accumulators, one per signal.

```julia
StateLayout(core_names, obs_names, latent_names; signal_names = (:y,), accumulator_indices = nothing)
StateLayout(petri_net, driver_specs; signal_names = nothing)
```

Without `accumulator_indices` the observation states are split evenly over the signals and each signal's last state is its accumulator.
The second form reads the observation chains of a net built with `AlgebraicEpiMech.attach_observation`; only drivers with [`carries_state`](configurableepi.md#ConfigurableEpi.carries_state-Tuple{ParamSpec}) claim latent slots.

<a id='ConfigurableEpi.extract_latent-Union{Tuple{L}, Tuple{M}, Tuple{N}, Tuple{AbstractVector, StateLayout{N, M, L}}} where {N, M, L}'></a> <a id='ConfigurableEpi.extract_latent-Union{Tuple{L}, Tuple{M}, Tuple{N}, Tuple{AbstractVector, StateLayout{N, M, L}}} where {N, M, L}-1'></a> **`ConfigurableEpi.extract_latent`** &mdash; *Method*.

```julia
extract_latent(state, layout) -> NamedTuple
```

The latent slots of `state` as a NamedTuple of unconstrained values.

<a id='ConfigurableEpi.AR1ParamSpec'></a> <a id='ConfigurableEpi.AR1ParamSpec-1'></a> **`ConfigurableEpi.AR1ParamSpec`** &mdash; *Type*.

```julia
AR1ParamSpec(name; init, mu, tau, sigma)
```

Mean-reverting Ornstein–Uhlenbeck latent in the unconstrained chart of `init` (its initial prior, which also fixes the constraint): `u' | u ~ Normal(m + rho (u - m), sigma^2 (1 - rho^2))` with `rho = exp(-dt / tau)`.
`mu` is the constrained stationary mean, `tau` the correlation time in days and `sigma` the stationary sd; each may be a `Real`, a `ParameterDistribution` (learned under its own name) or a `ParamSpec`.

<a id='ConfigurableEpi.ArrivalProcess'></a> <a id='ConfigurableEpi.ArrivalProcess-1'></a> **`ConfigurableEpi.ArrivalProcess`** &mdash; *Type*.

```julia
ArrivalProcess(name; rate, mark, transition!)
```

A marked point process on the compartments (particle filter only).
On each stochastic step it fires with probability `1 - exp(-rate * dt)`, where `rate` is a constant or `(x_model, latent, hyper, t) -> λ`; on firing it draws `mark(hyper, rng) -> NamedTuple` and applies `transition!(x_model, mark, hyper)` in place.
It carries no state: the compartments record that it fired, so a single-shot arrival is a rate that reads the compartment its own jump seeds.

<a id='ConfigurableEpi.FixedParam'></a> <a id='ConfigurableEpi.FixedParam-1'></a> **`ConfigurableEpi.FixedParam`** &mdash; *Type*.

```julia
FixedParam(name, value)
HyperParam(prior)
DerivedParam(name, formula)
```

How a process parameter is resolved from the merged `(hyperparams, constrained latents)` NamedTuple: a constant, a lookup under the prior's name, or `formula(params)`.

<a id='ConfigurableEpi.IntegratedParamSpec'></a> <a id='ConfigurableEpi.IntegratedParamSpec-1'></a> **`ConfigurableEpi.IntegratedParamSpec`** &mdash; *Type*.

```julia
IntegratedParamSpec(name; init, rate, per_days = 1.0)
```

Noise-free latent whose unconstrained coordinate advances by `rate_value * dt / per_days`, where `rate_value` is the constrained value of the latent or hyperparameter named `rate` at the start of the step (explicit Euler).
With a random-walk `rate` the pair is an integrated Brownian motion.
A negative `per_days` integrates with the opposite sign.

<a id='ConfigurableEpi.RWParamSpec'></a> <a id='ConfigurableEpi.RWParamSpec-1'></a> **`ConfigurableEpi.RWParamSpec`** &mdash; *Type*.

```julia
RWParamSpec(name; init, sigma_rate)
```

Driftless random walk in the unconstrained chart; a step of `dt` days has sd `sigma_rate * sqrt(dt)`.

<a id='ConfigurableEpi.StochasticUpdate'></a> <a id='ConfigurableEpi.StochasticUpdate-1'></a> **`ConfigurableEpi.StochasticUpdate`** &mdash; *Type*.

```julia
StochasticUpdate{L}
```

The model's per-step stochastic driver over `L` coefficient latents, from \[`build_stochastic_update`\](configurableepi.md#ConfigurableEpi.build_stochastic_update-Union{Tuple{L}, Tuple{M}, Tuple{N}, Tuple{StateLayout{N, M, L}, Tuple}} where {N, M, L}):

- `advance(x, hyper, w, rng, t, dt)`: the pre-flow state after advancing each coefficient from unit noise `w[1:L]` and firing each jump driver on `rng` (`rng === nothing` skips jumps).
- `extract(x)`: the coefficients' constrained values as a NamedTuple.
- `extract_params(x, hyper)`: `merge(hyper, extract(x))`.
- `to_unconstrained(constrained::NamedTuple)`: the coefficients as an unconstrained `SVector{L}`.

<a id='ConfigurableEpi.advance_arrival!-Tuple{ArrivalProcess, Vararg{Any, 6}}'></a> <a id='ConfigurableEpi.advance_arrival!-Tuple{ArrivalProcess, Vararg{Any, 6}}-1'></a> **`ConfigurableEpi.advance_arrival!`** &mdash; *Method*.

```julia
advance_arrival!(process, x_model, latent, hyper, dt, t, rng) -> x_model
```

Fire one jump driver in place with probability `step_arrival_probability(rate, dt)`.

<a id='ConfigurableEpi.assert_gaussian_filter_compatible-Tuple{ParamSpec}'></a> <a id='ConfigurableEpi.assert_gaussian_filter_compatible-Tuple{ParamSpec}-1'></a> **`ConfigurableEpi.assert_gaussian_filter_compatible`** &mdash; *Method*.

```julia
assert_gaussian_filter_compatible(specs)
```

Throw if any driver is particle-only.

<a id='ConfigurableEpi.beta_mark-Tuple{}'></a> <a id='ConfigurableEpi.beta_mark-Tuple{}-1'></a> **`ConfigurableEpi.beta_mark`** &mdash; *Method*.

```julia
beta_mark(; mean_key = :mark_mean, concentration = 1.0, out_key = :mark)
```

Mark sampler `(hyper, rng) -> (; out_key => Beta(μν, (1 - μ)ν))` with mean `μ = hyper[mean_key]`.

<a id='ConfigurableEpi.build_stochastic_update-Union{Tuple{L}, Tuple{M}, Tuple{N}, Tuple{StateLayout{N, M, L}, Tuple}} where {N, M, L}'></a> <a id='ConfigurableEpi.build_stochastic_update-Union{Tuple{L}, Tuple{M}, Tuple{N}, Tuple{StateLayout{N, M, L}, Tuple}} where {N, M, L}-1'></a> **`ConfigurableEpi.build_stochastic_update`** &mdash; *Method*.

```julia
build_stochastic_update(layout, driver_specs) -> StochasticUpdate
```

Split `driver_specs` into state-carrying coefficient drivers (which must match `layout.latent_names` in order) and stateless jump drivers, and build the step driver.
Jumps read the pre-step coefficients; the flow then runs on the advanced ones.

<a id='ConfigurableEpi.carries_state-Tuple{ParamSpec}'></a> <a id='ConfigurableEpi.carries_state-Tuple{ParamSpec}-1'></a> **`ConfigurableEpi.carries_state`** &mdash; *Method*.

```julia
carries_state(spec) -> Bool
supports_gaussian_filter(spec) -> Bool
```

Whether a driver claims latent slots in the [`StateLayout`](configurableepi.md#ConfigurableEpi.StateLayout), and whether a Gaussian (UKF) filter can propagate it.
Both hold for the coefficient processes and fail for [`ArrivalProcess`](configurableepi.md#ConfigurableEpi.ArrivalProcess), whose fired/not-fired mixture no single Gaussian represents.

<a id='ConfigurableEpi.ou_step-Tuple{Real, Real}'></a> <a id='ConfigurableEpi.ou_step-Tuple{Real, Real}-1'></a> **`ConfigurableEpi.ou_step`** &mdash; *Method*.

```julia
ou_step(tau, dt) -> (rho, innovation_factor)
```

Exact one-step OU transition, `rho = exp(-dt / tau)` and `sqrt(1 - rho^2)`, via `expm1` so a very long `tau` does not cancel to zero.

<a id='ConfigurableEpi.pool_redistribute!-NTuple{4, Any}'></a> <a id='ConfigurableEpi.pool_redistribute!-NTuple{4, Any}-1'></a> **`ConfigurableEpi.pool_redistribute!`** &mdash; *Method*.

```julia
pool_redistribute!(x_model, sources, targets, weights) -> x_model
```

Pool the mass in `sources`, empty them, then add `weights[k] * pool` to `targets[k]` (weights should sum to 1).
Sources and targets may overlap; the pool is read before anything is written.

<a id='ConfigurableEpi.pro_rata_move!-NTuple{4, Any}'></a> <a id='ConfigurableEpi.pro_rata_move!-NTuple{4, Any}-1'></a> **`ConfigurableEpi.pro_rata_move!`** &mdash; *Method*.

```julia
pro_rata_move!(x_model, sources, targets, amount) -> x_model
```

Move up to `amount` individuals out of `sources[k]` into `targets[k]`, split pro rata by each source's occupancy and capped at what is available.
`sources` and `targets` must be disjoint.

<a id='ConfigurableEpi.seed_transition-Tuple{Any, Symbol, Symbol}'></a> <a id='ConfigurableEpi.seed_transition-Tuple{Any, Symbol, Symbol}-1'></a> **`ConfigurableEpi.seed_transition`** &mdash; *Method*.

```julia
seed_transition(model_names, from, into; size_key) -> transition!
```

Move `mark[size_key]` individuals from compartment `from` into `into`, capped at what `from` holds.

<a id='ConfigurableEpi.step_arrival_probability-Tuple{Real, Real}'></a> <a id='ConfigurableEpi.step_arrival_probability-Tuple{Real, Real}-1'></a> **`ConfigurableEpi.step_arrival_probability`** &mdash; *Method*.

```julia
step_arrival_probability(rate, dt) -> Float64
```

`1 - exp(-rate * dt)` for a constant hazard `rate >= 0` over a step `dt > 0`.

<a id='ConfigurableEpi.update_single'></a> <a id='ConfigurableEpi.update_single-1'></a> **`ConfigurableEpi.update_single`** &mdash; *Function*.

```julia
update_single(spec, old_unc, w, params, dt[, constraint])
```

Advance one latent's unconstrained coordinate over `dt` days given unit noise `w`.
`params` merges the hyperparameters with every latent's constrained value at the start of the step; `constraint` is `spec.init`'s pre-resolved `ScalarConstraint`.

<a id='ConfigurableEpi.build_R1-Tuple{StateLayout}'></a> <a id='ConfigurableEpi.build_R1-Tuple{StateLayout}-1'></a> **`ConfigurableEpi.build_R1`** &mdash; *Method*.

```julia
build_R1(layout) -> Diagonal
```

Identity process-noise covariance sized `n_latent + n_accumulators`; every noise magnitude is applied inside \[`build_full_dynamics`\](configurableepi.md#ConfigurableEpi.build_full_dynamics-Union{Tuple{S}, Tuple{L}, Tuple{M}, Tuple{N}, Tuple{Any, StochasticUpdate{L}, StateLayout{N, M, L, S}}} where {N, M, L, S}).

<a id='ConfigurableEpi.build_full_dynamics-Union{Tuple{S}, Tuple{L}, Tuple{M}, Tuple{N}, Tuple{Any, StochasticUpdate{L}, StateLayout{N, M, L, S}}} where {N, M, L, S}'></a> <a id='ConfigurableEpi.build_full_dynamics-Union{Tuple{S}, Tuple{L}, Tuple{M}, Tuple{N}, Tuple{Any, StochasticUpdate{L}, StateLayout{N, M, L, S}}} where {N, M, L, S}-1'></a> **`ConfigurableEpi.build_full_dynamics`** &mdash; *Method*.

```julia
build_full_dynamics(petri_vf!, stochastic, layout; dt = 1.0, supersample = 2, obs_jitter = 1.0)
    -> dynamics(x, u, p, t, w[, rng])
```

One filter step of the augmented state: apply the stochastic driver (coefficient noise from `w[1:L]`, jumps on `rng`), zero the reset accumulators, integrate the flow over `dt` with `supersample` RK4 substeps, then add the accumulator whisker `obs_jitter * w[L+1:end]`.
`w` has `size(build_R1(layout), 1)` entries.

<a id='ConfigurableEpi.build_petri_vf-Tuple{Any, Any}'></a> <a id='ConfigurableEpi.build_petri_vf-Tuple{Any, Any}-1'></a> **`ConfigurableEpi.build_petri_vf`** &mdash; *Method*.

```julia
build_petri_vf(pn, rates; defaults = (;)) -> petri_vf!(du, u, (hyperparams, latent), t)
```

In-place mass-action vector field of `pn` whose transition rates are `merge(defaults, rates(latent, hyperparams, t))`: `defaults` holds the fixed rates and the rate function returns only the dynamic ones, keyed by flattened transition name.

<a id='ConfigurableEpi.build_unified_vf-Union{Tuple{M}, Tuple{N}, Tuple{Any, StateLayout{N, M}}} where {N, M}'></a> <a id='ConfigurableEpi.build_unified_vf-Union{Tuple{M}, Tuple{N}, Tuple{Any, StateLayout{N, M}}} where {N, M}-1'></a> **`ConfigurableEpi.build_unified_vf`** &mdash; *Method*.

```julia
build_unified_vf(petri_vf!, layout) -> (x, u, p, t) -> dx
```

Out-of-place form of the Petri vector field for `SeeToDee.Rk4`, naming the ODE slots of `x`.

<a id='ConfigurableEpi.make_lvector_constructor-Union{Tuple{NTuple{N, Symbol}}, Tuple{N}} where N'></a> <a id='ConfigurableEpi.make_lvector_constructor-Union{Tuple{NTuple{N, Symbol}}, Tuple{N}} where N-1'></a> **`ConfigurableEpi.make_lvector_constructor`** &mdash; *Method*.

```julia
make_lvector_constructor(names) -> x -> LArray{names}(collect(x))
```

<a id='ConfigurableEpi.LogNormalNoise'></a> <a id='ConfigurableEpi.LogNormalNoise-1'></a> **`ConfigurableEpi.LogNormalNoise`** &mdash; *Type*.

```julia
LogNormalNoise(sigma)
```

Multiplicative noise `y = μ exp(σ v)`.

<a id='ConfigurableEpi.NegBinomialNoise'></a> <a id='ConfigurableEpi.NegBinomialNoise-1'></a> **`ConfigurableEpi.NegBinomialNoise`** &mdash; *Type*.

```julia
NegBinomialNoise(; phi, sigma_mult = 0.0)
```

Count noise with `Var(y) = μ + μ²/φ + (σμ)²`.
The UKF uses the Gaussian approximation with one unit-normal term; the particle filter uses the exact `NegativeBinomial(φ, φ/(φ+μ))`, which drops the `(σμ)²` reporting term.
Each parameter is a `Real` or a function `(latent, hyper, t) -> Real`.

<a id='ConfigurableEpi.PoissonNoise'></a> <a id='ConfigurableEpi.PoissonNoise-1'></a> **`ConfigurableEpi.PoissonNoise`** &mdash; *Type*.

```julia
PoissonNoise(sigma_mult = nothing)
```

Poisson count noise `y ≈ μ + sqrt(μ) v`; with `sigma_mult` the mean is first perturbed by `exp(σ v₁)` (UKF only: the particle filter rejects the mixture).

<a id='ConfigurableEpi.SignalObservationSpec'></a> <a id='ConfigurableEpi.SignalObservationSpec-1'></a> **`ConfigurableEpi.SignalObservationSpec`** &mdash; *Type*.

```julia
SignalObservationSpec(signal_idx, noise; mean_modifier = nothing, baseline = nothing, name)
AggregatedSignalSpec(signal_indices, noise; mean_modifier = nothing, baseline = nothing, name)
AggregatedSignalSpec(noise; ...)                       # the sum of every signal
```

One observation: a signal's reset accumulator, or the sum over several.
Its mean is `raw * mean_modifier + baseline`, where each of the two is `nothing`, a `Real` or a function `(latent, hyper, t) -> Real` (an [`AscertainmentPath`](configurableepi.md#ConfigurableEpi.AscertainmentPath), say).

<a id='ConfigurableEpi.apply_noise-Tuple{NegBinomialNoise, Vararg{Any, 5}}'></a> <a id='ConfigurableEpi.apply_noise-Tuple{NegBinomialNoise, Vararg{Any, 5}}-1'></a> **`ConfigurableEpi.apply_noise`** &mdash; *Method*.

```julia
apply_noise(noise, mean, v, latent, hyper, t)
```

Gaussian-approximation observation given unit noise `v` (the UKF measurement).

<a id='ConfigurableEpi.build_measurement_logpdf-Tuple{StateLayout, Tuple{Vararg{ObservationSpec}}, StochasticUpdate}'></a> <a id='ConfigurableEpi.build_measurement_logpdf-Tuple{StateLayout, Tuple{Vararg{ObservationSpec}}, StochasticUpdate}-1'></a> **`ConfigurableEpi.build_measurement_logpdf`** &mdash; *Method*.

```julia
build_measurement_logpdf(layout, obs_specs, stochastic; learned = nothing) -> g(x, u, y, p, t)
```

Particle weighting: the summed exact `observation_logpdf` of `y` over the observation specs.
`learned` (a `LearnedHyperparams`) lets each particle's own hyperparameters override `p`.

<a id='ConfigurableEpi.build_measurement_model-Tuple{StateLayout, Tuple{Vararg{ObservationSpec}}, StochasticUpdate}'></a> <a id='ConfigurableEpi.build_measurement_model-Tuple{StateLayout, Tuple{Vararg{ObservationSpec}}, StochasticUpdate}-1'></a> **`ConfigurableEpi.build_measurement_model`** &mdash; *Method*.

```julia
build_measurement_model(layout, obs_specs, stochastic) -> (; measure, n_obs, n_noise)
build_measurement_model(layout, noise::ObservationNoiseSpec, stochastic)
```

The UKF measurement `measure(x, u, p, t, v) -> SVector{n_obs}` with unit noise `v` of length `n_noise` (so `R2 = I`).
The second form observes the single signal of a one-signal layout.

<a id='ConfigurableEpi.observation_gaussian_moments-NTuple{6, Any}'></a> <a id='ConfigurableEpi.observation_gaussian_moments-NTuple{6, Any}-1'></a> **`ConfigurableEpi.observation_gaussian_moments`** &mdash; *Method*.

```julia
observation_gaussian_moments(spec, raw_mean, raw_var, latent, hyper, t) -> (; mean, var)
```

Gaussian moments of a NegBinomial observation given Gaussian moments of its accumulator (the UKF forecast mapping): `mean = observation_mean(max(raw_mean, 0))` and `var = scale² raw_var + mean + mean²/φ + (σ mean)²`, with the spec's scale and noise at `t`.

<a id='ConfigurableEpi.observation_logpdf-Tuple{NegBinomialNoise, Vararg{Any, 5}}'></a> <a id='ConfigurableEpi.observation_logpdf-Tuple{NegBinomialNoise, Vararg{Any, 5}}-1'></a> **`ConfigurableEpi.observation_logpdf`** &mdash; *Method*.

```julia
observation_logpdf(noise, y, mean, latent, hyper, t)
sample_observation(noise, mean, latent, hyper, t, rng)
```

Exact log-density and draw of an observation (particle weighting and simulation).

<a id='ConfigurableEpi.observation_mean-NTuple{5, Any}'></a> <a id='ConfigurableEpi.observation_mean-NTuple{5, Any}-1'></a> **`ConfigurableEpi.observation_mean`** &mdash; *Method*.

```julia
observation_mean(spec, raw, latent, hyper, t)      # raw * modifier + baseline
observation_scale(spec, latent, hyper, t)          # the modifier at `t`
observation_baseline(spec, latent, hyper, t)       # the baseline at `t`
```

Reporting code must map accumulators to counts through these, never through a hyperparameter read directly, so a time-varying or latent-driven ascertainment is honoured.

<a id='ConfigurableEpi.resolve_signal_indices-Tuple{SignalObservationSpec, Integer}'></a> <a id='ConfigurableEpi.resolve_signal_indices-Tuple{SignalObservationSpec, Integer}-1'></a> **`ConfigurableEpi.resolve_signal_indices`** &mdash; *Method*.

```julia
resolve_signal_indices(spec, n_signals) -> spec
```

Check a spec's signal indices against the layout, expanding an all-signals aggregate.

<a id='ConfigurableEpi.N_SEASON_KNOTS'></a> <a id='ConfigurableEpi.N_SEASON_KNOTS-1'></a> **`ConfigurableEpi.N_SEASON_KNOTS`** &mdash; *Constant*.

Knots per `indoor_activity` climatology curve: one per week of the year.

<a id='ConfigurableEpi.UnitForcing'></a> <a id='ConfigurableEpi.UnitForcing-1'></a> **`ConfigurableEpi.UnitForcing`** &mdash; *Type*.

```julia
UnitForcing()
CosineForcing(day0)
IndoorActivityForcing(curve, u0)
```

The three forcings: flat `1.0`; the annual harmonic `1 + hyper.seasonal_amp * cos(2π (t + day0 - hyper.seasonal_phase) / 365.25)` with `day0` the day-of-year of `t = 0`; and `1 + hyper.seasonal_kappa * (σ(t) - 1)` with `σ` a unit-mean periodic spline through a location's climatology and `u0` the year fraction of `t = 0`.

<a id='ConfigurableEpi.assert_seasonal_learnable'></a> <a id='ConfigurableEpi.assert_seasonal_learnable-1'></a> **`ConfigurableEpi.assert_seasonal_learnable`** &mdash; *Function*.

```julia
assert_seasonal_learnable(cfg::SeasonalityConfig, learn_params, prior_specs = nothing)
```

Reject learning a seasonality parameter the active mode never reads (an unidentified dimension), and require a `unit_interval` prior when `seasonal_kappa` is learned.

<a id='ConfigurableEpi.build_periodic_curve-Tuple{AbstractVector{Float64}}'></a> <a id='ConfigurableEpi.build_periodic_curve-Tuple{AbstractVector{Float64}}-1'></a> **`ConfigurableEpi.build_periodic_curve`** &mdash; *Method*.

```julia
build_periodic_curve(knots) -> CubicSpline
```

Unit-mean periodic cubic spline through `knots`, knot `k` at `u = (k - 0.5) / N` within the year.
Periodicity comes from tiling five years and evaluating in the central one (the interpolant's own periodic extrapolation repeats with period `(N-1)/N`); the annual mean is normalised by the analytic integral.

<a id='ConfigurableEpi.build_seasonal_forcing-Tuple{SeasonalityConfig, Any, Dates.Date}'></a> <a id='ConfigurableEpi.build_seasonal_forcing-Tuple{SeasonalityConfig, Any, Dates.Date}-1'></a> **`ConfigurableEpi.build_seasonal_forcing`** &mdash; *Method*.

```julia
build_seasonal_forcing(cfg::SeasonalityConfig, location, start_date::Date; climatology = nothing)
build_seasonal_forcings(cfg, locations, start_date; climatology = nothing) -> Vector
```

The forcing selected by `cfg`, anchored so model time `t = 0` is `start_date`.
`climatology` (`Dict(location => 52 knots)`) is required by `indoor_activity`.
The result type depends on the mode, so specialise the vector field on it through a `where {F}` function barrier.

<a id='ConfigurableEpi.default_seasonal_learned-Tuple{SeasonalityConfig}'></a> <a id='ConfigurableEpi.default_seasonal_learned-Tuple{SeasonalityConfig}-1'></a> **`ConfigurableEpi.default_seasonal_learned`** &mdash; *Method*.

```julia
default_seasonal_learned(cfg::SeasonalityConfig) -> Vector{Symbol}
```

Seasonality parameters learned by default: the cosine's amplitude and phase, nothing otherwise (`kappa` scales an externally estimated curve, so it is opt-in via `learn_params`).

<a id='ConfigurableEpi.load_indoor_activity_climatology-Tuple{AbstractString}'></a> <a id='ConfigurableEpi.load_indoor_activity_climatology-Tuple{AbstractString}-1'></a> **`ConfigurableEpi.load_indoor_activity_climatology`** &mdash; *Method*.

```julia
load_indoor_activity_climatology(path) -> Dict{String, Vector{Float64}}
```

Read a `location,knot,value` table of 52-knot, unit-mean curves (lowercase location keys, `us` included).
The package ships no data; the caller owns this file.

<a id='ConfigurableEpi.seasonal_forcing_upper_bound'></a> <a id='ConfigurableEpi.seasonal_forcing_upper_bound-1'></a> **`ConfigurableEpi.seasonal_forcing_upper_bound`** &mdash; *Function*.

```julia
seasonal_forcing_upper_bound(cfg, fixed_amp, prior_specs, learn_params = ();
                             probability = 0.95, climatology = nothing) -> Float64
```

Mode-aware upper bound on the seasonal multiplier for the RK4 stability guard: 1 for `none`, `1 + |amplitude|` for `cosine` (from the prior when learned), and the largest curve value in `climatology` scaled by the fixed or learned `kappa` for `indoor_activity`.
An empty `learn_params` means the mode's default learned set.

<a id='ConfigurableEpi.validate_indoor_activity_climatology-Tuple{AbstractDict}'></a> <a id='ConfigurableEpi.validate_indoor_activity_climatology-Tuple{AbstractDict}-1'></a> **`ConfigurableEpi.validate_indoor_activity_climatology`** &mdash; *Method*.

```julia
validate_indoor_activity_climatology(climatology; source = "supplied climatology")
```

Check that every (lowercase) location carries `N_SEASON_KNOTS` finite, positive knots with mean 1.

<a id='ConfigurableEpi.validate_seasonality-Tuple{SeasonalityConfig}'></a> <a id='ConfigurableEpi.validate_seasonality-Tuple{SeasonalityConfig}-1'></a> **`ConfigurableEpi.validate_seasonality`** &mdash; *Method*.

```julia
validate_seasonality(cfg::SeasonalityConfig) -> cfg
```

Check the mode, the fallback and `kappa ∈ [0, 1]` (the seasonally forced share of transmission).

<a id='ConfigurableEpi.year_fraction-Tuple{Dates.Date}'></a> <a id='ConfigurableEpi.year_fraction-Tuple{Dates.Date}-1'></a> **`ConfigurableEpi.year_fraction`** &mdash; *Method*.

Position of `d` within its year in `[0, 1)`, leap-aware.

<a id='ConfigurableEpi.ASCERTAINMENT_TREND_LEVEL'></a> <a id='ConfigurableEpi.ASCERTAINMENT_TREND_LEVEL-1'></a> **`ConfigurableEpi.ASCERTAINMENT_TREND_LEVEL`** &mdash; *Constant*.

Latent name of the ascertainment level (observations per infection now); positive, stored as `log alpha`.

<a id='ConfigurableEpi.ASCERTAINMENT_TREND_LEVEL_LOG_SD'></a> <a id='ConfigurableEpi.ASCERTAINMENT_TREND_LEVEL_LOG_SD-1'></a> **`ConfigurableEpi.ASCERTAINMENT_TREND_LEVEL_LOG_SD`** &mdash; *Constant*.

Initial spread of the level in log units (pinned: the level is not identifiable alongside transmission).

<a id='ConfigurableEpi.ASCERTAINMENT_TREND_RATE'></a> <a id='ConfigurableEpi.ASCERTAINMENT_TREND_RATE-1'></a> **`ConfigurableEpi.ASCERTAINMENT_TREND_RATE`** &mdash; *Constant*.

Latent name of the decline rate per year; the same name (and prior) as the declining path's hyperparameter.

<a id='ConfigurableEpi.ASCERTAINMENT_TREND_WANDER'></a> <a id='ConfigurableEpi.ASCERTAINMENT_TREND_WANDER-1'></a> **`ConfigurableEpi.ASCERTAINMENT_TREND_WANDER`** &mdash; *Constant*.

Hyperparameter name of the trend's bend scale; see [`DEFAULT_ASCERTAINMENT_TREND_WANDER`](configurableepi.md#ConfigurableEpi.DEFAULT_ASCERTAINMENT_TREND_WANDER).

<a id='ConfigurableEpi.DEFAULT_ASCERTAINMENT_TREND_WANDER'></a> <a id='ConfigurableEpi.DEFAULT_ASCERTAINMENT_TREND_WANDER-1'></a> **`ConfigurableEpi.DEFAULT_ASCERTAINMENT_TREND_WANDER`** &mdash; *Constant*.

The 1-sd departure of log-ascertainment from a straight line after one year, in log units: too stiff to mimic a wave, loose enough for the rate to re-diversify over a season.

<a id='ConfigurableEpi.DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS'></a> <a id='ConfigurableEpi.DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS-1'></a> **`ConfigurableEpi.DEFAULT_ASCERTAINMENT_TREND_WANDER_MEMORY_DAYS`** &mdash; *Constant*.

Memory (days) of the Kulhavý forgetting applied to a learned wander: a season and a half.

<a id='ConfigurableEpi.AscertainmentPath'></a> <a id='ConfigurableEpi.AscertainmentPath-1'></a> **`ConfigurableEpi.AscertainmentPath`** &mdash; *Type*.

```julia
AscertainmentPath(floor_fraction, t_ref_days, rate_bound = Inf)
AscertainmentPath(; floor_fraction, reference_date::Date, start_date::Date, rate_bound = Inf)
```

Callable `(hyper, t)` and `(latent, hyper, t)` giving observations per infection at model time `t`,

```julia
alpha(t) = level * (1 + (1 - floor_fraction) * expm1(-rate * (t - t_ref_days) / 365.25))
```

with `level = hyper.ascertainment` and `rate = clamp(hyper.ascertainment_decline_rate, ±rate_bound)` read on every call, so a learned rate reaches each particle or optimiser candidate.
A zero rate returns `level` bit-for-bit.
Use it as an observation spec's `mean_modifier` and as `t -> path(hyper, t)` in the initial-state inversion.

<a id='ConfigurableEpi.TrendAscertainment'></a> <a id='ConfigurableEpi.TrendAscertainment-1'></a> **`ConfigurableEpi.TrendAscertainment`** &mdash; *Type*.

```julia
TrendAscertainment()
```

The trend model's observation `mean_modifier`: `(latent, hyper, t) -> latent.ascertainment_level`.

<a id='ConfigurableEpi.TrendAscertainmentSeed'></a> <a id='ConfigurableEpi.TrendAscertainmentSeed-1'></a> **`ConfigurableEpi.TrendAscertainmentSeed`** &mdash; *Type*.

```julia
TrendAscertainmentSeed(level0, decline_rate)
```

`t -> level0 * exp(-decline_rate * t / 365.25)` for the initial-state inversion, which evaluates ascertainment at and before `t = 0`.

<a id='ConfigurableEpi.ascertainment_at-NTuple{4, Any}'></a> <a id='ConfigurableEpi.ascertainment_at-NTuple{4, Any}-1'></a> **`ConfigurableEpi.ascertainment_at`** &mdash; *Method*.

```julia
ascertainment_at(level, floor_fraction, rate, days_since_ref)
```

The path's kernel, `level * (1 + (1 - floor_fraction) * expm1(-rate * days_since_ref / 365.25))`.

<a id='ConfigurableEpi.ascertainment_trend_sigma_rate-Tuple{Real}'></a> <a id='ConfigurableEpi.ascertainment_trend_sigma_rate-Tuple{Real}-1'></a> **`ConfigurableEpi.ascertainment_trend_sigma_rate`** &mdash; *Method*.

```julia
ascertainment_trend_sigma_rate(wander) -> Float64
```

The rate's random-walk diffusion per sqrt-day that makes the level's one-year departure from a straight line equal `wander`: `wander * sqrt(3 / 365.25)`.

<a id='ConfigurableEpi.ascertainment_trend_specs-Tuple{}'></a> <a id='ConfigurableEpi.ascertainment_trend_specs-Tuple{}-1'></a> **`ConfigurableEpi.ascertainment_trend_specs`** &mdash; *Method*.

```julia
ascertainment_trend_specs(; rate_prior, level0) -> (rate_spec, level_spec)
```

The trend model's two drivers: the decline rate as a random walk whose diffusion derives from the `ascertainment_trend_wander` hyperparameter, and the level as the noise-free integral of `-rate` per year.
`rate_prior` must be named `ascertainment_decline_rate`.

<a id='ConfigurableEpi.assert_ascertainment_learnable-Tuple{Any, Any}'></a> <a id='ConfigurableEpi.assert_ascertainment_learnable-Tuple{Any, Any}-1'></a> **`ConfigurableEpi.assert_ascertainment_learnable`** &mdash; *Method*.

```julia
assert_ascertainment_learnable(fixed, learn_params)
```

Refuse to learn the decline rate when `ascertainment_floor_fraction == 1`, where the path never reads it.

<a id='ConfigurableEpi.build_ascertainment_path-Tuple{Any, Dates.Date}'></a> <a id='ConfigurableEpi.build_ascertainment_path-Tuple{Any, Dates.Date}-1'></a> **`ConfigurableEpi.build_ascertainment_path`** &mdash; *Method*.

```julia
build_ascertainment_path(fixed, start_date::Date) -> AscertainmentPath
```

Anchor a submodel's ascertainment block so model time `t = 0` is `start_date`.

<a id='ConfigurableEpi.parse_ascertainment_reference_date-Tuple{Any}'></a> <a id='ConfigurableEpi.parse_ascertainment_reference_date-Tuple{Any}-1'></a> **`ConfigurableEpi.parse_ascertainment_reference_date`** &mdash; *Method*.

```julia
parse_ascertainment_reference_date(value) -> Date
```

Parse the ISO `YYYY-MM-DD` reference date carried as a string in the epi config.

<a id='ConfigurableEpi.validate_ascertainment-Tuple{Any}'></a> <a id='ConfigurableEpi.validate_ascertainment-Tuple{Any}-1'></a> **`ConfigurableEpi.validate_ascertainment`** &mdash; *Method*.

```julia
validate_ascertainment(fixed)
```

Check the `ascertainment`, `ascertainment_decline_rate`, `ascertainment_floor_fraction`, `ascertainment_rate_bound` and `ascertainment_reference_date` fields of a submodel's fixed config.

<a id='ConfigurableEpi.DayOfWeekModifier'></a> <a id='ConfigurableEpi.DayOfWeekModifier-1'></a> **`ConfigurableEpi.DayOfWeekModifier`** &mdash; *Type*.

```julia
DayOfWeekModifier(inner, dow0, weights)
```

An observation `mean_modifier` equal to `inner(hyper, t) * w_{d(t)}`, callable as `(hyper, t)` and `(latent, hyper, t)`.
`weights::NTuple{7}` is the plugin effect; `weights = nothing` reads the learned multipliers from `hyper.dow_z1 … dow_z6`.

<a id='ConfigurableEpi.HyperPhi'></a> <a id='ConfigurableEpi.HyperPhi-1'></a> **`ConfigurableEpi.HyperPhi`** &mdash; *Type*.

```julia
HyperPhi()                            # (latent, hyper, t) -> hyper.phi
DayOfWeekDispersion(dow0, extra_var)  # (latent, hyper, t) -> 1 / (1/hyper.phi + extra_var[d(t)])
```

The negative-binomial dispersion without and with the plugin's per-weekday widening.

<a id='ConfigurableEpi.assert_day_of_week_learnable-Tuple{DayOfWeekConfig, Any}'></a> <a id='ConfigurableEpi.assert_day_of_week_learnable-Tuple{DayOfWeekConfig, Any}-1'></a> **`ConfigurableEpi.assert_day_of_week_learnable`** &mdash; *Method*.

```julia
assert_day_of_week_learnable(cfg::DayOfWeekConfig, learn_params)
```

The weekday coordinates in an explicit learned set must be exactly `day_of_week_learned_names(cfg)`.

<a id='ConfigurableEpi.build_day_of_week_observation-Tuple{NoDayOfWeekConfig, Any, Dates.Date, Any}'></a> <a id='ConfigurableEpi.build_day_of_week_observation-Tuple{NoDayOfWeekConfig, Any, Dates.Date, Any}-1'></a> **`ConfigurableEpi.build_day_of_week_observation`** &mdash; *Method*.

```julia
build_day_of_week_observation(cfg, inner, start_date, history; phi, precomputed = nothing)
    -> (; modifier, dispersion, effects, derived_hyperparameters)
```

The observation pieces for weekday effect `cfg`: the `mean_modifier` (`inner` itself under `NoDayOfWeekConfig`), the NB dispersion function, the plugin's history estimates to report, and the learned multipliers to summarise.
`history` is `(; times, counts)` on the model clock; `precomputed` reuses a first-vintage plugin estimate.

<a id='ConfigurableEpi.day_of_week_learned_names-Tuple{DayOfWeekConfig}'></a> <a id='ConfigurableEpi.day_of_week_learned_names-Tuple{DayOfWeekConfig}-1'></a> **`ConfigurableEpi.day_of_week_learned_names`** &mdash; *Method*.

The θ names this effect adds to the learned set: the six Helmert coordinates under `learned`.

<a id='ConfigurableEpi.day_of_week_multipliers-Tuple{Any}'></a> <a id='ConfigurableEpi.day_of_week_multipliers-Tuple{Any}-1'></a> **`ConfigurableEpi.day_of_week_multipliers`** &mdash; *Method*.

```julia
day_of_week_multipliers(hyper) -> NTuple{7}
```

The learned weekday multipliers Monday … Sunday, `7 * softmax(H z)` (shifted by the maximum so a large coordinate cannot overflow).

<a id='ConfigurableEpi.day_of_week_report_rows-Tuple{Nothing}'></a> <a id='ConfigurableEpi.day_of_week_report_rows-Tuple{Nothing}-1'></a> **`ConfigurableEpi.day_of_week_report_rows`** &mdash; *Method*.

```julia
day_of_week_report_rows(effects) -> Tuple of (parameter, statistic, value)
```

Summary rows for the plugin's estimates: `dow_multiplier_<day>` and `dow_extra_var_<day>`.

<a id='ConfigurableEpi.day_of_week_weight-Tuple{Any, Any, Any}'></a> <a id='ConfigurableEpi.day_of_week_weight-Tuple{Any, Any, Any}-1'></a> **`ConfigurableEpi.day_of_week_weight`** &mdash; *Method*.

```julia
day_of_week_weight(modifier, hyper, t)
```

The weekday factor alone at observation time `t`; `true` for a modifier without one.

<a id='ConfigurableEpi.dow_helmert_prior_sd-Tuple{Real}'></a> <a id='ConfigurableEpi.dow_helmert_prior_sd-Tuple{Real}-1'></a> **`ConfigurableEpi.dow_helmert_prior_sd`** &mdash; *Method*.

Prior sd of each Helmert coordinate that gives every centred weekday log-effect marginal sd `sd`.

<a id='ConfigurableEpi.estimate_day_of_week_effects-Tuple{AbstractVector{Dates.Date}, AbstractVector{<:Real}}'></a> <a id='ConfigurableEpi.estimate_day_of_week_effects-Tuple{AbstractVector{Dates.Date}, AbstractVector{<:Real}}-1'></a> **`ConfigurableEpi.estimate_day_of_week_effects`** &mdash; *Method*.

```julia
estimate_day_of_week_effects(dates, counts; phi, window_days = 182, exclude_recent_days = 14,
                             min_weeks = 4) -> (; weights, extra_var, fallback, n_used)
```

Multiplicative weekday decomposition of a gap-free daily series against a leave-one-out centred weekly baseline: weights `w_d = 7 q_d / (6 + q_d)` from the ratio of sums `q_d = Σy / Σb`, normalised to mean 1, and per-weekday extra variance `s_d²` after removing the Poisson/NB sampling variance of the counts and of the baseline.
Falls back to no effect (with a warning) when any weekday has fewer than `min_weeks` usable days.

<a id='ConfigurableEpi.prepare_day_of_week_history-Tuple{AbstractVector{Dates.Date}, AbstractVector{<:Real}, Dates.Date}'></a> <a id='ConfigurableEpi.prepare_day_of_week_history-Tuple{AbstractVector{Dates.Date}, AbstractVector{<:Real}, Dates.Date}-1'></a> **`ConfigurableEpi.prepare_day_of_week_history`** &mdash; *Method*.

```julia
prepare_day_of_week_history(dates, counts, report_date) -> (; dates, counts)
```

Check one daily as-of series ends on `report_date - 1` and is gap-free, and drop its final observation (the model's history contract).

<a id='ConfigurableEpi.remove_day_of_week_effect-Tuple{Any, Any, Any}'></a> <a id='ConfigurableEpi.remove_day_of_week_effect-Tuple{Any, Any, Any}-1'></a> **`ConfigurableEpi.remove_day_of_week_effect`** &mdash; *Method*.

```julia
remove_day_of_week_effect(history, modifier, hyper) -> history
```

Divide each historical count by the weekday weight of its own observation date so the initial-state reconstruction inverts with ascertainment alone.

<a id='ConfigurableEpi.validate_day_of_week-Tuple{Union{LearnedDayOfWeekConfig, NoDayOfWeekConfig}}'></a> <a id='ConfigurableEpi.validate_day_of_week-Tuple{Union{LearnedDayOfWeekConfig, NoDayOfWeekConfig}}-1'></a> **`ConfigurableEpi.validate_day_of_week`** &mdash; *Method*.

```julia
validate_day_of_week(cfg::DayOfWeekConfig) -> cfg
```

The plugin window must hold at least five weeks; other modes have nothing to check.

<a id='ConfigurableEpi.RK4_STABILITY_LIMIT'></a> <a id='ConfigurableEpi.RK4_STABILITY_LIMIT-1'></a> **`ConfigurableEpi.RK4_STABILITY_LIMIT`** &mdash; *Constant*.

Classical RK4 is stable on a real negative eigenvalue only while `|lambda * h| < 2.785`.

<a id='ConfigurableEpi.RK4_WARN_LAMBDA_H'></a> <a id='ConfigurableEpi.RK4_WARN_LAMBDA_H-1'></a> **`ConfigurableEpi.RK4_WARN_LAMBDA_H`** &mdash; *Constant*.

Proximity warning threshold, about 1.4x below the stability limit.

<a id='ConfigurableEpi.anchor_is_exact-Tuple{Integer, Integer}'></a> <a id='ConfigurableEpi.anchor_is_exact-Tuple{Integer, Integer}-1'></a> **`ConfigurableEpi.anchor_is_exact`** &mdash; *Method*.

Whether `R_eff = 1` holds exactly at the peak of `E + I`: only for a single `I` stage.

<a id='ConfigurableEpi.assert_integration_stable-Tuple{Durations, Integer, Integer, Real, Integer}'></a> <a id='ConfigurableEpi.assert_integration_stable-Tuple{Durations, Integer, Integer, Real, Integer}-1'></a> **`ConfigurableEpi.assert_integration_stable`** &mdash; *Method*.

```julia
assert_integration_stable(dur, n_E, n_I, dt, supersample; R_eff_max = 1.0) -> lambda_h
```

Error beyond [`RK4_STABILITY_LIMIT`](configurableepi.md#ConfigurableEpi.RK4_STABILITY_LIMIT) and warn beyond [`RK4_WARN_LAMBDA_H`](configurableepi.md#ConfigurableEpi.RK4_WARN_LAMBDA_H).
Size `R_eff_max` from the priors' upper tail (\[`prior_R_eff_bound`\](configurableepi.md#ConfigurableEpi.prior_R_eff_bound-Tuple{AbstractDict{String, PriorSpec}})), not the endemic seed.

<a id='ConfigurableEpi.carry_susceptible-Tuple{Real, Real, Real, AbstractVector{<:Real}, AbstractVector{<:Real}, Real}'></a> <a id='ConfigurableEpi.carry_susceptible-Tuple{Real, Real, Real, AbstractVector{<:Real}, AbstractVector{<:Real}, Real}-1'></a> **`ConfigurableEpi.carry_susceptible`** &mdash; *Method*.

```julia
carry_susceptible(s_anchor, t_anchor, t_target, times, daily_incidence, omega) -> Float64
```

Integrate `ds/dt = omega (1 - s) - i(t)` from the anchor to the target time (midpoint RK2, steps of at most one day), holding the per-capita incidence flat outside its range.

<a id='ConfigurableEpi.find_observed_peak-Tuple{AbstractVector{<:Real}}'></a> <a id='ConfigurableEpi.find_observed_peak-Tuple{AbstractVector{<:Real}}-1'></a> **`ConfigurableEpi.find_observed_peak`** &mdash; *Method*.

```julia
find_observed_peak(counts; window = 5, min_prominence = 1.1) -> Union{Int, Nothing}
```

Index of the observed wave peak: an interior, strict maximum of the log-smoothed series at least `min_prominence` above the higher flank.
`nothing` when the history contains no identified peak.

<a id='ConfigurableEpi.initial_infection_state-Tuple{Any, Durations, Any, Any}'></a> <a id='ConfigurableEpi.initial_infection_state-Tuple{Any, Durations, Any, Any}-1'></a> **`ConfigurableEpi.initial_infection_state`** &mdash; *Method*.

```julia
initial_infection_state(y0, dur::Durations, ascertainment, accumulation_window_days)
    -> (; daily_incidence, exposed, infectious, obs_stage)
```

Invert one partially ascertained count into quasi-steady compartment occupancies: `daily_incidence = y0 / (ascertainment * accumulation_window_days)` and each upstream compartment holds `daily_incidence * mean_duration`.

<a id='ConfigurableEpi.max_transition_rate-Tuple{Durations, Integer, Integer}'></a> <a id='ConfigurableEpi.max_transition_rate-Tuple{Durations, Integer, Integer}-1'></a> **`ConfigurableEpi.max_transition_rate`** &mdash; *Method*.

```julia
max_transition_rate(dur::Durations, n_E, n_I; R_eff_max = 1.0) -> Float64
```

The fastest model rate in 1/day: the Erlang stage rates, observation progression, waning, and the faster eigenvalue of the linearised `(E, I)` block at `R_eff_max`.

<a id='ConfigurableEpi.peak_anchored_susceptible_fraction-Tuple{Any, Durations, Integer, Any, Real, Real, Any, Real}'></a> <a id='ConfigurableEpi.peak_anchored_susceptible_fraction-Tuple{Any, Durations, Integer, Any, Real, Real, Any, Real}-1'></a> **`ConfigurableEpi.peak_anchored_susceptible_fraction`** &mdash; *Method*.

```julia
peak_anchored_susceptible_fraction(history, dur, n_obs_stages, ascertainment, pop, dt, chi_at, R0;
                                   n_E = 1, n_I = 1) -> Union{NamedTuple, Nothing}
```

Find the observed peak in `history = (; times, counts)` (model clock, bin-end labels), shift it back by half a bin and the reporting delay and forward by the prevalence lag, evaluate `S/N = 1 / (R0 * chi_at(t_anchor))` there (`R_eff = 1` at the prevalence peak, `Rt = 1` by construction), and carry `S/N` back to `t = 0` against the reconstructed incidence.
`ascertainment` is a constant or `t -> observations per infection`.
Returns `nothing` when no peak is identified or the anchor is inexact (`n_I > 1`).

<a id='ConfigurableEpi.prevalence_peak_lag_days'></a> <a id='ConfigurableEpi.prevalence_peak_lag_days-1'></a> **`ConfigurableEpi.prevalence_peak_lag_days`** &mdash; *Function*.

```julia
prevalence_peak_lag_days(dur::Durations, n_E = 1, n_I = 1) -> Float64
```

How long after the incidence peak total infected prevalence peaks: the normalised mean `E[T²] / (2 E[T])` of the Erlang residence kernel.

<a id='ConfigurableEpi.reporting_delay_days-Tuple{Durations, Integer}'></a> <a id='ConfigurableEpi.reporting_delay_days-Tuple{Durations, Integer}-1'></a> **`ConfigurableEpi.reporting_delay_days`** &mdash; *Method*.

```julia
reporting_delay_days(dur::Durations, n_obs_stages) -> Float64
```

Mean infection-to-report delay, `(n_obs_stages - 1) * obs_progression`.

<a id='ConfigurableEpi.required_supersample-Tuple{Durations, Integer, Integer, Real}'></a> <a id='ConfigurableEpi.required_supersample-Tuple{Durations, Integer, Integer, Real}-1'></a> **`ConfigurableEpi.required_supersample`** &mdash; *Method*.

```julia
required_supersample(dur, n_E, n_I, dt; R_eff_max = 1.0, target = RK4_WARN_LAMBDA_H) -> Int
```

The smallest `supersample` whose substep keeps `|lambda * h| <= target`.

<a id='ConfigurableEpi.contact_matrix-Tuple{AbstractMatrix, Real}'></a> <a id='ConfigurableEpi.contact_matrix-Tuple{AbstractMatrix, Real}-1'></a> **`ConfigurableEpi.contact_matrix`** &mdash; *Method*.

```julia
contact_matrix(M, a_com) -> Matrix{Float64}
```

`(1 - a_com) I + a_com M`: within-location contact at `a_com = 0`, the radiation matrix at 1.

<a id='ConfigurableEpi.load_radiation_matrix-Tuple{Any}'></a> <a id='ConfigurableEpi.load_radiation_matrix-Tuple{Any}-1'></a> **`ConfigurableEpi.load_radiation_matrix`** &mdash; *Method*.

```julia
load_radiation_matrix(locations; path) -> Matrix{Float64}
```

Row-normalised contact matrix for `locations` (in order) from an `origin,destination,flow` table.
Errors when a location is absent, has a self-flow, or retains no flow to the others.

<a id='ConfigurableEpi.EpiModel'></a> <a id='ConfigurableEpi.EpiModel-1'></a> **`ConfigurableEpi.EpiModel`** &mdash; *Type*.

```julia
EpiModel(; vectorfield!, layout, stochastic, observation, hyperparams, priors, initial_state,
         initial_latent_variance = (;), initial_learned_variance = (;),
         initial_accumulator_variance = (;), forgetting_memory_days = (;),
         derived_hyperparameters = nothing)
```

Everything \[`build_inference`\](configurableepi.md#ConfigurableEpi.build_inference-Tuple{StateFilter, HyperMethod, EpiModel}) needs to fit and forecast one model:

- `vectorfield!`: the Petri vector field from \[`build_petri_vf`\](configurableepi.md#ConfigurableEpi.build_petri_vf-Tuple{Any, Any}).
- `layout`, `stochastic`: the [`StateLayout`](configurableepi.md#ConfigurableEpi.StateLayout) and [`StochasticUpdate`](configurableepi.md#ConfigurableEpi.StochasticUpdate).
- `observation`: a tuple of observation specs.
- `hyperparams`: every hyperparameter the rates, drivers and observation read, including the starting value of each learned one.
- `priors`: `name => ParameterDistribution` for the hyperparameters to learn (a subset of `hyperparams`, at least one).
- `initial_state`: the model-state vector at `t = 0`, or a function `hyperparams -> vector` when the seed depends on the parameters (an equilibrium `S(0) = N / R0`, say).
- `initial_*_variance`: by-name overrides of the initial variance of latent coefficient slots, learned slots (PF) and reset accumulators (EnKF), in unconstrained space.
- `forgetting_memory_days`: the model's default Liu-West forgetting memories (parameter => days), overridden per key by `LiuWest.forgetting_memory_days`.
- `derived_hyperparameters`: optional `hyper -> NamedTuple` summarised alongside the learned parameters by the particle filter.

<a id='ConfigurableEpi.initial_state'></a> <a id='ConfigurableEpi.initial_state-1'></a> **`ConfigurableEpi.initial_state`** &mdash; *Function*.

```julia
initial_state(model::EpiModel, hyperparams = model.hyperparams) -> Vector
```

The model-state vector at `t = 0` under `hyperparams`.

<a id='ConfigurableEpi.learned_names-Tuple{EpiModel}'></a> <a id='ConfigurableEpi.learned_names-Tuple{EpiModel}-1'></a> **`ConfigurableEpi.learned_names`** &mdash; *Method*.

The names of the learned hyperparameters.

## Inference {#Inference}

<a id='ConfigurableEpi.DEFAULT_LATENT_VARIANCE'></a> <a id='ConfigurableEpi.DEFAULT_LATENT_VARIANCE-1'></a> **`ConfigurableEpi.DEFAULT_LATENT_VARIANCE`** &mdash; *Constant*.

Default initial variance of a latent coefficient slot (unconstrained space): in a log chart, sd 0.45.

<a id='ConfigurableEpi.DEFAULT_MODEL_RELATIVE_SD'></a> <a id='ConfigurableEpi.DEFAULT_MODEL_RELATIVE_SD-1'></a> **`ConfigurableEpi.DEFAULT_MODEL_RELATIVE_SD`** &mdash; *Constant*.

Initial sd of each compartment as a fraction of its own initial value.

<a id='ConfigurableEpi.EngineSettings'></a> <a id='ConfigurableEpi.EngineSettings-1'></a> **`ConfigurableEpi.EngineSettings`** &mdash; *Type*.

```julia
EngineSettings(; dt, supersample, n_ahead, n_draws)
```

The run-level numbers every engine shares: the observation interval in days, RK4 substeps per interval, forecast horizon and forecast draw count.

<a id='ConfigurableEpi.build_inference-Tuple{StateFilter, HyperMethod, EpiModel}'></a> <a id='ConfigurableEpi.build_inference-Tuple{StateFilter, HyperMethod, EpiModel}-1'></a> **`ConfigurableEpi.build_inference`** &mdash; *Method*.

```julia
build_inference(filter, hyper, model::EpiModel; dt, supersample = 2, n_ahead, n_draws = 2000,
                rng = Random.default_rng(), kwargs...) -> InferenceEngine
build_inference(cfg::RunConfig, model::EpiModel; rng = Random.default_rng())
```

Build the inference engine for a `(filter, hyper)` pairing: [`UKF`](configurableepi.md#ConfigurableEpi.UKF) + [`Optimise`](configurableepi.md#ConfigurableEpi.Optimise), [`PF`](configurableepi.md#ConfigurableEpi.PF) + [`LiuWest`](configurableepi.md#ConfigurableEpi.LiuWest) or [`EnKF`](configurableepi.md#ConfigurableEpi.EnKF) + [`EKP`](configurableepi.md#ConfigurableEpi.EKP).
Drive it with [`fit_forecast!`](configurableepi.md#ConfigurableEpi.fit_forecast!).
The `RunConfig` form reads `dt`, `supersample`, `n_ahead` and `n_draws` from the run config.

<a id='ConfigurableEpi.fit_forecast!'></a> <a id='ConfigurableEpi.fit_forecast!-1'></a> **`ConfigurableEpi.fit_forecast!`** &mdash; *Function*.

```julia
fit_forecast!(engine, observations, forecast_number; update_range = eachindex(observations),
              emit_forecast = true) -> (; quantiles, fitted_means, summary, samples)
```

Assimilate `observations`, one entry per slot of the regular `dt` grid (a number or a vector, or `missing` where there is no observation: the filter then predicts through the slot without correcting), and forecast `n_ahead` steps ahead.
`forecast_number` counts fitted origins and drives the re-optimisation cadence.
The replay engines (UKF, EnKF) require the complete `update_range`; the online PF engine keeps its cloud between calls and `update_range` must start at the slot after the last one it assimilated (the default), so a replay is `reset!(engine.filter)` followed by the full range.
`fitted_means` covers `update_range` (a nowcast at a `missing` slot).
`quantiles` is `[horizon, quantile]` (`[horizon, observation, quantile]` for a multi-signal EnKF) and `samples` the predictive draws behind it (`nothing` for the analytic UKF); both are `nothing` when `emit_forecast = false`.
`summary` is a `(parameter, statistic, value)` table of estimates and diagnostics.

<a id='ConfigurableEpi.marginal_loglik-Tuple{Any, Any, Any}'></a> <a id='ConfigurableEpi.marginal_loglik-Tuple{Any, Any, Any}-1'></a> **`ConfigurableEpi.marginal_loglik`** &mdash; *Method*.

```julia
marginal_loglik(filter, ys, p) -> Real
```

Filter marginal log-likelihood of the observation vectors `ys` under hyperparameters `p` after `reset!`, accumulated at the promoted element type so it differentiates (`forward_trajectory(...).ll` sizes its buffers as Float64).
A `missing` entry is a grid slot without an observation: the filter predicts through it without correcting.

<a id='ConfigurableEpi.positive_cholesky!-Tuple{Any}'></a> <a id='ConfigurableEpi.positive_cholesky!-Tuple{Any}-1'></a> **`ConfigurableEpi.positive_cholesky!`** &mdash; *Method*.

```julia
positive_cholesky!(R)
```

Positive-definite Cholesky via `PositiveFactorizations.ldlt!`, which unlike its `cholesky!` wrapper accepts `ForwardDiff.Dual` matrices.

<a id='ConfigurableEpi.build_inference-Tuple{UKF, Optimise, EpiModel}'></a> <a id='ConfigurableEpi.build_inference-Tuple{UKF, Optimise, EpiModel}-1'></a> **`ConfigurableEpi.build_inference`** &mdash; *Method*.

```julia
build_inference(filter::UKF, hyper::Optimise, model; dt, supersample = 2, n_ahead, n_draws = 2000,
                rng = nothing, optimiser = DEFAULT_OPTIMISER_STAGES, adtype = AutoForwardDiff())
```

`optimiser` is a single optimiser, a tuple of them, or `(optimiser, options)` pairs; `rng` is unused.

<a id='ConfigurableEpi.PFLiuWestEngine'></a> <a id='ConfigurableEpi.PFLiuWestEngine-1'></a> **`ConfigurableEpi.PFLiuWestEngine`** &mdash; *Type*.

```julia
PFLiuWestEngine
```

PF + Liu-West: one persistent particle cloud carrying the learned hyperparameters in its tail, assimilating only the observations it has not yet seen.

<a id='ConfigurableEpi.build_pf_dynamics-Tuple{Any, StateLayout}'></a> <a id='ConfigurableEpi.build_pf_dynamics-Tuple{Any, StateLayout}-1'></a> **`ConfigurableEpi.build_pf_dynamics`** &mdash; *Method*.

```julia
build_pf_dynamics(dynamics, layout; rng = Random.default_rng(), learned = nothing, threads = false)
    -> pf_dynamics(x, u, p, t, noise = false)
```

Adapt the augmented `dynamics` from \[`build_full_dynamics`\](configurableepi.md#ConfigurableEpi.build_full_dynamics-Union{Tuple{S}, Tuple{L}, Tuple{M}, Tuple{N}, Tuple{Any, StochasticUpdate{L}, StateLayout{N, M, L, S}}} where {N, M, L, S}) to the `AdvancedParticleFilter` convention: `noise = true` draws the process noise (and fires the jump drivers) from `rng`, or from a per-thread pool when `threads = true`, so a seeded run reproduces for a fixed thread count.
With `learned` each particle's tail overrides `p` and is carried forward unchanged.

<a id='ConfigurableEpi.build_pf_measurement-Tuple{StateLayout, Tuple{Vararg{ObservationSpec}}, StochasticUpdate}'></a> <a id='ConfigurableEpi.build_pf_measurement-Tuple{StateLayout, Tuple{Vararg{ObservationSpec}}, StochasticUpdate}-1'></a> **`ConfigurableEpi.build_pf_measurement`** &mdash; *Method*.

```julia
build_pf_measurement(layout, obs_specs, stochastic; rng = Random.default_rng(), learned = nothing)
    -> pf_measure(x, u, p, t, noise = false)
```

The particle filter's measurement: the observation means (floored at `1e-6`), or with `noise = true` a draw from the exact observation distribution via `sample_observation`.

<a id='ConfigurableEpi.EnKFEKPEngine'></a> <a id='ConfigurableEpi.EnKFEKPEngine-1'></a> **`ConfigurableEpi.EnKFEKPEngine`** &mdash; *Type*.

```julia
EnKFEKPEngine
```

EnKF + EKP: replays the complete series each origin and recalibrates every `reopt_interval` origins, warm-starting from the previous outer ensemble.

<a id='ConfigurableEpi.LearnedHyperparams'></a> <a id='ConfigurableEpi.LearnedHyperparams-1'></a> **`ConfigurableEpi.LearnedHyperparams`** &mdash; *Type*.

```julia
LearnedHyperparams{H}
```

A block of `H` hyperparameters carried per particle in the tail after the model state (`offset == layout.total_dim`).
`extract(x)` reads them constrained as a NamedTuple; `to_unconstrained(nt)` maps a constrained NamedTuple to the tail's `SVector{H}`.

<a id='ConfigurableEpi.build_hyperparam_updater-Union{Tuple{LearnedHyperparams{H}}, Tuple{H}} where H'></a> <a id='ConfigurableEpi.build_hyperparam_updater-Union{Tuple{LearnedHyperparams{H}}, Tuple{H}} where H-1'></a> **`ConfigurableEpi.build_hyperparam_updater`** &mdash; *Method*.

```julia
build_hyperparam_updater(learned; discount = 0.95, jitter_floor_fraction = DEFAULT_JITTER_FLOOR_FRACTION,
                         forgetting_memory_days = (;), rng = Random.default_rng(), dt = 1.0) -> update!
```

`update!(particles, weights)` refreshes every particle's learned slots in place by the Liu-West kernel `θᵢ ← a θᵢ + (1 - a) θ̄ + ε`, `ε ~ N(0, (1 - a²) V)` with `a = (3δ - 1) / 2δ` and `θ̄, V` the weighted cloud moments, which preserves the weighted mean and variance.
The jitter variance is floored at `jitter_floor_fraction` of each prior's variance so a collapsed cloud can re-expand.
Parameters named in `forgetting_memory_days` are additionally pulled toward their prior by Kulhavý forgetting with `λ = exp(-dt / memory)`: the marginal `N(m, v)` becomes the geometric mean of itself and the prior.
Call between `correct!` and `predict!` on `state(pf).xprev` with `expweights(pf)`.

<a id='ConfigurableEpi.build_learned_hyperparams-Tuple{NamedTuple, StateLayout}'></a> <a id='ConfigurableEpi.build_learned_hyperparams-Tuple{NamedTuple, StateLayout}-1'></a> **`ConfigurableEpi.build_learned_hyperparams`** &mdash; *Method*.

```julia
build_learned_hyperparams(priors, layout) -> LearnedHyperparams
```

`priors` is a NamedTuple (keys equal to the prior names), a tuple of priors, or one prior.

<a id='ConfigurableEpi.DEFAULT_OPTIMISER_STAGES'></a> <a id='ConfigurableEpi.DEFAULT_OPTIMISER_STAGES-1'></a> **`ConfigurableEpi.DEFAULT_OPTIMISER_STAGES`** &mdash; *Constant*.

```julia
DEFAULT_OPTIMISER_STAGES
```

Adam first (its step is bounded by the learning rate however badly the filter log-posterior is scaled), then LBFGS to polish; each stage is `remake`d from the previous solution.

<a id='ConfigurableEpi.optimiser_stages-Tuple{Any}'></a> <a id='ConfigurableEpi.optimiser_stages-Tuple{Any}-1'></a> **`ConfigurableEpi.optimiser_stages`** &mdash; *Method*.

```julia
optimiser_stages(method) -> Tuple of (optimiser, options)
```

Normalise a single optimiser, a tuple of optimisers, or a tuple of `(optimiser, options)` pairs.

<a id='ConfigurableEpi.optimize_hyperparams-Tuple{Any, NamedTuple, ParameterPriorBundle}'></a> <a id='ConfigurableEpi.optimize_hyperparams-Tuple{Any, NamedTuple, ParameterPriorBundle}-1'></a> **`ConfigurableEpi.optimize_hyperparams`** &mdash; *Method*.

```julia
optimize_hyperparams(neg_logposterior, initial::NamedTuple, bundle::ParameterPriorBundle;
                     stages = DEFAULT_OPTIMISER_STAGES, adtype = AutoForwardDiff(), options = (;))
    -> (; θ, ll, retcode, unconstrained)
```

Minimise `neg_logposterior(u, _)` over the bundle's unconstrained coordinates from the constrained `initial` values.
`options` (such as `maxiters`) are merged over each stage's own; a stage whose result is non-finite or worse than the incumbent is discarded.

<a id='ConfigurableEpi.AugmentedEnsembleKalmanFilter'></a> <a id='ConfigurableEpi.AugmentedEnsembleKalmanFilter-1'></a> **`ConfigurableEpi.AugmentedEnsembleKalmanFilter`** &mdash; *Type*.

```julia
AugmentedEnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny = size(R2, 1),
                              p = nothing, Ts = 1.0, inflation = 1.0, rng = Xoshiro(),
                              threads = false, names = nothing)
```

Ensemble Kalman filter with `N` members drawn from `d0`, process noise `w ~ N(0, R1)` applied inside `dynamics` and measurement noise `v ~ N(0, R2)` inside `measurement`.
`inflation >= 1` multiplies the ensemble spread after each propagation; `threads` parallelises member propagation.

## Output {#Output}

<a id='ConfigurableEpi.DEFAULT_QS'></a> <a id='ConfigurableEpi.DEFAULT_QS-1'></a> **`ConfigurableEpi.DEFAULT_QS`** &mdash; *Constant*.

Default forecast quantile levels.

<a id='ConfigurableEpi.append_latent_audit!-Tuple{Any, Any, AbstractVector, AbstractUnitRange{<:Integer}}'></a> <a id='ConfigurableEpi.append_latent_audit!-Tuple{Any, Any, AbstractVector, AbstractUnitRange{<:Integer}}-1'></a> **`ConfigurableEpi.append_latent_audit!`** &mdash; *Method*.

```julia
append_latent_audit!(summary, latent_names, xt, latent_range; dt) -> summary
```

In-sample behaviour of each latent's filtered path (unconstrained chart): `is_sd`, the lag-1 autocorrelation `is_acf1`, the implied correlation time `is_tau_days = -dt / log(acf1)` (`NaN` when `acf1 <= 0`) and `is_drift`, the second-half mean minus the first-half mean.
Filtered paths absorb data innovations, so read `is_acf1` as a lower bound on the process's own.

<a id='ConfigurableEpi.append_latent_spread!-Tuple{Any, Any, AbstractMatrix}'></a> <a id='ConfigurableEpi.append_latent_spread!-Tuple{Any, Any, AbstractMatrix}-1'></a> **`ConfigurableEpi.append_latent_spread!`** &mdash; *Method*.

```julia
append_latent_spread!(summary, latent_names, log_sd) -> summary
```

Per-horizon predictive spread of each latent coefficient: `fc_log_sd_h<h>` (the unconstrained sd, `log_sd[h, l]`) and `fc_factor_h<h> = exp(sd)`, a multiplicative ±1sd spread for a log-chart latent.

<a id='ConfigurableEpi.asof_series-Tuple{Any}'></a> <a id='ConfigurableEpi.asof_series-Tuple{Any}-1'></a> **`ConfigurableEpi.asof_series`** &mdash; *Method*.

```julia
asof_series(triangle; r_k, date_col = :date, issue_col = :as_of, group_cols = ()) -> DataFrame
```

No-leakage as-of series at report date `r_k`: rows with `as_of <= r_k`, then the latest issue per reference date (within each `group_cols` series, e.g. `(:location,)`), sorted date-major.

<a id='ConfigurableEpi.backtest_forecast_rows-Tuple{AbstractMatrix, Any, Any, Any}'></a> <a id='ConfigurableEpi.backtest_forecast_rows-Tuple{AbstractMatrix, Any, Any, Any}-1'></a> **`ConfigurableEpi.backtest_forecast_rows`** &mdash; *Method*.

```julia
backtest_forecast_rows(quantiles, origin, target_dates, truth; qs = DEFAULT_QS, model_id,
                       date_col = :date, value_col = :counts[, locations, loc_col = :location]) -> DataFrame
```

One row per (horizon, quantile) with columns `origin, horizon, target_date, quantile, value, observed, model_id`, `observed` joined from `truth` at each target date.
A `[horizon, observation, quantile]` array with `locations` labelling the observations prepends a `location` column and joins on `(location, date)`.

<a id='ConfigurableEpi.forecast_ensemble-Tuple{Any, AbstractVector, Any}'></a> <a id='ConfigurableEpi.forecast_ensemble-Tuple{Any, AbstractVector, Any}-1'></a> **`ConfigurableEpi.forecast_ensemble`** &mdash; *Method*.

```julia
forecast_ensemble(filter, init_states, p; n_ahead, t0, dt = 1.0, u = Float64[], latent_range = 1:0, n_obs = 1)
    -> (samples, latent_samples)
```

Roll each state forward `n_ahead` steps with `sample_state` and draw a predictive observation with `sample_measurement` (no `correct!`).
`samples[h, j]` (or `[h, j, o]` for `n_obs > 1`) is the non-negative predictive observation; `latent_samples[h, j, l]` the requested latent slots in their unconstrained chart, the forecast-spread diagnostic.

<a id='ConfigurableEpi.forecast_quantiles-Tuple{AbstractMatrix}'></a> <a id='ConfigurableEpi.forecast_quantiles-Tuple{AbstractMatrix}-1'></a> **`ConfigurableEpi.forecast_quantiles`** &mdash; *Method*.

```julia
forecast_quantiles(samples; qs = DEFAULT_QS)
```

Per-horizon quantiles over draws: `[horizon, quantile]` from `samples[h, j]`, or `[horizon, observation, quantile]` from `samples[h, j, o]`.

<a id='ConfigurableEpi.forecast_states-Tuple{Any, Any, Any}'></a> <a id='ConfigurableEpi.forecast_states-Tuple{Any, Any, Any}-1'></a> **`ConfigurableEpi.forecast_states`** &mdash; *Method*.

```julia
forecast_states(kf, x0, R0; n_ahead, t0, dt = 1.0, u = Float64[], p) -> (means, covs)
```

Analytic Kalman forecast: seed a copy of `kf` at the filtered Gaussian `(x0, R0)` and roll `n_ahead` steps with `predict!` only, returning each horizon's state mean and covariance.

<a id='ConfigurableEpi.forecast_sample_rows-Tuple{AbstractMatrix, Any}'></a> <a id='ConfigurableEpi.forecast_sample_rows-Tuple{AbstractMatrix, Any}-1'></a> **`ConfigurableEpi.forecast_sample_rows`** &mdash; *Method*.

```julia
forecast_sample_rows(samples, target_dates; geo_value, disease, variable, resolution, metadata = (;))
forecast_sample_rows(samples, target_dates; geo_values, diseases, variables, resolution, metadata = (;))
```

Long draw-level rows from `samples[horizon, draw]` (or `[horizon, draw, signal]` with one label per signal): draw-major, with `Int32` draw ids.
`metadata` appends scalar or row-length columns after the routine ones (a backtest's `origin`, say).

<a id='ConfigurableEpi.write_forecast_samples-Tuple{AbstractString, DataFrames.DataFrame}'></a> <a id='ConfigurableEpi.write_forecast_samples-Tuple{AbstractString, DataFrames.DataFrame}-1'></a> **`ConfigurableEpi.write_forecast_samples`** &mdash; *Method*.

```julia
write_forecast_samples(path, table_or_tables) -> path
```

Write one sample table, or an iterable of schema-identical tables, to `samples.parquet` through DuckDB (`COPY ... FORMAT parquet`), combining the batches inside DuckDB.

<a id='ConfigurableEpi.DataLink'></a> <a id='ConfigurableEpi.DataLink-1'></a> **`ConfigurableEpi.DataLink`** &mdash; *Type*.

```julia
DataLink(schema[, value_column = :value, pivot_column = nothing])
```

How a DataFrame maps onto a schema: already wide (one column per observation name), or long with a `pivot_column` whose values are the observation names and a `value_column`.

<a id='ConfigurableEpi.ObservationSchema'></a> <a id='ConfigurableEpi.ObservationSchema-1'></a> **`ConfigurableEpi.ObservationSchema`** &mdash; *Type*.

```julia
ObservationSchema(names, time_column)
```

The observation names, in the order the measurement model emits them, and the time column.

<a id='ConfigurableEpi.build_observation_schema-Tuple{Tuple{Vararg{ObservationSpec}}}'></a> <a id='ConfigurableEpi.build_observation_schema-Tuple{Tuple{Vararg{ObservationSpec}}}-1'></a> **`ConfigurableEpi.build_observation_schema`** &mdash; *Method*.

```julia
build_observation_schema(obs_specs; time_column = :date)
```

The schema named by a tuple of observation specs, in their order.

<a id='ConfigurableEpi.build_observations-Union{Tuple{N}, Tuple{DataFrames.DataFrame, DataLink{N}}} where N'></a> <a id='ConfigurableEpi.build_observations-Union{Tuple{N}, Tuple{DataFrames.DataFrame, DataLink{N}}} where N-1'></a> **`ConfigurableEpi.build_observations`** &mdash; *Method*.

```julia
build_observations(df, link::DataLink; T = Float64) -> (; y, times)
build_observations(df, obs_specs; time_column = :date, value_column = :value, pivot_column = nothing, T = Float64)
```

Observation vectors `y::Vector{SVector{N, T}}` in schema order, one per time point, and the times.
Missing cells throw.

<a id='ConfigurableEpi.pivot_to_wide-Tuple{DataFrames.DataFrame, DataLink}'></a> <a id='ConfigurableEpi.pivot_to_wide-Tuple{DataFrames.DataFrame, DataLink}-1'></a> **`ConfigurableEpi.pivot_to_wide`** &mdash; *Method*.

```julia
pivot_to_wide(df, link::DataLink) -> DataFrame
```

Wide frame with the time column followed by one column per schema name, sorted by time.
A wide input is validated and sorted; a long one is unstacked on `link.pivot_column`.

<a id='ConfigurableEpi.require_complete_grid-Union{Tuple{N}, Tuple{DataFrames.DataFrame, DataLink{N}}} where N'></a> <a id='ConfigurableEpi.require_complete_grid-Union{Tuple{N}, Tuple{DataFrames.DataFrame, DataLink{N}}} where N-1'></a> **`ConfigurableEpi.require_complete_grid`** &mdash; *Method*.

```julia
require_complete_grid(df, link) -> df
```

Assert exactly one row for every (observation, time) cell of a long frame, reporting every missing and duplicated cell at once (the pivot alone would take the first duplicate silently).
