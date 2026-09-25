# ConfigurableEpi API

Reference for the exported API of `ConfigurableEpi`, generated from docstrings and grouped by layer.

```@autodocs
Modules = [ConfigurableEpi]
Order = [:module]
```

## Configuration

```@autodocs
Modules = [ConfigurableEpi]
Private = false
Pages = ["config.jl"]
```

## Model

```@autodocs
Modules = [ConfigurableEpi]
Private = false
Pages = ["model/priors.jl", "model/layout.jl", "model/processes.jl", "model/dynamics.jl",
         "model/observation.jl", "model/seasonality.jl", "model/ascertainment.jl",
         "model/day_of_week.jl", "model/initialisation.jl", "model/radiation_mixing.jl",
         "model/epi_model.jl"]
```

## Inference

```@autodocs
Modules = [ConfigurableEpi]
Private = false
Pages = ["inference/engine.jl", "inference/ukf.jl", "inference/particle_filter.jl",
         "inference/ensemble.jl", "inference/learned_hyperparams.jl", "inference/optimise.jl",
         "inference/augmented_enkf.jl"]
```

## Output

```@autodocs
Modules = [ConfigurableEpi]
Private = false
Pages = ["output/forecast.jl", "output/samples.jl", "output/data_linkage.jl"]
```
