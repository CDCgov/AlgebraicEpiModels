# Getting started

## Installation

The packages need Julia 1.11 or later and are not yet registered.
Add them from GitHub, AlgebraicEpiMech first, because ConfigurableEpi depends on it:

```julia
using Pkg
Pkg.add(url = "https://github.com/CDCgov/AlgebraicEpiModels", subdir = "AlgebraicEpiMech")
Pkg.add(url = "https://github.com/CDCgov/AlgebraicEpiModels", subdir = "ConfigurableEpi")
```

Drawing Petri nets with `to_graphviz` needs the [Graphviz](https://graphviz.org/download/) `dot` executable on the `PATH`.

To work on the packages themselves, clone the repository and instantiate the package you are working on; each package's test environment is a workspace project:

```bash
git clone https://github.com/CDCgov/AlgebraicEpiModels
cd AlgebraicEpiModels
julia --project=ConfigurableEpi -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## A first model

Build an SIR model, look at its structure and solve it as an ODE:

```julia
using AlgebraicEpiMech, AlgebraicPetri, Catlab, LabelledArrays, OrdinaryDiffEqTsit5

pn = dom(create_model(OnePopulationTyping(), SIR()))
snames(pn), tnames(pn)  # ([:S, :I, :R], [:transmission_S_I, :I_to_R])

u0 = LVector(S = 990.0, I = 10.0, R = 0.0)
p = LVector(transmission_S_I = 0.5 / 1000, I_to_R = 0.25)
sol = solve(ODEProblem(vectorfield_flat(pn), u0, (0.0, 120.0), p), Tsit5())
```

[Compartmental models](examples/compartmental_models.md) continues from here, and [Stratified models](examples/stratified_models.md) and [Multistrain models and immune history](examples/multistrain_immune_history.md) show composition.

## A first fit

[Inference engines](examples/inference_engines.md) takes a model through the ConfigurableEpi contract: describe it as an `EpiModel`, build an engine with `build_inference`, and call `fit_forecast!` on a count series.
[ConfigurableEpi](packages/configurableepi.md) describes the parts of an `EpiModel` and the run configuration.
