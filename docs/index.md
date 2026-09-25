# AlgebraicEpiModels

AlgebraicEpiModels is a pair of Julia packages for building compartmental epidemic models by algebraic composition [Libkind et al. (2023)](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0309) and fitting them to surveillance data.

  | Package                                          | What it does                                                                                                                                                                                                                                                                                                                                                                               |
  | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | [AlgebraicEpiMech](packages/algebraicepimech.md) | Builds compartmental models as typed Petri nets (*nets*) by algebraic compostions: 1) **Algebraic pullback** combines template models, such as SIR and SEIRS, with stratification models, and models of multistrain and immune-history structure. 2) **Algebraic pushout** which combine core composed models with observation delay chains. Any composed net becomes an ODE vector field. |
  | [ConfigurableEpi](packages/configurableepi.md)   | Turns an AlgebraicEpiMech net into a stochastic state-space model with latent drivers and an observation model, then fits it to a count series using filtering methods and gives utilities for forecasting.                                                                                                                                                                                |

## How the pieces fit

```mermaid
flowchart TB
    subgraph AEM["AlgebraicEpiMech — mechanistic structure"]
        direction LR
        A["Model vocabulary<br/><b>typing + disease templates</b><br/><code>create_model</code>"]
        B["Algebraic composition<br/><b>strata · strains · immune history</b><br/><code>typed_product</code> · pullback"]
        C["Observation structure<br/><b>target + delay chain</b><br/><code>attach_observation</code> · pushout"]
        D(["Typed Petri net<br/><b>named states + transitions</b>"])
        A --> B --> C --> D
    end

    subgraph CE["ConfigurableEpi — statistical model, inference and forecasts"]
        direction LR
        E["State-space dynamics<br/><b>rates · latent drivers · layout</b><br/><code>build_petri_vf</code> · <code>StateLayout</code>"]
        F["Model contract<br/><b>likelihood · priors · EpiModel</b>"]
        G["Inference engine<br/><b>UKF · PF · EnKF</b><br/>Optimise · Liu-West · EKP"]
        H(["Forecast products<br/><b>fits · quantiles · samples · backtests</b><br/><code>fit_forecast!</code>"])
        E --> F --> G --> H
    end

    AEM ==>|"typed Petri net hand-off"| CE

    classDef mech fill:#e8f3f8,stroke:#14799e,color:#102a36,stroke-width:1.5px
    classDef infer fill:#f2ebf8,stroke:#76529b,color:#2d1d3b,stroke-width:1.5px
    class A,B,C,D mech
    class E,F,G,H infer
    style AEM fill:#f8fbfc,stroke:#14799e,color:#102a36,stroke-width:2px
    style CE fill:#fbf9fc,stroke:#76529b,color:#2d1d3b,stroke-width:2px
```

AlgebraicEpiMech governs the structure of the epidemic and surveillance mechanism, producing a Petri net whose species and transitions have predictable, unique names.
That net is the package boundary: ConfigurableEpi supplies its rates, latent processes and statistical observation model, then fits and forecasts it.

## Where next

- [Getting started](getting-started.md): installation and a first model.
- [Examples](examples/compartmental_models.md): executed walkthroughs, from building models to fitting and forecasting.
- API reference for [AlgebraicEpiMech](api/algebraicepimech.md) and [ConfigurableEpi](api/configurableepi.md).
