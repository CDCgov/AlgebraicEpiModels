# AlgebraicEpiMech

AlgebraicEpiMech builds compartmental epidemic models as typed Petri nets, following the algebraic framework of [Libkind et al. (2023), _An algebraic framework for structured epidemic modelling_](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0309).
Species are compartments, transitions are flows, and each model is typed over a small type system so that models built separately can be composed.

## Typing

An `EpidemiologicalTyping` fixes the type system every model is mapped into; `type_system(typing)` materialises it as a Petri net.

- `OnePopulationTyping()`: every compartment has one population type, so every compartment is stratified the same way.
- `UninfectedInfectedTyping()`: uninfected and infected compartments have different types and can be stratified differently.
  Competing strains and immune history need this.

Transitions are typed by role: transmission, disease progression, reversion (waning back to susceptible), and observation.

## Templates

`create_model(typing, model)` builds a typed Petri net from a template, via an undirected wiring diagram (`create_model_uwd`) assembled from mechanisms such as `add_infection!` and `add_disease_progression!`.

  | Template        | Compartments |
  | --------------- | ------------ |
  | `SI`, `SIS`     | S, I         |
  | `SEI`, `SEIS`   | S, E, I      |
  | `SIR`           | S, I, R      |
  | `SEIR`, `SEIRS` | S, E, I, R   |

`number_E_stages` and `number_I_stages` split a compartment into sequential stages for phase-type dwell times, for example `SEIR(number_E_stages = 2, number_I_stages = 3)`.

Every transition gets a unique name (`generate_transition_names`): a single-input transition is `input_to_output` (`E1_to_E2`), and a multi-input one is `box_input1_input2` (`transmission_S_I2`).

## Composition

Structure is added by building a second model over the same typing and taking the typed product with `AlgebraicPetri.TypedPetri.typed_product`.

- **Stratification.** `AgeStratification`, `GeographicStratification` and the general `ContactStratification` replicate each compartment per stratum, keep progression within strata, and give one transmission transition per (infectee, infector) pair of strata.
  Stratifications stack with `*` or `compose_stratifications`.
- **Strains.** `NoCrossImmunity` (independent strains, over `OnePopulationTyping`) and `CompleteCrossImmunity` (strains competing for one susceptible pool, over `UninfectedInfectedTyping`).
- **Immune history.** `ImmuneHistory(strains; mode)` indexes susceptibles by the strains they are immune to, with `FullHistory()` ($2^n$ classes) or `LatestInfection()` ($n + 1$ classes).
  Escape and history updates follow from the product.

Composed species and transitions have tuple names such as `(:S, :child)`; `flatten_symbols` joins them into `:S_child`.

## Observation

`attach_observation(pn, target; n_stages)` adds an Erlang observation delay chain to a built net, by pushout:

- `AtCompartment(:I)` samples a compartment at a rate times its occupancy (prevalence-type signals).
- `AtEvent(:transmission)` records a transition's flow (incidence-type signals).

Attach observation after composition, so that each stratum gets its own chain.
`observation_layout` reports the chains of an augmented net.

## Vector fields

`vectorfield_flat(pn)` returns an in-place mass-action ODE right-hand side `f!(du, u, p, t)` that indexes state and parameters by flattened name, so `LVector`s and `NamedTuple`s work directly.
ConfigurableEpi builds on the same nets with `build_petri_vf`, which lets transition rates depend on latent processes, hyperparameters and time.

## Examples

- [Compartmental models](../examples/compartmental_models.md)
- [Stratified models](../examples/stratified_models.md)
- [Multistrain models and immune history](../examples/multistrain_immune_history.md)

The full list of exports is in the [API reference](../api/algebraicepimech.md).
