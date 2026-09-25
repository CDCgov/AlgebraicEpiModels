# AlgebraicEpiMech API {#AlgebraicEpiMech-API}

Reference for the exported API of `AlgebraicEpiMech`, generated from docstrings.

<a id='AlgebraicEpiMech.AtCompartment'></a> <a id='AlgebraicEpiMech.AtCompartment-1'></a> **`AlgebraicEpiMech.AtCompartment`** &mdash; *Type*.

```julia
struct AtCompartment <: ObservationTarget
```

```julia
AtCompartment(species)
```

Sample a SPECIES: prevalence.
Every species whose (flattened) name begins with `species` gains a catalytic tap `X -> X + O`, carrying its own detection rate — whoever is in the compartment is currently detectable (although detection does not cause removal)

--------------------------------------------------------------------------------

**Fields**

- `species::Symbol`

<a id='AlgebraicEpiMech.AtEvent'></a> <a id='AlgebraicEpiMech.AtEvent-1'></a> **`AlgebraicEpiMech.AtEvent`** &mdash; *Type*.

```julia
struct AtEvent <: ObservationTarget
```

```julia
AtEvent(transition)
```

Record a TRANSITION: incidence.
Every transition whose (flattened) name begins with `transition` gains an output arc into an accumulator, so it fires at that transition's own rate and counts exactly one observation per occurrence.

`AtEvent(:transmission)` counts infections as they happen, which is what decouples a reporting delay from the latent period — the delay is then whatever the chain adds and nothing else.

--------------------------------------------------------------------------------

**Fields**

- `transition::Symbol`

<a id='AlgebraicEpiMech.CompartmentalModel'></a> <a id='AlgebraicEpiMech.CompartmentalModel-1'></a> **`AlgebraicEpiMech.CompartmentalModel`** &mdash; *Type*.

```julia
abstract type CompartmentalModel <: EpiMechModel
```

Abstract base type for compartmental epidemiological models.

Compartmental models define the specific disease dynamics and transitions between epidemiological states (e.g., S→I→R) as undirected wiring diagrams.

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.CompleteCrossImmunity'></a> <a id='AlgebraicEpiMech.CompleteCrossImmunity-1'></a> **`AlgebraicEpiMech.CompleteCrossImmunity`** &mdash; *Type*.

```julia
struct CompleteCrossImmunity <: MultiStrainModel
```

Complete cross-immunity multistrain model.

In the complete cross-immunity model:

- All strains share a common susceptible pool
- Infection by any strain confers immunity to all strains
- Typing: UninfectedInfectedTyping (typed S vs I/R compartments)
- Composition: Strains compete for susceptibles via shared S depletion

**Fields**

- `number_of_strains::Int`: Number of competing strains in the model
- `strain_names::Vector{Symbol}`: Names for each strain (e.g., \[:h1n1, :h3n2\])

**Examples**

```julia
# Create a 2-strain model with custom names
multistrain = CompleteCrossImmunity([:wild_type, :variant])

# Create a 3-strain model with auto-generated names
multistrain = CompleteCrossImmunity(3)  # Creates [:strain_1, :strain_2, :strain_3]

# Compose to create competing strain SIR
typing = UninfectedInfectedTyping()
strain_typed = create_model(typing, multistrain)
sir_typed = create_model(typing, SIR())
combined = typed_product(sir_typed, strain_typed)
```

--------------------------------------------------------------------------------

**Fields**

- `number_of_strains::Int64`
- `strain_names::Vector{Symbol}`

<a id='AlgebraicEpiMech.ContactStratification'></a> <a id='AlgebraicEpiMech.ContactStratification-1'></a> **`AlgebraicEpiMech.ContactStratification`** &mdash; *Type*.

```julia
struct ContactStratification <: Stratification
```

Contact-based stratification for epidemiological models.

Represents stratifications where populations are divided into strata with contact-based interactions (full contact matrix between all pairs).
This generalizes age groups, geographic regions modelled as cross-region contact (rather than explicit movement), risk groups, etc.

Multiple contact stratifications can be stacked via `compose_stratifications` to create product strata (e.g., age × geography).

**Fields**

- `stratum_names::Vector{Symbol}`: Names of strata (e.g., \[:child, :adult\] or \[:urban, :rural\])
- `label::Symbol`: Semantic label for the stratification (e.g., :age, :geography, :risk)

Reflexive boxes (disease, reversion, and waning) are controlled via the `include_reflexives` keyword on `create_model`, not on the struct itself.

**Examples**

```julia
# Age stratification
age = ContactStratification([:child, :adult], :age)

# Geographic stratification
geo = ContactStratification([:urban, :rural], :geography)

# Compose to create age × geography strata
typing = OnePopulationTyping()
sir = create_model(typing, SIR())
age_sir = typed_product(sir, create_model(typing, age))
age_geo_sir = typed_product(age_sir, create_model(typing, geo))
# Result: 4 strata (childxurban, childxrural, adultxurban, adultxrural)
```

See also: [`AgeStratification`](algebraicepimech.md#AlgebraicEpiMech.AgeStratification-Tuple{Vector{Symbol}}), [`GeographicStratification`](algebraicepimech.md#AlgebraicEpiMech.GeographicStratification-Tuple{Vector{Symbol}})

--------------------------------------------------------------------------------

**Fields**

- `stratum_names::Vector{Symbol}`
- `label::Symbol`

<a id='AlgebraicEpiMech.EpiMechModel'></a> <a id='AlgebraicEpiMech.EpiMechModel-1'></a> **`AlgebraicEpiMech.EpiMechModel`** &mdash; *Type*.

```julia
abstract type EpiMechModel
```

Abstract base type for all epidemiological-mechanical models in the AlgebraicEpiMech framework.

All concrete epidemiological-mechanical model types should be subtypes of `EpiMechModel`.
This abstract type serves as the root of the type hierarchy for models that combine epidemiological dynamics with mechanical or algebraic structures.

**Extended help**

Subtypes of `EpiMechModel` should implement the necessary interface methods for their specific model formulation.

**See also**

- Related concrete model types (define as needed)

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.EpidemiologicalTyping'></a> <a id='AlgebraicEpiMech.EpidemiologicalTyping-1'></a> **`AlgebraicEpiMech.EpidemiologicalTyping`** &mdash; *Type*.

```julia
abstract type EpidemiologicalTyping
```

Abstract base type for epidemiological typing strategies.

An `EpidemiologicalTyping` describes how populations and transitions are typed when constructing epidemiological models.
\[`type_system`\](algebraicepimech.md#AlgebraicEpiMech.type_system-Tuple{EpidemiologicalTyping, Vararg{Any}}) materializes the strategy as the `LabelledPetriNet` used as the codomain of a typed Petri net.

**Reference**

[Libkind et al. (2023) *An algebraic framework for structured epidemic modelling*](https://royalsocietypublishing.org/rsta/article/380/2233/20210309/112239)

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.FullHistory'></a> <a id='AlgebraicEpiMech.FullHistory-1'></a> **`AlgebraicEpiMech.FullHistory`** &mdash; *Type*.

```julia
struct FullHistory <: ImmuneHistoryMode
```

Full (unordered) immune history: uninfected individuals are indexed by the **set** of strains they are immune to (`2^n` classes).
Immunity accumulates — reversion routes to `h ∪ {i}`.

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.ImmuneHistory'></a> <a id='AlgebraicEpiMech.ImmuneHistory-1'></a> **`AlgebraicEpiMech.ImmuneHistory`** &mdash; *Type*.

```julia
struct ImmuneHistory{M<:ImmuneHistoryMode} <: MultiStrainModel
```

```julia
ImmuneHistory{M<:ImmuneHistoryMode} <: MultiStrainModel
```

Immune-history combination of the available strain/subtype names and `ImmuneHistoryMode`.

**Fields**

- `strain_names::Vector{Symbol}`: strain names (e.g. `[:current, :invader]`)
- `mode::M`: `FullHistory()` or `LatestInfection()`

**Examples**

```julia
typing = UninfectedInfectedTyping()

seirs   = create_model(typing, SEIRS())
history = create_model(typing, ImmuneHistory([:current, :invader]))

model = typed_product(seirs, history)   # immune-history-resolved SEIRS
# States: S_naive, S_current, …, E_invader_from_current, I_…, R_…
```

--------------------------------------------------------------------------------

**Fields**

- `strain_names::Vector{Symbol}`
- `mode::ImmuneHistoryMode`

<a id='AlgebraicEpiMech.ImmuneHistory-Tuple{Int64}'></a> <a id='AlgebraicEpiMech.ImmuneHistory-Tuple{Int64}-1'></a> **`AlgebraicEpiMech.ImmuneHistory`** &mdash; *Method*.

```julia
ImmuneHistory(n::Int; kwargs...)
```

Auto-name `n` strains as `strain_1 … strain_n`.

<a id='AlgebraicEpiMech.ImmuneHistoryMode'></a> <a id='AlgebraicEpiMech.ImmuneHistoryMode-1'></a> **`AlgebraicEpiMech.ImmuneHistoryMode`** &mdash; *Type*.

```julia
abstract type ImmuneHistoryMode
```

Abstract base type for selections of how much immune history is retained.

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.LatestInfection'></a> <a id='AlgebraicEpiMech.LatestInfection-1'></a> **`AlgebraicEpiMech.LatestInfection`** &mdash; *Type*.

```julia
struct LatestInfection <: ImmuneHistoryMode
```

Latest-infection status: uninfected individuals are indexed by their **most recent** infection (`n + 1` classes).
Reversion overwrites immune status — routes to `{i}`.

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.MultiStrainModel'></a> <a id='AlgebraicEpiMech.MultiStrainModel-1'></a> **`AlgebraicEpiMech.MultiStrainModel`** &mdash; *Type*.

```julia
abstract type MultiStrainModel <: EpiMechModel
```

Abstract base type for multistrain epidemiological models.

Multistrain models define how multiple pathogen strains interact through different assumptions about cross-immunity.
These compose with compartmental models via typed_product to create strain-structured disease dynamics.

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.NoCrossImmunity'></a> <a id='AlgebraicEpiMech.NoCrossImmunity-1'></a> **`AlgebraicEpiMech.NoCrossImmunity`** &mdash; *Type*.

```julia
struct NoCrossImmunity <: MultiStrainModel
```

No cross-immunity multistrain model.

In the no cross-immunity model:

- Each strain operates independently with no interaction
- Strains act as independent strata (similar to age groups)
- Typing: OnePopulationTyping (all compartments have the same type)
- Composition: Creates parallel disease dynamics per strain via typed_product

**Fields**

- `number_of_strains::Int`: Number of independent strains in the model
- `strain_names::Vector{Symbol}`: Names for each strain (e.g., \[:h1n1, :h3n2, :b\])

**Examples**

```julia
# Create a 3-strain model with custom names
multistrain = NoCrossImmunity([:h1n1, :h3n2, :b])

# Create a 2-strain model with auto-generated names
multistrain = NoCrossImmunity(2)  # Creates [:strain_1, :strain_2]

# Compose to create strain-structured SIR
typing = OnePopulationTyping()
strain_typed = create_model(typing, multistrain)
sir_typed = create_model(typing, SIR())
combined = typed_product(sir_typed, strain_typed)
```

--------------------------------------------------------------------------------

**Fields**

- `number_of_strains::Int64`
- `strain_names::Vector{Symbol}`

<a id='AlgebraicEpiMech.ObservationChainLayout'></a> <a id='AlgebraicEpiMech.ObservationChainLayout-1'></a> **`AlgebraicEpiMech.ObservationChainLayout`** &mdash; *Type*.

```julia
struct ObservationChainLayout{N}
```

```julia
ObservationChainLayout(source_name, obs_names)
```

Metadata for one observation chain.

**Fields**

- `source_name`: Flattened name of the event or compartment being observed
- `obs_names`: Observation-state names in stage order
- `cumulative_name`: Terminal observation-state name, derived from `obs_names`

--------------------------------------------------------------------------------

**Fields**

- `source_name::Symbol`
- `obs_names::NTuple{N, Symbol} where N`
- `cumulative_name::Symbol`

<a id='AlgebraicEpiMech.ObservationLayout'></a> <a id='AlgebraicEpiMech.ObservationLayout-1'></a> **`AlgebraicEpiMech.ObservationLayout`** &mdash; *Type*.

```julia
struct ObservationLayout{N, C<:Tuple, M}
```

```julia
ObservationLayout(obs_names, chains)
```

Metadata for all observation chains in an augmented Petri net.

**Fields**

- `obs_names`: All flattened observation-state names in Petri-net order
- `chains`: Observation chains in first-encounter order
- `cumulative_names`: Terminal state of each chain, derived from `chains`

--------------------------------------------------------------------------------

**Fields**

- `obs_names::NTuple{N, Symbol} where N`
- `chains::Tuple`
- `cumulative_names::NTuple{M, Symbol} where M`

<a id='AlgebraicEpiMech.ObservationTarget'></a> <a id='AlgebraicEpiMech.ObservationTarget-1'></a> **`AlgebraicEpiMech.ObservationTarget`** &mdash; *Type*.

```julia
abstract type ObservationTarget
```

```julia
ObservationTarget
```

What an observation attaches to.
[`AtEvent`](algebraicepimech.md#AlgebraicEpiMech.AtEvent) records a transition (incidence); [`AtCompartment`](algebraicepimech.md#AlgebraicEpiMech.AtCompartment) samples a species (prevalence).

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.OnePopulationTyping'></a> <a id='AlgebraicEpiMech.OnePopulationTyping-1'></a> **`AlgebraicEpiMech.OnePopulationTyping`** &mdash; *Type*.

```julia
struct OnePopulationTyping <: EpidemiologicalTyping
```

```julia
OnePopulationTyping(; population_type = :Population)
```

Typing strategy in which every compartment belongs to one population type and therefore has the same available stratifications.

**Examples**

```julia
typing = OnePopulationTyping()
typing = OnePopulationTyping(population_type = :CityPopulation)
```

--------------------------------------------------------------------------------

**Fields**

- `population_type::Symbol`: The symbolic name for the population type.

<a id='AlgebraicEpiMech.SEI'></a> <a id='AlgebraicEpiMech.SEI-1'></a> **`AlgebraicEpiMech.SEI`** &mdash; *Type*.

```julia
struct SEI <: CompartmentalModel
```

Susceptible-Exposed-Infected (SEI) compartmental model.

The SEI model extends SI with an exposed compartment:

- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious

Transitions:

- S + I → E + I (exposure/transmission)
- E → I (progression to infectious)

The SEI model represents diseases with an incubation period but no recovery.

--------------------------------------------------------------------------------

**Fields**

- `number_E_stages::Int64`
- `number_I_stages::Int64`
- `number_of_states::Int64`

<a id='AlgebraicEpiMech.SEIR'></a> <a id='AlgebraicEpiMech.SEIR-1'></a> **`AlgebraicEpiMech.SEIR`** &mdash; *Type*.

```julia
struct SEIR <: CompartmentalModel
```

Susceptible-Exposed-Infected-Recovered (SEIR) compartmental model.

The SEIR model extends SEI with recovery:

- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious
- Recovered (R): Individuals who have recovered and gained immunity

Transitions:

- S + I → E + I (exposure/transmission) \[from SEI\]
- E → I (progression to infectious) \[from SEI\]
- I → R (recovery) \[extension\]

Built by extending SEI model with recovery transition.

--------------------------------------------------------------------------------

**Fields**

- `number_E_stages::Int64`
- `number_I_stages::Int64`
- `number_of_states::Int64`

<a id='AlgebraicEpiMech.SEIRS'></a> <a id='AlgebraicEpiMech.SEIRS-1'></a> **`AlgebraicEpiMech.SEIRS`** &mdash; *Type*.

```julia
struct SEIRS <: CompartmentalModel
```

Susceptible-Exposed-Infected-Recovered-Susceptible (SEIRS) compartmental model.

The SEIRS model extends SEIR with waning immunity:

- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious
- Recovered (R): Individuals who have recovered but may lose immunity

Transitions:

- S + I → E + I (exposure/transmission) \[from SEIR\]
- E → I (progression to infectious) \[from SEIR\]
- I → R (recovery) \[from SEIR\]
- R → S (waning immunity) \[extension\]

Built by extending SEIR model with waning transition.

--------------------------------------------------------------------------------

**Fields**

- `number_E_stages::Int64`
- `number_I_stages::Int64`
- `number_of_states::Int64`

<a id='AlgebraicEpiMech.SEIS'></a> <a id='AlgebraicEpiMech.SEIS-1'></a> **`AlgebraicEpiMech.SEIS`** &mdash; *Type*.

```julia
struct SEIS <: CompartmentalModel
```

Susceptible-Exposed-Infected-Susceptible (SEIS) compartmental model.

The SEIS model extends SEI with waning immunity (reversion):

- Susceptible (S): Individuals who can become infected
- Exposed (E): Individuals who are infected but not yet infectious
- Infected (I): Individuals who are infectious

Transitions:

- S + I → E + I (exposure/transmission) \[from SEI\]
- E → I (progression to infectious) \[from SEI\]
- I → S (waning immunity/reversion) \[extension\]

The SEIS model represents diseases with incubation period but no lasting immunity.

--------------------------------------------------------------------------------

**Fields**

- `number_E_stages::Int64`
- `number_I_stages::Int64`
- `number_of_states::Int64`

<a id='AlgebraicEpiMech.SI'></a> <a id='AlgebraicEpiMech.SI-1'></a> **`AlgebraicEpiMech.SI`** &mdash; *Type*.

```julia
struct SI <: CompartmentalModel
```

Susceptible-Infected (SI) compartmental model.

The SI model includes:

- Susceptible (S): Individuals who can become infected
- Infected (I): Individuals who are infectious

Transitions:

- S + I → I + I (infection/transmission)

The SI model represents endemic diseases with no recovery.

--------------------------------------------------------------------------------

**Fields**

- `number_I_stages::Int64`
- `number_of_states::Int64`

<a id='AlgebraicEpiMech.SIR'></a> <a id='AlgebraicEpiMech.SIR-1'></a> **`AlgebraicEpiMech.SIR`** &mdash; *Type*.

```julia
struct SIR <: CompartmentalModel
```

Susceptible-Infected-Recovered (SIR) compartmental model.

The SIR model extends SI with recovery:

- Susceptible (S): Individuals who can become infected
- Infected (I): Individuals who are infectious
- Recovered (R): Individuals who have recovered and gained immunity

Transitions:

- S + I → I + I (infection/transmission) \[from SI\]
- I → R (recovery) \[extension\]

Built by extending SI model with recovery transition.

--------------------------------------------------------------------------------

**Fields**

- `number_I_stages::Int64`
- `number_of_states::Int64`

<a id='AlgebraicEpiMech.SIS'></a> <a id='AlgebraicEpiMech.SIS-1'></a> **`AlgebraicEpiMech.SIS`** &mdash; *Type*.

```julia
struct SIS <: CompartmentalModel
```

Susceptible-Infected-Susceptible (SIS) compartmental model.

The SIS model extends SI with waning immunity (reversion):

- Susceptible (S): Individuals who can become infected
- Infected (I): Individuals who are infectious

Transitions:

- S + I → I + I (infection/transmission) \[from SI\]
- I → S (waning immunity/reversion) \[extension\]

The SIS model represents diseases with no lasting immunity.

--------------------------------------------------------------------------------

**Fields**

- `number_I_stages::Int64`
- `number_of_states::Int64`

<a id='AlgebraicEpiMech.Stratification'></a> <a id='AlgebraicEpiMech.Stratification-1'></a> **`AlgebraicEpiMech.Stratification`** &mdash; *Type*.

```julia
abstract type Stratification <: EpiMechModel
```

Represents a stratification structure for epidemiological models.

--------------------------------------------------------------------------------

**Fields**

<a id='AlgebraicEpiMech.UninfectedInfectedTyping'></a> <a id='AlgebraicEpiMech.UninfectedInfectedTyping-1'></a> **`AlgebraicEpiMech.UninfectedInfectedTyping`** &mdash; *Type*.

```julia
struct UninfectedInfectedTyping <: EpidemiologicalTyping
```

```julia
UninfectedInfectedTyping(; uninfected_type = :Uninfected, infected_type = :Infected)
```

Typing strategy that distinguishes uninfected and infected populations.
The two populations may have different stratifications.

**Examples**

```julia
typing = UninfectedInfectedTyping()
typing = UninfectedInfectedTyping(
    uninfected_type = :Susceptible,
    infected_type = :Infectious,
)
```

See also: \[`type_system`\](algebraicepimech.md#AlgebraicEpiMech.type_system-Tuple{EpidemiologicalTyping, Vararg{Any}}), [`OnePopulationTyping`](algebraicepimech.md#AlgebraicEpiMech.OnePopulationTyping)

--------------------------------------------------------------------------------

**Fields**

- `uninfected_type::Symbol`: The symbolic name for the uninfected population.
- `infected_type::Symbol`: The symbolic name for the infected population.

<a id='AlgebraicEpiMech.AgeStratification-Tuple{Vector{Symbol}}'></a> <a id='AlgebraicEpiMech.AgeStratification-Tuple{Vector{Symbol}}-1'></a> **`AlgebraicEpiMech.AgeStratification`** &mdash; *Method*.

```julia
AgeStratification(
    names::Vector{Symbol}
) -> ContactStratification
```

```julia
AgeStratification(age_group_names::Vector{Symbol})
AgeStratification(names...)
```

Convenience constructor for age-based contact stratification.

Creates a `ContactStratification` with label `:age`.
Provided for backward compatibility and semantic clarity.

**Examples**

```julia
age = AgeStratification([:child, :adult, :elderly])
age = AgeStratification(:child, :adult, :elderly)  # splatting syntax
```

<a id='AlgebraicEpiMech.GeographicStratification-Tuple{Vector{Symbol}}'></a> <a id='AlgebraicEpiMech.GeographicStratification-Tuple{Vector{Symbol}}-1'></a> **`AlgebraicEpiMech.GeographicStratification`** &mdash; *Method*.

```julia
GeographicStratification(
    names::Vector{Symbol}
) -> ContactStratification
```

```julia
GeographicStratification(location_names::Vector{Symbol})
GeographicStratification(names...)
```

Convenience constructor for geography-based contact stratification.

Creates a `ContactStratification` with label `:geography`.
Represents geographic regions with cross-region contact (not movement-based models).

**Examples**

```julia
geo = GeographicStratification([:urban, :suburban, :rural])
geo = GeographicStratification(:urban, :suburban, :rural)  # splatting syntax
```

<a id='AlgebraicEpiMech.add_disease_progression!-Tuple{Any, Any, Any, OnePopulationTyping}'></a> <a id='AlgebraicEpiMech.add_disease_progression!-Tuple{Any, Any, Any, OnePopulationTyping}-1'></a> **`AlgebraicEpiMech.add_disease_progression!`** &mdash; *Method*.

```julia
add_disease_progression!(
    uwd,
    from_junction,
    to_junction,
    typing::OnePopulationTyping
) -> Any
```

Add disease progression mechanism: from*compartment → to*compartment

Used for transitions like E→I, I→R, etc. (infected → infected transitions)

<a id='AlgebraicEpiMech.add_infection!-Tuple{Any, Any, Any, Any, OnePopulationTyping}'></a> <a id='AlgebraicEpiMech.add_infection!-Tuple{Any, Any, Any, Any, OnePopulationTyping}-1'></a> **`AlgebraicEpiMech.add_infection!`** &mdash; *Method*.

```julia
add_infection!(
    uwd,
    infectee_junction,
    infector_junction,
    first_infected_junction,
    typing::OnePopulationTyping
) -> Any
```

Add infection mechanism: Infectee + Infector → first_infected + Infector

General infection mechanism where an infectee (typically S) interacts with an infector (typically I) to produce a newly infected individual that enters the first\*infected compartment, while the infector remains unchanged.
The `first*infected_junction` can be:

- An I stage for SIR-like models (direct infection)
- An E stage for SEIR-like models (exposure before infectiousness)

The `typing` argument determines the population types used.

<a id='AlgebraicEpiMech.add_reversion_progression!-Tuple{Any, Any, Any, OnePopulationTyping}'></a> <a id='AlgebraicEpiMech.add_reversion_progression!-Tuple{Any, Any, Any, OnePopulationTyping}-1'></a> **`AlgebraicEpiMech.add_reversion_progression!`** &mdash; *Method*.

```julia
add_reversion_progression!(
    uwd,
    from_junction,
    to_junction,
    typing::OnePopulationTyping
) -> Any
```

Add reversion progression mechanism: e.g. R → S (infected → uninfected transitions)

Used for transitions from infected compartments back to uninfected compartments.

<a id='AlgebraicEpiMech.add_uninfected_density_progression!-Tuple{Any, Any, Any, OnePopulationTyping}'></a> <a id='AlgebraicEpiMech.add_uninfected_density_progression!-Tuple{Any, Any, Any, OnePopulationTyping}-1'></a> **`AlgebraicEpiMech.add_uninfected_density_progression!`** &mdash; *Method*.

```julia
add_uninfected_density_progression!(
    uwd,
    from_junction,
    to_junction,
    typing::OnePopulationTyping
) -> Any
```

Add uninfected density progression mechanisms.
This is useful for transitions like waning partial immunity (uninfected → uninfected transitions, e.g., V → S).

<a id='AlgebraicEpiMech.attach_observation-Tuple{Any, ObservationTarget}'></a> <a id='AlgebraicEpiMech.attach_observation-Tuple{Any, ObservationTarget}-1'></a> **`AlgebraicEpiMech.attach_observation`** &mdash; *Method*.

```julia
attach_observation(
    pn,
    target::ObservationTarget;
    n_stages,
    prefix
) -> Any
```

```julia
attach_observation(pn, target; n_stages = 1, prefix = :O) -> LabelledPetriNet
```

Attach an observation chain to `pn`, returning a new net.
The result is the colimit described at the top of this file, materialized in one direct pass (see the note there for why it is not built by pushout).

`n_stages` Erlang delay stages are added per chain: the first receives the observation, each subsequent one takes flow from the last, and the final stage is cumulative (no outflow), i.e. the accumulator the observation model reads.

Reapplying the same target, prefix, and stage count is idempotent.
Reusing a target and prefix for an incompatible chain shape throws an `ArgumentError`; choose a different prefix when both observation layouts are required.

Chains are grouped by stratum.
For [`AtEvent`](algebraicepimech.md#AlgebraicEpiMech.AtEvent) the group is the transition's net product — for infection routes `S_x + I_y -> E_x + I_y` that is `E_x`, so all routes infecting `x` count into `x`'s chain.
For [`AtCompartment`](algebraicepimech.md#AlgebraicEpiMech.AtCompartment) each matched species gets its own chain.

Apply this after all `typed_product` composition so chains are grouped over the final strata.

<a id='AlgebraicEpiMech.compose_stratifications-Tuple{ContactStratification, ContactStratification}'></a> <a id='AlgebraicEpiMech.compose_stratifications-Tuple{ContactStratification, ContactStratification}-1'></a> **`AlgebraicEpiMech.compose_stratifications`** &mdash; *Method*.

```julia
compose_stratifications(
    a::ContactStratification,
    b::ContactStratification
) -> ContactStratification
```

Compose two `ContactStratification`s via Cartesian product.

**Arguments**

- `a::ContactStratification`: First contact stratification
- `b::ContactStratification`: Second contact stratification

**Returns**

- `ContactStratification`: New contact stratification representing the product of `a` and `b`, with combined stratum names and a composite label.

**Examples**

```julia
age = ContactStratification([:child, :adult], :age)
geo = ContactStratification([:urban, :rural], :geography)
age_geo = compose_stratifications(age, geo)
# Resulting stratum names: [:childxurban, :childxrural, :adultxurban, :adultxrural]
# Resulting label: :age_x_geography
```

<a id='AlgebraicEpiMech.create_model-Tuple{EpidemiologicalTyping, EpiMechModel}'></a> <a id='AlgebraicEpiMech.create_model-Tuple{EpidemiologicalTyping, EpiMechModel}-1'></a> **`AlgebraicEpiMech.create_model`** &mdash; *Method*.

```julia
create_model(
    typing::EpidemiologicalTyping,
    model::EpiMechModel;
    include_reflexives
) -> Catlab.CategoricalAlgebra.Pointwise.ACSetTransformations.StructACSetTransformation{ACSets.Schemas.TypeLevelBasicSchema{Symbol, Tuple{:T, :S, :I, :O}, Tuple{(:it, :I, :T), (:is, :I, :S), (:ot, :O, :T), (:os, :O, :S)}, Tuple{:Name}, Tuple{(:tname, :T, :Name), (:sname, :S, :Name)}, Tuple{}}, Comp, AlgebraicPetri.LabelledPetriNet, AlgebraicPetri.LabelledPetriNet} where Comp<:NamedTuple
```

Create a typed Petri net model from an epidemiological typing strategy and model specification.

This is the unified entry point for creating all model types in AlgebraicEpiMech.
Through multiple dispatch on the `model` parameter, it supports any subtype of `EpiMechModel`.
The undirected wiring diagram (UWD) provides an intermediate structural representation, recording how compartments and mechanisms connect independently of the final typed Petri net constructed by `oapply_typed`.

**Functionality Overview**

The `create_model` function builds the typed Petri net representation of the epidemiological-mechanical model in four steps:

1. **Build Type System**: Materializes the typing strategy as a `LabelledPetriNet`
2. **Create UWD**: Builds an undirected wiring diagram with model structure and box names
3. **Generate Transition Names**: Creates unique names for ODE parameters via dispatch
4. **Apply Typing**: Uses `oapply_typed` to create a typed Petri net

The resulting typed Petri net can be used directly for ODE simulation or composed with other typed Petri nets via `typed_product` to build complex hierarchical models.

**Arguments**

- `typing::EpidemiologicalTyping`: The strategy defining possible compartment and transition types.
- `model::EpiMechModel`: The model specification (uses dispatch for different types)

**Keyword Arguments**

Keyword arguments are forwarded to `create_model_uwd`.
Model-specific keywords include:

- `include_reflexives::Bool=true` (for `ContactStratification`): Controls whether per-stratum reflexive boxes (disease, reversion, and waning) are added.
  Required for `typed_product` composition; disable for standalone use to avoid zero-effect transitions.

**Returns**

- `ACSetTransformation`: A typed Petri net with two components accessible via:

- `dom(typed_model)`: Extract the underlying `LabelledPetriNet` for ODE solving

- `codom(typed_model)`: Access the Petri-net type system

**Model-Specific Transition Naming**

Transition names are generated automatically based on model type:

- **Compartmental**: `transmission_S_I`, `E_to_I`, `I_to_R` (describes flow between compartments)
- **Age Stratification**: `child_child`, `child_adult`, `adult_child` (infectee_infector pattern)
- **Multistrain**: Strain identifiers like `h1n1`, `h3n2` (enables clean composition)

These names become ODE parameters and differ from box names (`:transmission`, `:density`) used for composition.

**Examples**

```julia
# Basic compartmental model
typing = OnePopulationTyping()
seir = create_model(typing, SEIR())
seir_pn = dom(seir)  # Extract for ODE solving
# Transitions: :transmission_S_I, :E_to_I, :I_to_R

# Multi-stage compartments (gamma-distributed delays)
si_multi = create_model(typing, SI(number_I_stages=3))
# Transitions: :transmission_S_I1, :transmission_S_I2, :transmission_S_I3, :I1_to_I2, :I2_to_I3

# Age-structured model
age_strat = AgeStratification([:child, :adult, :elderly])
age_model = create_model(typing, age_strat)
# Transitions: :child_child, :child_adult, :child_elderly, :adult_child, ...

# Multistrain model (no cross-immunity)
strains = NoCrossImmunity([:h1n1, :h3n2, :seasonal_b])
strain_model = create_model(typing, strains)

# Compose models via typed_product
sir = create_model(typing, SIR())
age = create_model(typing, AgeStratification([:child, :adult]))
age_sir = typed_product(sir, age)
age_sir_pn = dom(age_sir)
# Result: S_child, I_child, R_child, S_adult, I_adult, R_adult compartments
# Transition names become tuples: (:transmission_S_I, :child_adult)
```

**Compositional Design**

Models expressed as typed Petri nets compose via `typed_product` using their common type-system codomain when box names match.
The `vectorfield_flat` function automatically handles tuple transition names in composed models for ODE parameter assignment.

See also: \[`create_model_uwd`\](algebraicepimech.md#AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, CompartmentalModel}), \[`generate_transition_names`\](algebraicepimech.md#AlgebraicEpiMech.generate_transition_names-Tuple{Any, CompartmentalModel}), \[`type_system`\](algebraicepimech.md#AlgebraicEpiMech.type_system-Tuple{EpidemiologicalTyping, Vararg{Any}}), `typed_product` (from `AlgebraicPetri.TypedPetri`), [`vectorfield_flat`](algebraicepimech.md#AlgebraicEpiMech.vectorfield_flat-Tuple{AlgebraicPetri.AbstractPetriNet})

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, CompartmentalModel}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, CompartmentalModel}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::CompartmentalModel
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Create an undirected wiring diagram (UWD) for a compartmental model within a specific typing.

This function returns the undirected wiring diagram that defines the compartmental structure and transitions, using the type names from the typing.
The compartments are typed according to the typing's population structure.
The returned UWD does not declare an outer boundary/ports; `oapply_typed` derives the boundary from its junctions when materializing the typed Petri net.

**Arguments**

- `typing::EpidemiologicalTyping`: The population typing defining type names
- `model::CompartmentalModel`: The compartmental model instance (SI(), SEI(), SIR(), SEIR(), etc.)

**Returns**

- Undirected wiring diagram defining the compartmental model structure with typing-specific types

**Examples**

```julia
# Create SIR model UWD with single population typing
typing = OnePopulationTyping(population_type = :Individual)
sir_uwd = create_model_uwd(typing, SIR())  # Uses Individual type

# Create SEIR model UWD with uninfected/infected typing
typing = UninfectedInfectedTyping(uninfected_type = :Susceptible, infected_type = :Infectious)
seir_uwd = create_model_uwd(typing, SEIR())  # S::Susceptible, E,I,R::Infectious
```

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, ContactStratification}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, ContactStratification}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::ContactStratification;
    include_reflexives
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct an undirected wiring diagram (UWD) for contact strata.

Creates an UWD where transmission boxes represent contact patterns between different strata.
Example strata include age groups, risk levels, or demographic divisions, but only covers instantaneous transmission dynamics, e.g. movement between geographic regions would require additional demographic modeling.

The UWD structure depends on the typing:

- `OnePopulationTyping`: Single junction per age group (combined susceptible/infected)
- `UninfectedInfectedTyping`: Two junctions per age group (uninfected and infected)

All transmission boxes are created with the `:transmission` name, enabling composition with compartmental models that have `:transmission` boxes (SI, SIR, SEIR, etc.).

**Arguments**

- `typing::EpidemiologicalTyping`: The population typing defining type system
- `model::ContactStratification`: The contact stratification configuration with stratum names

**Returns**

- Undirected wiring diagram with stratum junctions and transmission boxes for composition

**Mathematical Structure**

For n strata, creates n² transmission boxes representing the contact matrix:

- Diagonal boxes (i→i): within-stratum transmission
- Off-diagonal boxes (i→j, i≠j): between-stratum transmission

**Composition Behavior**

When composed with a compartmental model UWD via `typed_product`, the transmission boxes will align based on the `:transmission` name, allowing the contact structure to modulate the disease dynamics defined in the compartmental model.

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SEIRS}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SEIRS}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::SEIRS
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct a complete SEIRS model UWD by extending SEI with recovery and waning immunity.

Builds on the SEI base model (S→E→I dynamics with multi-stage E and I support) and adds:

- R (recovered) compartment
- Recovery transition: I_last → R
- Waning immunity transition: R → S

Represents diseases with latent periods and temporary immunity.
The dual multi-stage capability enables independent control of latent and infectious period distributions.
Recovery occurs from the last I stage, and waning immunity returns individuals from R to S.

**Usage with oapply_typed**

Apply to typing with `oapply_typed` to create a typed Petri net for composition.

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SEIR}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SEIR}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::SEIR
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct a complete SEIR model UWD by extending SEI with recovery.

Builds on the SEI base model (S→E→I dynamics with multi-stage E and I support) and adds:

- R (recovered) compartment
- Recovery transition: I_last → R

The dual multi-stage capability (independent E and I stages) enables independent control of latent and infectious period distributions.
Recovery always occurs from the last I stage.

**Usage with oapply_typed**

Apply to typing with `oapply_typed` to create a typed Petri net for composition.

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SEIS}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SEIS}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::SEIS
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct a complete SEIS model UWD by extending SEI with waning immunity.

Builds on the SEI base model (S→E→I dynamics with multi-stage E and I support) and adds:

- Reversion transition: I_last → S (waning immunity)

Represents diseases with latent periods but no lasting immunity.
The dual multi-stage capability enables independent control of latent and infectious period distributions.
Reversion occurs from the last I stage back to S.

**Usage with oapply_typed**

Apply to typing with `oapply_typed` to create a typed Petri net for composition.

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SIR}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SIR}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::SIR
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct a complete SIR model UWD by extending SI with recovery.

Builds on the SI base model (S→I dynamics with multi-stage I support) and adds:

- R (recovered) compartment
- Recovery transition: I_last → R

The multi-stage I capability from SI is preserved, enabling gamma-distributed infectious periods.
Recovery always occurs from the last I stage, ensuring proper sequencing.

**Usage with oapply_typed**

Apply to typing with `oapply_typed` to create a typed Petri net for composition.

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SIS}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, SIS}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::SIS
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct a complete SIS model UWD by extending SI with waning immunity.

Builds on the SI base model (S→I dynamics with multi-stage I support) and adds:

- Reversion transition: I_last → S (waning immunity)

Represents diseases where recovery returns individuals to susceptible state.
The multi-stage I capability enables realistic infectious period distributions.
Reversion occurs from the last I stage back to S.

**Usage with oapply_typed**

Apply to typing with `oapply_typed` to create a typed Petri net for composition.

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, Union{SEI, SI}}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{EpidemiologicalTyping, Union{SEI, SI}}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::EpidemiologicalTyping,
    model::Union{SEI, SI}
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct a complete undirected wiring diagram (UWD) for SI or SEI models.

Populates an initially empty UWD with compartments and mechanisms via `setup_basic!`.
These are the base models that other compartmental models extend from.

**Usage with oapply_typed**

The returned UWD can be applied to the typing using `oapply_typed` to create a typed Petri net, enabling composition with other typed Petri nets (e.g., demographic processes, interventions).

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{OnePopulationTyping, ImmuneHistory}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{OnePopulationTyping, ImmuneHistory}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(_::OnePopulationTyping, _::ImmuneHistory)
```

Immune-history stratification requires the uninfected/infected type split.

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{OnePopulationTyping, NoCrossImmunity}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{OnePopulationTyping, NoCrossImmunity}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::OnePopulationTyping,
    multistrain::NoCrossImmunity
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct an undirected wiring diagram (UWD) for a no cross-immunity multistrain model.

Creates a strain-stratified UWD where each strain operates independently.
The strains are represented as separate junctions, and each strain has its own set of disease transition boxes (`:transmission`, `:disease`, `:reversion`) that will compose with corresponding boxes from a compartmental model via `typed_product`.

All box types are always included; `typed_product` will only compose boxes that exist in both UWDs, naturally filtering out non-matching transitions.
That filtering is silent — a transition present in one factor and absent from the other is dropped without error — which is why each factor carries a reflexive box per transition type it wants preserved.

There is deliberately no reflexive observation box.
Observation is attached to the COMPOSED net by `attach_observation`, so a stratification factor never has to know it exists.

**Arguments**

- `typing::OnePopulationTyping`: The required single-population typing
- `multistrain::NoCrossImmunity`: The multistrain model configuration with strain names

**Returns**

- Undirected wiring diagram with strain junctions and transition boxes for composition

**Examples**

```julia
# Create a 3-strain model with custom names
typing = OnePopulationTyping()
multistrain = NoCrossImmunity([:h1n1, :h3n2, :b])
strain_uwd = create_model_uwd(typing, multistrain)

# Compose with compartmental model
sir_typed = create_model(typing, SIR())
strain_typed = create_model(typing, multistrain)
combined = typed_product(sir_typed, strain_typed)
```

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{UninfectedInfectedTyping, CompleteCrossImmunity}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{UninfectedInfectedTyping, CompleteCrossImmunity}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::UninfectedInfectedTyping,
    multistrain::CompleteCrossImmunity
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct an undirected wiring diagram (UWD) for a complete cross-immunity multistrain model.

Creates a strain-structured UWD where all strains share a common susceptible pool but have separate infected compartments.
Infection by any strain depletes the shared susceptible pool and confers immunity to all strains.

The UWD has:

- One shared uninfected/susceptible junction
- N infected strain junctions
- Transmission boxes connecting shared susceptible to each strain's infected
- Disease progression boxes for each strain's infected compartments
- Reversion boxes to return to shared susceptible pool

**Arguments**

- `typing::UninfectedInfectedTyping`: The required uninfected/infected typing
- `multistrain::CompleteCrossImmunity`: The multistrain model configuration with strain names

**Returns**

- Undirected wiring diagram with shared susceptible and strain-specific infected junctions

**Examples**

```julia
# Create a 2-strain competing model
typing = UninfectedInfectedTyping()
multistrain = CompleteCrossImmunity([:wild_type, :variant])
strain_uwd = create_model_uwd(typing, multistrain)

# Compose with compartmental model
sir_typed = create_model(typing, SIR())
strain_typed = create_model(typing, multistrain)
combined = typed_product(sir_typed, strain_typed)
```

<a id='AlgebraicEpiMech.create_model_uwd-Tuple{UninfectedInfectedTyping, ImmuneHistory}'></a> <a id='AlgebraicEpiMech.create_model_uwd-Tuple{UninfectedInfectedTyping, ImmuneHistory}-1'></a> **`AlgebraicEpiMech.create_model_uwd`** &mdash; *Method*.

```julia
create_model_uwd(
    typing::UninfectedInfectedTyping,
    model::ImmuneHistory
) -> Catlab.WiringDiagrams.RelationDiagrams.TypedUnnamedRelationDiagram{Symbol, Symbol, Symbol}
```

Construct the UWD for the `ImmuneHistory` stratification **factor** over a `UninfectedInfectedTyping`.

This is a stratification factor (like `ContactStratification`), meant to be `typed_product`-composed with a disease model — the disease model supplies `S→E→I→R`, while this factor supplies the immune-status structure.

- Uninfected junctions `U_h`, one per immune-history class.
- Infected junctions `(h,i)` = "history `h`, currently fighting strain `i`", one per (class, susceptible-strain) pair.
- **`:transmission`** (escape-selective): `U_h + (h',i) → (h,i)` for every susceptible class `h` (`i ∉ h`) and every infector `(h',i)`.
  A class immune to `i` simply has no such box, so the pullback drops that infection — the escape.
- **`:disease`** is reflexive on each `(h,i)`, so the base model's progression runs inside a fixed `(h,i)`.
- **`:reversion`** is off-diagonal: `(h,i) → U_{recover}` with `recover = h∪{i}` (`FullHistory`) or `{i}` (`LatestInfection`).
  This is the one stratum-changing move — recovering from `i` folds it into the immune history.
- **`:waning`** reflexive on each `U_h` (inert with SEIRS; lets a waning-bearing base compose).

**Examples**

```julia
typing = UninfectedInfectedTyping()
history = create_model(typing, ImmuneHistory([:current, :invader]))
model = typed_product(create_model(typing, SEIRS()), history)
```

<a id='AlgebraicEpiMech.flatten_symbols-Tuple{Tuple}'></a> <a id='AlgebraicEpiMech.flatten_symbols-Tuple{Tuple}-1'></a> **`AlgebraicEpiMech.flatten_symbols`** &mdash; *Method*.

```julia
flatten_symbols(name::Tuple) -> Symbol
```

Recursively flatten a tuple of symbols and join them with underscores.

**Examples**

```julia
flatten_symbols((:x, :y, :z)) # returns :x_y_z
flatten_symbols(((:x, :y), :z)) # returns :x_y_z
flatten_symbols((((:a, :b), :c), :d)) # returns :a_b_c_d
```

<a id='AlgebraicEpiMech.generate_transition_names-Tuple{Any, CompartmentalModel}'></a> <a id='AlgebraicEpiMech.generate_transition_names-Tuple{Any, CompartmentalModel}-1'></a> **`AlgebraicEpiMech.generate_transition_names`** &mdash; *Method*.

```julia
generate_transition_names(
    uwd,
    model::CompartmentalModel
) -> Vector{Symbol}
```

Generate transition names from UWD structure for compartmental models.

For boxes with 2 ports (one input, one output), generates names in the format `input_to_output`.
For boxes with 4 ports (two inputs, two outputs), generates names in the format `box_name_input1_input2` where inputs are sorted in reverse alphabetical order for consistency (e.g., `transmission_S_I`).

**Arguments**

- `uwd`: The undirected wiring diagram with species names in junction :variable attributes
- `model::CompartmentalModel`: The compartmental model (used for dispatch)

**Returns**

- `Vector{Symbol}`: Vector of unique transition names, one per box in the UWD

**Examples**

```julia
# Used internally by create_model
typing = OnePopulationTyping()
uwd = create_model_uwd(typing, SEIR())
names = generate_transition_names(uwd, SEIR())
# Returns: [:transmission_S_I, :E_to_I, :I_to_R]
```

See also: \[`create_model`\](algebraicepimech.md#AlgebraicEpiMech.create_model-Tuple{EpidemiologicalTyping, EpiMechModel})

<a id='AlgebraicEpiMech.generate_transition_names-Tuple{Any, ContactStratification}'></a> <a id='AlgebraicEpiMech.generate_transition_names-Tuple{Any, ContactStratification}-1'></a> **`AlgebraicEpiMech.generate_transition_names`** &mdash; *Method*.

```julia
generate_transition_names(
    uwd,
    model::ContactStratification
) -> Vector{Symbol}
```

Generate transition names from UWD structure for age stratification models.

For transmission boxes with 4 ports (two inputs, two outputs), generates names capturing the age-to-age contact pattern in the format `infectee_infector`.
This naming convention clearly identifies which age group is being infected (infectee) by which age group (infector), enabling interpretation of age-structured contact matrices.

For boxes with 2 ports (one input, one output), generates names in the format `input_to_output`.
This fallback handles any non-standard box types that might appear in extended models.

**Arguments**

- `uwd`: The undirected wiring diagram with age group names in junction :variable attributes
- `model::AgeStratification`: The age stratification model (used for dispatch)

**Returns**

- `Vector{Symbol}`: Vector of transition names, one per box in the UWD

**Examples**

```julia
# Used internally by create_model
typing = OnePopulationTyping()
age_strat = AgeStratification([:child, :adult])
uwd = create_model_uwd(typing, age_strat)
names = generate_transition_names(uwd, age_strat)
# Returns: [:child_child, :child_adult, :adult_child, :adult_adult]
# Representing: child←child, child←adult, adult←child, adult←adult transmission
```

See also: \[`create_model`\](algebraicepimech.md#AlgebraicEpiMech.create_model-Tuple{EpidemiologicalTyping, EpiMechModel}), [`ContactStratification`](algebraicepimech.md#AlgebraicEpiMech.ContactStratification), [`AgeStratification`](algebraicepimech.md#AlgebraicEpiMech.AgeStratification-Tuple{Vector{Symbol}})

<a id='AlgebraicEpiMech.generate_transition_names-Tuple{Any, ImmuneHistory}'></a> <a id='AlgebraicEpiMech.generate_transition_names-Tuple{Any, ImmuneHistory}-1'></a> **`AlgebraicEpiMech.generate_transition_names`** &mdash; *Method*.

```julia
generate_transition_names(
    uwd,
    model::ImmuneHistory
) -> Vector{Symbol}
```

Generate transition names for the `ImmuneHistory` stratification factor.
These become the second component of the composed `typed_product` transition names.

- **`:transmission`** (4-port) → `infect_<strain>_<susceptible-class>_by_<infector-class>`, exposing the infecting strain (recovered from the infector junction) so the composed name stays keyable per strain.
- **reflexive / reversion boxes** (2–3 port) → the first junction's variable, mirroring `ContactStratification` — e.g. the `:disease` reflexives and the off-diagonal `:reversion` on `(h,i)` are named by that `(h,i)`, and `:waning` by its `U_h`.
  Composition with the base disambiguates them (base name differs).

**Examples**

```julia
factor = create_model_uwd(UninfectedInfectedTyping(), ImmuneHistory([:current, :invader]))
names = generate_transition_names(factor, ImmuneHistory([:current, :invader]))
# transmission names begin `infect_current…` / `infect_invader…`
```

<a id='AlgebraicEpiMech.observation_layout-Tuple{Any}'></a> <a id='AlgebraicEpiMech.observation_layout-Tuple{Any}-1'></a> **`AlgebraicEpiMech.observation_layout`** &mdash; *Method*.

```julia
observation_layout(pn; prefix) -> ObservationLayout
```

```julia
observation_layout(pn::LabelledPetriNet; prefix::Symbol = :O)
```

Return deterministic observation-chain metadata for an augmented Petri net.

The result is an [`ObservationLayout`](algebraicepimech.md#AlgebraicEpiMech.ObservationLayout) with fields:

- `obs_names`: all flattened observation state names in Petri-net order
- `chains`: ordered [`ObservationChainLayout`](algebraicepimech.md#AlgebraicEpiMech.ObservationChainLayout) values
- `cumulative_names`: flattened terminal observation state for each chain

This centralizes the observation naming/ordering contract, so downstream packages do not each re-implement observation-state parsing.
`prefix` must match the prefix passed to \[`attach_observation`\](algebraicepimech.md#AlgebraicEpiMech.attach_observation-Tuple{Any, ObservationTarget}).
The function recognises the `<prefix>_<source>_<stage>` leaf that `attach_observation` produces; a chain named otherwise is not recognised as one.

<a id='AlgebraicEpiMech.type_system-Tuple{EpidemiologicalTyping, Vararg{Any}}'></a> <a id='AlgebraicEpiMech.type_system-Tuple{EpidemiologicalTyping, Vararg{Any}}-1'></a> **`AlgebraicEpiMech.type_system`** &mdash; *Method*.

```julia
type_system(
    typing::EpidemiologicalTyping,
    population_transitions...
) -> AlgebraicPetri.LabelledPetriNet
```

```julia
type_system(typing::EpidemiologicalTyping, population_transitions...)
```

Materialize an epidemiological typing strategy as the `LabelledPetriNet` used as the codomain of a typed Petri net.
Concrete `EpidemiologicalTyping` subtypes must implement this interface.

<a id='AlgebraicEpiMech.type_system-Tuple{OnePopulationTyping, Vararg{Any}}'></a> <a id='AlgebraicEpiMech.type_system-Tuple{OnePopulationTyping, Vararg{Any}}-1'></a> **`AlgebraicEpiMech.type_system`** &mdash; *Method*.

```julia
type_system(
    typing::OnePopulationTyping,
    population_transitions...
) -> AlgebraicPetri.LabelledPetriNet
```

```julia
type_system(typing::OnePopulationTyping, population_transitions...)
```

Create the single-population Petri-net type system described by `typing`.
Additional transition specifications are appended to the standard `:transmission`, `:disease`, `:reversion`, and `:waning` transitions.

**Examples**

```julia
typing = OnePopulationTyping(population_type = :Individual)
codomain = type_system(
    typing,
    :birth => (:Individual => (:Individual, :Individual)),
)
```

<a id='AlgebraicEpiMech.type_system-Tuple{UninfectedInfectedTyping, Vararg{Any}}'></a> <a id='AlgebraicEpiMech.type_system-Tuple{UninfectedInfectedTyping, Vararg{Any}}-1'></a> **`AlgebraicEpiMech.type_system`** &mdash; *Method*.

```julia
type_system(
    typing::UninfectedInfectedTyping,
    population_transitions...
) -> AlgebraicPetri.LabelledPetriNet
```

```julia
type_system(typing::UninfectedInfectedTyping, population_transitions...)
```

Create the uninfected/infected Petri-net type system described by `typing`.
Additional transition specifications are appended to the standard `:transmission`, `:disease`, `:reversion`, and `:waning` transitions.

**Examples**

```julia
typing = UninfectedInfectedTyping(
    uninfected_type = :Susceptible,
    infected_type = :Infectious,
)
codomain = type_system(
    typing,
    :vaccination => (:Susceptible => :Susceptible),
)
```

<a id='AlgebraicEpiMech.vectorfield_flat-Tuple{AlgebraicPetri.AbstractPetriNet}'></a> <a id='AlgebraicEpiMech.vectorfield_flat-Tuple{AlgebraicPetri.AbstractPetriNet}-1'></a> **`AlgebraicEpiMech.vectorfield_flat`** &mdash; *Method*.

```julia
vectorfield_flat(
    pn::AlgebraicPetri.AbstractPetriNet
) -> AlgebraicEpiMech.var"#vectorfield!#vectorfield_flat##8"{AlgebraicEpiMech._NamePositions, AlgebraicEpiMech._NamePositions, AlgebraicEpiMech._NamePositions, Vector{Vector{Tuple{Int64, Int64}}}, Vector{Vector{Tuple{Int64, Int64}}}, Vector{T}, Vector{T1}, Int64, Int64} where {T, T1}
```

Generate an ODE vectorfield function from a Petri net using mass action kinetics as per `AlgebraicPetri.vectorfield`.
The only difference is that species and transition names are flattened using `flatten_symbols`, converting nested tuples into single symbols joined by underscores.

**Arguments**

- `pn::AbstractPetriNet`: A Petri net representing the reaction network

**Returns**

A function `(du, u, p, t) -> du` where:

- `du`: Array/Dict to store computed derivatives
- `u`: Current state (species concentrations), indexed by flattened species names
- `p`: Parameters (rate constants), indexed by flattened transition names
- `t`: Current time

This function complies with SciML conventions for in-place ODE vectorfields.

**Performance**

Everything that can be resolved from the net alone — flattened names, the mass-action input lists, the nonzero stoichiometry — is resolved once at build time, and the right-hand side is a sparse loop over arcs rather than a dense loop over `transitions × species`.
Name lookup into the arguments is also done once per concrete argument type: `u`, `du` and `p` may be anything whose `propertynames` lists the flattened names in the same order as its integer indexing (`LVector`, `NamedTuple`, ...), and that position map is cached on first sight of the type; an `AbstractDict` is looked up by name on every call.
