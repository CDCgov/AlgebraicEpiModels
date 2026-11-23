# Helpers for constructing compartmental model UWDs
# Not exported - used internally by create_model_uwd methods

"""
Create the Susceptible (S) compartment junction.

The returned junction ID is used directly when adding infection and reversion mechanisms
(for example, I→S in SIS).
"""
function set_S_junction!(uwd::RelationDiagram, typing::EpidemiologicalTyping)
    error("set_S_junction! not implemented for typing type $(typeof(typing)).")
end

function set_S_junction!(uwd::RelationDiagram, typing::OnePopulationTyping)
    pop_type = typing.population_type
    return add_junction!(uwd, pop_type, variable = :S)
end

function set_S_junction!(uwd::RelationDiagram, typing::UninfectedInfectedTyping)
    uninfected_type = typing.uninfected_type
    return add_junction!(uwd, uninfected_type, variable = :S)
end

"""
Create stratum junctions: one per stratum for `OnePopulationTyping`, or an uninfected and
infected pair per stratum for `UninfectedInfectedTyping`.

Returns a tuple of junction IDs for the stratum:
- For `OnePopulationTyping`: (stratum_junction, stratum_junction)
- For `UninfectedInfectedTyping`: (uninfected_stratum_junction, infected_stratum_junction)
"""
function set_stratum_junction!(
        uwd::RelationDiagram, typing::OnePopulationTyping, stratum::Symbol
    )
    pop_type = typing.population_type
    stratum_junction = add_junction!(uwd, pop_type, variable = stratum)
    return (stratum_junction, stratum_junction)
end

function set_stratum_junction!(
        uwd::RelationDiagram, typing::UninfectedInfectedTyping, stratum::Symbol
    )
    uninfected_type = typing.uninfected_type
    infected_type = typing.infected_type
    # Use unique variable names by prefixing with type
    uninfected_stratum_junction = add_junction!(
        uwd, uninfected_type, variable = Symbol(
            string(stratum) *
                "_U"
        )
    )
    infected_stratum_junction = add_junction!(
        uwd, infected_type, variable = Symbol(
            string(stratum) *
                "_I"
        )
    )
    return (uninfected_stratum_junction, infected_stratum_junction)
end

"""
Generate junction variable names for multi-stage compartments.

When `number_of_stages == 1`, return the conventional base name (`:E` or `:I`).
For multiple stages, append the stage number to produce stable, readable names such as
`:E1`, `:E2`, and `:I1`. These names aid visualization, inspection, and downstream lookup;
progression mechanisms themselves are connected using junction IDs.

# Examples
```julia
variable_name(:E, 1, 1)  # Returns :E (single stage)
variable_name(:E, 2, 3)  # Returns :E2 (stage 2 of 3)
variable_name(:I, 1, 4)  # Returns :I1 (stage 1 of 4)
```
"""
function variable_name(var_name, stage, number_of_stages)
    if number_of_stages == 1
        return var_name
    else
        return Symbol("$(var_name)$(stage)")
    end
end

"""
Create a sequence of compartment stages with progression transitions between them.

This is the workhorse for multi-stage compartments, enabling Erlang-distributed dwell times.
It creates N junctions (for example, E1→E2→E3 or I1→I2→I3) and chains them with
progression mechanisms.

# Construction behavior
- Returns the vector of ALL junction IDs (not just first/last) for flexible mechanism attachment
- First stage receives incoming transitions (e.g., S+I→E1), last stage feeds next compartment

# Arguments
- `uwd`: The undirected wiring diagram to modify (follows convention of other mutation functions)
- `variable_symbol`: Base compartment name (:E or :I)
- `number_of_stages`: Number of sequential stages to create
- `pop_type`: The population type symbol for the junctions
- `typing`: The epidemiological typing for mechanism dispatch
"""
function add_stages!(
        uwd::RelationDiagram, variable_symbol::Symbol, number_of_stages::Int,
        pop_type::Symbol, typing::EpidemiologicalTyping
    )
    number_of_stages >= 1 || throw(ArgumentError("number_of_stages must be at least 1"))

    # Create junctions for each stage
    junctions = [
        add_junction!(
                uwd,
                pop_type,
                variable = variable_name(variable_symbol, stage, number_of_stages)
            )
            for stage in 1:number_of_stages
    ]
    # Add progression between stages
    for stage in 1:(number_of_stages - 1)
        add_disease_progression!(uwd, junctions[stage], junctions[stage + 1], typing)  # e.g. E_stage → E_(stage+1) or I_stage → I_(stage+1)
    end
    return junctions
end

"""
Populate an SI model UWD with junctions and infection mechanisms.

# Construction Role
This is the base layer for all direct-infection models (SI, SIR, SIS). It establishes:
1. S compartment (via `set_S_junction!`)
2. I compartment chain (potentially multi-stage via `add_stages!`)
3. Infection dynamics: every I stage can infect S, but new infections enter I1 only

Returns (uwd, S_junction, last_I_junction) to enable extensions:
- SIR adds last_I_junction → R
- SIS adds last_I_junction → S_junction (reversion)

The multi-stage capability means SI can represent gamma-distributed infectious periods
even before extending to SIR.
"""
function setup_basic!(uwd::RelationDiagram, typing::EpidemiologicalTyping, model::SI)
    # Add junctions for S compartment
    S_junction = set_S_junction!(uwd, typing)
    pop_type = get_infected_type(typing)

    I_junctions = add_stages!(uwd, :I, model.number_I_stages, pop_type, typing)

    # Add infection to first I stage
    for I_junction in I_junctions
        add_infection!(uwd, S_junction, I_junction, I_junctions[1], typing)  # S + I → I + I
    end

    return (uwd, S_junction, I_junctions[end])
end

"""
Populate an SEI model UWD with junctions, exposure, and progression mechanisms.

# Construction Role
This is the base layer for all exposure-based models (SEI, SEIR, SEIS, SEIRS). It establishes:
1. S compartment (via `set_S_junction!`)
2. E compartment chain (potentially multi-stage for gamma-distributed latent period)
3. I compartment chain (potentially multi-stage for gamma-distributed infectious period)
4. Exposure dynamics: every I stage can expose S, but new exposures enter E1 only
5. Progression: E_last → I_1 connects the latent and infectious stages

Returns (uwd, S_junction, last_I_junction) to enable extensions:
- SEIR adds last_I_junction → R
- SEIS adds last_I_junction → S_junction (reversion)
- SEIRS adds both (recovery then waning)

The dual multi-stage capability enables independent control of latent and infectious period
distributions - critical for realistic disease modeling.
"""
function setup_basic!(uwd::RelationDiagram, typing::EpidemiologicalTyping, model::SEI)
    # Add junctions for S compartment
    S_junction = set_S_junction!(uwd, typing)
    pop_type = get_infected_type(typing)

    # For multiple E and I stages, create junctions for each stage
    E_junctions = add_stages!(uwd, :E, model.number_E_stages, pop_type, typing)
    I_junctions = add_stages!(uwd, :I, model.number_I_stages, pop_type, typing)

    # add infection to first E stage from any infection stage
    for I_junction in I_junctions
        add_infection!(uwd, S_junction, I_junction, E_junctions[1], typing)  # S + I → E + I
    end

    # Add progression from last E stage to first I stage
    add_disease_progression!(uwd, E_junctions[end], I_junctions[1], typing)  # E_last → I_1

    return (uwd, S_junction, I_junctions[end])
end
