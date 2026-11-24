# Helper functions for epidemiological mechanisms added to undirected wiring diagram (UWD)
# This file give primitives for programmatic construction of a UWDs

"""
    add_strata_transition!(uwd, from_junction, to_junction, typing)

Add a movement box between two junctions of the same population type, typed by that
population's movement type (`:strata`, or `:strata_U` / `:strata_I` under the
uninfected/infected split). Movement changes an individual's stratum and keeps its disease
state; in a `typed_product` it pairs with the reflexive movement boxes of the other factors
(see `include_reflexives` on [`create_model`](@ref)). The mass-action rate is per capita.
Returns the new box.
"""
function add_strata_transition!(
        uwd, from_junction::Integer, to_junction::Integer, typing::EpidemiologicalTyping
    )
    population = uwd[from_junction, :junction_type]
    uwd[to_junction, :junction_type] == population ||
        throw(ArgumentError("a strata transition must preserve population type"))
    box = add_box!(uwd, [population, population]; name = population_box(Movement(), typing, population))
    from_port, to_port = ports(uwd, box)
    set_junction!(uwd, from_port, from_junction)
    set_junction!(uwd, to_port, to_junction)
    return box
end

"""
    add_birth!(uwd, junction, typing)

Add a `:birth` box, an entry into `junction` with no source, so its mass-action rate is
zero-order (births per unit time). The junction must have the uninfected population type.
Returns the new box.
"""
function add_birth!(uwd, junction::Integer, typing::EpidemiologicalTyping)
    population = uwd[junction, :junction_type]
    population == get_uninfected_type(typing) || throw(
        ArgumentError(
            "births enter the $(get_uninfected_type(typing)) population; " *
                "`$(uwd[junction, :variable])` has type $population"
        )
    )
    box = add_box!(uwd, [population]; name = :birth)
    set_junction!(uwd, only(ports(uwd, box)), junction)
    return box
end

"""
    add_death!(uwd, junction, typing)

Add a death box, an exit from `junction` with no target, typed `:death` (or `:death_U` /
`:death_I` under the uninfected/infected split). The mass-action rate is per capita. Returns
the new box.
"""
function add_death!(uwd, junction::Integer, typing::EpidemiologicalTyping)
    population = uwd[junction, :junction_type]
    box = add_box!(uwd, [population]; name = population_box(Death(), typing, population))
    set_junction!(uwd, only(ports(uwd, box)), junction)
    return box
end

"""
    add_importation!(uwd, from_junction, to_junction, typing)

Add an `:importation` box: `from_junction` (uninfected) becomes `to_junction` (infected) with
no infector, so the mass-action rate is per capita. Returns the new box.
"""
function add_importation!(
        uwd, from_junction::Integer, to_junction::Integer, typing::EpidemiologicalTyping
    )
    U, I = get_uninfected_type(typing), get_infected_type(typing)
    uwd[from_junction, :junction_type] == U && uwd[to_junction, :junction_type] == I ||
        throw(ArgumentError("importation goes from the $U population to the $I population"))
    box = add_box!(uwd, [U, I]; name = :importation)
    from_port, to_port = ports(uwd, box)
    set_junction!(uwd, from_port, from_junction)
    set_junction!(uwd, to_port, to_junction)
    return box
end

# Births enter a name's first junction (a stratum's uninfected one); deaths leave all of them.
function _add_demography!(uwd, typing, demography, junctions_of)
    for name in demography.births
        add_birth!(uwd, first(junctions_of[name]), typing)
    end
    for name in demography.deaths, junction in junctions_of[name]
        add_death!(uwd, junction, typing)
    end
    return uwd
end

"""
Add infection mechanism: Infectee + Infector → first_infected + Infector

General infection mechanism where an infectee (typically S) interacts with an infector (typically I)
to produce a newly infected individual that enters the first_infected compartment, while the infector
remains unchanged. The `first_infected_junction` can be:
- An I stage for SIR-like models (direct infection)
- An E stage for SEIR-like models (exposure before infectiousness)

The `typing` argument determines the population types used.
"""
function add_infection!(
        uwd, infectee_junction, infector_junction,
        first_infected_junction, typing::OnePopulationTyping
    )
    pop_type = typing.population_type
    infect_box = add_box!(
        uwd, [pop_type, pop_type, pop_type, pop_type], name = :transmission
    )
    ports_infect = ports(uwd, infect_box)
    set_junction!(uwd, ports_infect[1], infectee_junction)  # Infectee input
    set_junction!(uwd, ports_infect[2], infector_junction)  # Infector input
    set_junction!(uwd, ports_infect[3], first_infected_junction)  # First infected output
    return set_junction!(uwd, ports_infect[4], infector_junction)  # Infector output (unchanged)
end

function add_infection!(
        uwd, infectee_junction, infector_junction,
        first_infected_junction, typing::UninfectedInfectedTyping
    )
    uninfected_type = typing.uninfected_type
    infected_type = typing.infected_type
    infect_box = add_box!(
        uwd, [uninfected_type, infected_type, infected_type, infected_type],
        name = :transmission
    )
    ports_infect = ports(uwd, infect_box)
    set_junction!(uwd, ports_infect[1], infectee_junction)  # Infectee input
    set_junction!(uwd, ports_infect[2], infector_junction)  # Infector input
    set_junction!(uwd, ports_infect[3], first_infected_junction)  # First infected output
    return set_junction!(uwd, ports_infect[4], infector_junction)  # Infector output (unchanged)
end

"""
Add disease progression mechanism: from_compartment → to_compartment

Used for transitions like E→I, I→R, etc. (infected → infected transitions)
"""
function add_disease_progression!(
        uwd, from_junction, to_junction, typing::OnePopulationTyping
    )
    pop_type = typing.population_type
    disease_box = add_box!(uwd, [pop_type, pop_type], name = :disease)
    ports_disease = ports(uwd, disease_box)
    set_junction!(uwd, ports_disease[1], from_junction)  # input
    return set_junction!(uwd, ports_disease[2], to_junction)    # output
end

function add_disease_progression!(
        uwd, from_junction, to_junction, typing::UninfectedInfectedTyping
    )
    pop_type = typing.infected_type
    density_box = add_box!(uwd, [pop_type, pop_type], name = :disease)
    ports_density = ports(uwd, density_box)
    set_junction!(uwd, ports_density[1], from_junction)  # input
    return set_junction!(uwd, ports_density[2], to_junction)    # output
end

"""
Add uninfected density progression mechanisms. This is useful for transitions like waning partial
immunity (uninfected → uninfected transitions, e.g., V → S).
"""
function add_uninfected_density_progression!(
        uwd, from_junction, to_junction, typing::OnePopulationTyping
    )
    pop_type = typing.population_type
    waning_box = add_box!(uwd, [pop_type, pop_type], name = :waning)
    ports_waning = ports(uwd, waning_box)
    set_junction!(uwd, ports_waning[1], from_junction)  # input
    return set_junction!(uwd, ports_waning[2], to_junction)    # output
end

function add_uninfected_density_progression!(
        uwd, from_junction, to_junction, typing::UninfectedInfectedTyping
    )
    pop_type = typing.uninfected_type
    density_box = add_box!(uwd, [pop_type, pop_type], name = :waning)
    ports_density = ports(uwd, density_box)
    set_junction!(uwd, ports_density[1], from_junction)  # input
    return set_junction!(uwd, ports_density[2], to_junction)    # output
end

"""
Add reversion progression mechanism: e.g. R → S (infected → uninfected transitions)

Used for transitions from infected compartments back to uninfected compartments.
"""
function add_reversion_progression!(
        uwd, from_junction, to_junction, typing::OnePopulationTyping
    )
    pop_type = typing.population_type
    reversion_box = add_box!(uwd, [pop_type, pop_type], name = :reversion)
    ports_reversion = ports(uwd, reversion_box)
    set_junction!(uwd, ports_reversion[1], from_junction)  # input
    return set_junction!(uwd, ports_reversion[2], to_junction)    # output
end

function add_reversion_progression!(
        uwd, from_junction, to_junction, typing::UninfectedInfectedTyping
    )
    infected_type = typing.infected_type
    uninfected_type = typing.uninfected_type
    reversion_box = add_box!(uwd, [infected_type, uninfected_type], name = :reversion)
    ports_reversion = ports(uwd, reversion_box)
    set_junction!(uwd, ports_reversion[1], from_junction)  # input
    return set_junction!(uwd, ports_reversion[2], to_junction)    # output
end
