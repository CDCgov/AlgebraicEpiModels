# Helper functions for epidemiological mechanisms added to undirected wiring diagram (UWD)
# This file give primitives for programmatic construction of a UWDs

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
