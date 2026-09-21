# ============================================================================
# STATE LAYOUT - Connecting named states to indices in model state vector
# ============================================================================

"""
    StateLayout{N,M,K,L,S}

Encapsulates index layout for model state vector, enabling zero-allocation
named access via @SLVector.

The state vector is organized as:
[core_states..., obs_states..., latent_states...]

# Type Parameters
- `N`: Number of core compartment states
- `M`: Number of observation delay states
- `K`: Reserved legacy parameter (always `0` after previous-cumulative-u refactor)
- `L`: Number of latent process states
- `S`: Number of observation signals

# Fields
- `core_names`: Names of core compartments (S, E, I, R, etc.)
- `obs_names`: Names of observation delay states (O_I_1, O_I_2, etc.)
- `latent_names`: Names of latent process states (:Rt_unc, etc.)
- `signal_names`: Names of observation signals (:hosp, :deaths, etc.)
- `core_range`, `obs_range`, `latent_range`: Index ranges
- `accumulator_indices`: Absolute indices in the full state vector of the
  reset-accumulator observation states (one per signal)
"""
struct StateLayout{N, M, K, L, S}
    core_names::NTuple{N, Symbol}
    obs_names::NTuple{M, Symbol}
    latent_names::NTuple{L, Symbol}
    signal_names::NTuple{S, Symbol}

    core_range::UnitRange{Int}
    obs_range::UnitRange{Int}
    latent_range::UnitRange{Int}
    accumulator_indices::NTuple{S, Int}
    total_dim::Int
end

"""
    StateLayout(core_names, obs_names, latent_names; signal_names=(:y,), accumulator_indices=nothing)

Construct a StateLayout from name tuples.

# Arguments
- `core_names`: Tuple of core compartment symbols
- `obs_names`: Tuple of observation delay compartment symbols
- `latent_names`: Tuple of latent process symbols
- `signal_names`: Tuple of signal names (default: (:y,) for single signal)
- `accumulator_indices`: Tuple of indices into obs_names for the reset-accumulator
  compartments. If nothing, defaults to evenly spaced terminal obs states when `M` is
  divisible by `S`, otherwise reuses the final obs state for each signal. `obs_names`
  must be non-empty when `signal_names` is non-empty.
"""
function StateLayout(
        core_names::NTuple{N, Symbol},
        obs_names::NTuple{M, Symbol},
        latent_names::NTuple{L, Symbol};
        signal_names::NTuple{S, Symbol} = (:y,),
        accumulator_indices::Union{Nothing, NTuple{S, Int}} = nothing
    ) where {N, M, L, S}
    core_start = 1
    core_end = N
    obs_start = core_end + 1
    obs_end = core_end + M

    if S == 0
        M == 0 || throw(
            ArgumentError("obs_names requires at least one signal; pass signal_names=() only when obs_names is empty")
        )
        accumulator_indices = ()
    elseif accumulator_indices === nothing
        if M == 0
            throw(
                ArgumentError(
                    "obs_names must be non-empty when signal_names is non-empty; pass signal_names=() when there are no observation signals"
                )
            )
        elseif M % S == 0
            obs_per_signal = M ÷ S
            accumulator_indices = ntuple(Val(S)) do s
                core_end + s * obs_per_signal
            end
        else
            final_obs_index = core_end + M
            accumulator_indices = ntuple(_ -> final_obs_index, Val(S))
        end
    else
        all(1 <= idx <= M for idx in accumulator_indices) || throw(
            ArgumentError("accumulator_indices must refer to obs_names positions")
        )
        accumulator_indices = ntuple(Val(S)) do s
            core_end + accumulator_indices[s]
        end
    end

    K = 0
    latent_start = obs_end + 1
    latent_end = obs_end + L
    return StateLayout{N, M, K, L, S}(
        core_names,
        obs_names,
        latent_names,
        signal_names,
        core_start:core_end,
        obs_start:obs_end,
        latent_start:latent_end,
        accumulator_indices,
        latent_end
    )
end

"""
    all_names(layout::StateLayout)

Return all state names as a single tuple.
"""
function all_names(layout::StateLayout)
    return (layout.core_names..., layout.obs_names..., layout.latent_names...)
end

# ============================================================================
# Petri Net Constructor
# ============================================================================

_default_signal_names(::Val{0}) = ()
_default_signal_names(::Val{1}) = (:y,)
_default_signal_names(::Val{S}) where {S} = ntuple(i -> Symbol(:y, i), Val(S))

function _resolve_signal_names(signal_names, n_signals::Int)
    if isnothing(signal_names)
        return _default_signal_names(Val(n_signals))
    end

    resolved_signal_names = Tuple(signal_names)
    length(resolved_signal_names) == n_signals || throw(
        ArgumentError(
            "signal_names length ($(length(resolved_signal_names))) must match number of observation chains ($n_signals)"
        )
    )
    return resolved_signal_names
end

function _relative_cumulative_indices(obs_names, cumulative_names)
    index_lookup = Dict(name => i for (i, name) in pairs(obs_names))
    return Tuple(index_lookup[name] for name in cumulative_names)
end

"""
    StateLayout(petri_net::LabelledPetriNet, driver_specs; signal_names=nothing)

Construct a StateLayout directly from a Petri net and the stochastic-driver specs.

Observation states and cumulative outputs are inferred from the augmented Petri
net via `AlgebraicEpiMech.observation_layout`. Species names are
flattened with `flatten_symbols`, so composed/stratified models work without
special-case layout code in ConfigurableEpi.

`driver_specs` is the full list of stochastic drivers (see [`build_stochastic_update`](@ref)). Only
drivers that **carry state** claim slots: the Gaussian coefficient processes (AR1/RW) get the
`latent_range`. An [`ArrivalProcess`](@ref) is *stateless* — its jump is absorbed into the
compartments, so it claims no slots and the layout is unchanged by its presence. (The layout's job is
"who must carry state to be Markov, and where" — it is deliberately agnostic to *how* a process
evolves; diffusion-vs-jump is the driver's business.)

# Keyword Arguments
- `signal_names`: Optional tuple of user-facing signal names. If omitted, defaults
  to `(:y,)` for a single observation chain, `(:y1, :y2, ...)` for multiple chains,
  and `()` when the Petri net has no observation chains.
"""
function StateLayout(
        petri_net,
        driver_specs::Tuple;
        signal_names = nothing
    )
    obs_info = observation_layout(petri_net)
    obs_names = obs_info.obs_names
    flat_species_names = Tuple(flatten_symbols(name) for name in snames(petri_net))
    obs_name_set = Set(obs_names)
    core_names = Tuple(name for name in flat_species_names if name ∉ obs_name_set)

    # Only state-carrying drivers claim slots; stateless jump drivers (ArrivalProcess) claim none.
    latent_names = Tuple(spec.name for spec in driver_specs if carries_state(spec))

    resolved_signal_names = _resolve_signal_names(signal_names, length(obs_info.chains))
    relative_cumulative_indices = _relative_cumulative_indices(
        obs_names,
        obs_info.cumulative_names
    )
    return StateLayout(
        core_names,
        obs_names,
        latent_names;
        signal_names = resolved_signal_names,
        accumulator_indices = relative_cumulative_indices
    )
end

"""
    ode_names(layout::StateLayout)

Return names for ODE states (core + obs).
"""
function ode_names(layout::StateLayout)
    return (layout.core_names..., layout.obs_names...)
end

"""
    n_ode_states(layout::StateLayout)

Number of ODE states (core + obs).
"""
n_ode_states(layout::StateLayout{N, M}) where {N, M} = N + M

"""
    n_latent(layout::StateLayout{N,M,K,L,S})

Number of latent process states.
"""
n_latent(::StateLayout{N, M, K, L, S}) where {N, M, K, L, S} = L

"""
    n_signals(layout::StateLayout{N,M,K,L,S})

Number of observation signals.
"""
n_signals(::StateLayout{N, M, K, L, S}) where {N, M, K, L, S} = S

"""
    extract_latent(state, layout)

Extract latent states from state vector as a NamedTuple (UNCONSTRAINED values).

Uses ntuple with Val(L) to ensure compile-time unrolling and
stack allocation - no heap allocations in hot loops.

Note: Returns raw unconstrained values (e.g., log(Rt), logit(rho)).
For constrained values, use `StochasticUpdate.extract(state)`.
"""
function extract_latent(
        state::AbstractVector,
        layout::StateLayout{N, M, K, L, S}
    ) where {N, M, K, L, S}
    start_idx = N + M + K
    vals = ntuple(i -> state[start_idx + i], Val(L))
    return NamedTuple{layout.latent_names}(vals)
end
