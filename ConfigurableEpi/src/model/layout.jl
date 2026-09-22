# Index layout of the filter state vector.

"""
    StateLayout{N, M, L, S}

Layout of the state vector `[core compartments (N); observation states (M); latent
coefficients (L)]` observed through `S` signals. `accumulator_indices` are the absolute slots of
the reset accumulators, one per signal.

    StateLayout(core_names, obs_names, latent_names; signal_names = (:y,), accumulator_indices = nothing)
    StateLayout(petri_net, driver_specs; signal_names = nothing)

Without `accumulator_indices` the observation states are split evenly over the signals and each
signal's last state is its accumulator. The second form reads the observation chains of a net
built with `AlgebraicEpiMech.attach_observation`; only drivers with [`carries_state`](@ref) claim
latent slots.
"""
struct StateLayout{N, M, L, S}
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

function StateLayout(
        core_names::NTuple{N, Symbol}, obs_names::NTuple{M, Symbol}, latent_names::NTuple{L, Symbol};
        signal_names::NTuple{S, Symbol} = (:y,), accumulator_indices = nothing,
    ) where {N, M, L, S}
    (S == 0) == (M == 0) || throw(
        ArgumentError(
            "observation states and signals go together; pass signal_names = () for a net " *
                "without observation chains",
        )
    )
    accumulators = if S == 0
        ()
    elseif accumulator_indices === nothing
        M % S == 0 || throw(
            ArgumentError("$M observation states do not split evenly over $S signals; pass accumulator_indices")
        )
        ntuple(s -> N + s * (M ÷ S), Val(S))
    else
        all(i -> 1 <= i <= M, accumulator_indices) ||
            throw(ArgumentError("accumulator_indices must index obs_names"))
        ntuple(s -> N + accumulator_indices[s], Val(S))
    end
    return StateLayout{N, M, L, S}(
        core_names, obs_names, latent_names, signal_names,
        1:N, (N + 1):(N + M), (N + M + 1):(N + M + L), accumulators, N + M + L,
    )
end

function StateLayout(petri_net, driver_specs::Tuple; signal_names = nothing)
    obs = observation_layout(petri_net)
    obs_set = Set(obs.obs_names)
    core_names = Tuple(n for n in flatten_symbols.(snames(petri_net)) if n ∉ obs_set)
    latent_names = Tuple(spec.name for spec in driver_specs if carries_state(spec))
    n_chains = length(obs.chains)
    signals = signal_names === nothing ? _default_signal_names(n_chains) : Tuple(signal_names)
    length(signals) == n_chains ||
        throw(ArgumentError("$(length(signals)) signal names for $n_chains observation chains"))
    position = Dict(name => i for (i, name) in enumerate(obs.obs_names))
    return StateLayout(
        core_names, obs.obs_names, latent_names;
        signal_names = signals,
        accumulator_indices = Tuple(position[name] for name in obs.cumulative_names),
    )
end

_default_signal_names(n) = n == 0 ? () : n == 1 ? (:y,) : ntuple(i -> Symbol(:y, i), n)

all_names(l::StateLayout) = (l.core_names..., l.obs_names..., l.latent_names...)
ode_names(l::StateLayout) = (l.core_names..., l.obs_names...)
n_ode_states(::StateLayout{N, M}) where {N, M} = N + M
n_latent(::StateLayout{N, M, L}) where {N, M, L} = L
n_signals(::StateLayout{N, M, L, S}) where {N, M, L, S} = S

@inline _named_tuple(::Val{names}, values::Tuple) where {names} = NamedTuple{names}(values)

"""
    extract_latent(state, layout) -> NamedTuple

The latent slots of `state` as a NamedTuple of unconstrained values.
"""
extract_latent(state::AbstractVector, l::StateLayout{N, M, L}) where {N, M, L} =
    NamedTuple{l.latent_names}(ntuple(i -> state[N + M + i], Val(L)))
