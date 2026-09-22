# Vector fields: the Petri-net right-hand side with dynamic rates, its out-of-place wrapper for
# SeeToDee, and the augmented one-step dynamics every filter shares.

# Names over the existing storage without copying (`LVector(NamedTuple(...))` is dynamic past a
# few dozen elements).
@inline function _named_lvector(::Val{names}, ::Val{N}, x) where {names, N}
    length(x) == N || throw(DimensionMismatch("expected a state of length $N, got $(length(x))"))
    return LArray{names}(x)
end
@inline _zero_lvector(::Val{names}, ::Val{N}, ::Type{T}) where {names, N, T} = LArray{names}(zeros(T, N))

"""
    build_petri_vf(pn, rates; defaults = (;)) -> petri_vf!(du, u, (hyperparams, latent), t)

In-place mass-action vector field of `pn` whose transition rates are
`merge(defaults, rates(latent, hyperparams, t))`: `defaults` holds the fixed rates and the rate
function returns only the dynamic ones, keyed by flattened transition name.
"""
function build_petri_vf(pn, rates; defaults::NamedTuple = (;))
    vf! = vectorfield_flat(pn)
    transitions = Set(flatten_symbols(n) for n in AlgebraicPetri.tnames(pn))
    for k in keys(defaults)
        k in transitions || throw(
            ArgumentError(
                "rate default `$k` is not a transition of the Petri net; transitions are " *
                    "$(sort!(collect(transitions)))",
            )
        )
    end
    function petri_vf!(du, u, p, t)
        hyperparams, latent = p
        return vf!(du, u, merge(defaults, rates(latent, hyperparams, t)), t)
    end
    return petri_vf!
end

"""
    build_unified_vf(petri_vf!, layout) -> (x, u, p, t) -> dx

Out-of-place form of the Petri vector field for `SeeToDee.Rk4`, naming the ODE slots of `x`.
"""
function build_unified_vf(petri_vf!, layout::StateLayout{N, M}) where {N, M}
    names_val, n_val = Val(ode_names(layout)), Val(N + M)
    @inline function unified_vf(x_flat, u, p, t)
        du = _zero_lvector(names_val, n_val, eltype(x_flat))
        petri_vf!(du, _named_lvector(names_val, n_val, x_flat), p, t)
        return du
    end
    return unified_vf
end

"""
    make_lvector_constructor(names) -> x -> LArray{names}(collect(x))
"""
function make_lvector_constructor(names::NTuple{N, Symbol}) where {N}
    names_val, n_val = Val(names), Val(N)
    @inline to_lvector(x) = _named_lvector(names_val, n_val, collect(x))
    return to_lvector
end

"""
    build_R1(layout) -> Diagonal

Identity process-noise covariance sized `n_latent + n_accumulators`; every noise magnitude is
applied inside [`build_full_dynamics`](@ref).
"""
build_R1(layout::StateLayout) = Diagonal(ones(n_latent(layout) + length(layout.accumulator_indices)))

"""
    build_full_dynamics(petri_vf!, stochastic, layout; dt = 1.0, supersample = 2, obs_jitter = 1.0)
        -> dynamics(x, u, p, t, w[, rng])

One filter step of the augmented state: apply the stochastic driver (coefficient noise from
`w[1:L]`, jumps on `rng`), zero the reset accumulators, integrate the flow over `dt` with
`supersample` RK4 substeps, then add the accumulator whisker `obs_jitter * w[L+1:end]`. `w` has
`size(build_R1(layout), 1)` entries.
"""
function build_full_dynamics(
        petri_vf!, stochastic::StochasticUpdate{L}, layout::StateLayout{N, M, L, S};
        dt::Real = 1.0, supersample::Integer = 2, obs_jitter::Real = 1.0,
    ) where {N, M, L, S}
    dt, jitter = float(dt), float(obs_jitter)
    step = Rk4(build_unified_vf(petri_vf!, layout), dt; supersample = Int(supersample))
    ode_range = 1:(N + M)
    accumulators = layout.accumulator_indices
    @inline function dynamics(x, u, p, t, w, rng = nothing)
        T = eltype(x)
        length(w) == L + S || throw(
            DimensionMismatch(
                "process-noise vector has length $(length(w)); this model needs $(L + S) = " *
                    "$L latent + $S reset-accumulator terms (accumulators, not observation signals; " *
                    "see build_R1)",
            )
        )
        xa = stochastic.advance(x, p, w, rng, t, dt)
        latent = stochastic.extract(xa)
        x_ode = max.(view(xa, ode_range), T(1.0e-6))
        @inbounds for i in accumulators
            x_ode[i] = zero(T)
        end
        x_next = max.(step(x_ode, u, (p, latent), t), T(1.0e-6))
        @inbounds for (s, i) in enumerate(accumulators)
            x_next[i] += T(jitter) * w[L + s]
        end
        @inbounds for (k, i) in enumerate(ode_range)
            xa[i] = x_next[k]
        end
        return xa
    end
    return dynamics
end
