# ============================================================================
# CORE DYNAMICS - UNIFIED PETRI NET VECTORFIELD FACTORY
# ============================================================================

# Names over the EXISTING storage — no copy. The ODE right-hand side wraps the full state this way
# on every evaluation, and the previous form, `LVector(NamedTuple{names}(NTuple{N}(x)))`, goes
# through `Base._totuple`, which is dynamic past a few dozen elements: at the 50-state geographic
# model (300 species) it cost 1.3 ms per call, thirty times the vectorfield itself.
@inline function _named_lvector(
        ::Val{names},
        ::Val{N},
        x
    ) where {names, N}
    length(x) == N || throw(
        DimensionMismatch("expected a state of length $N, got $(length(x))")
    )
    return LArray{names}(x)
end

@inline function _zero_named_lvector(
        ::Val{names},
        ::Val{N},
        ::Type{Ty}
    ) where {names, N, Ty}
    return LArray{names}(zeros(Ty, N))
end

"""
    build_petri_vf(pn, rates; defaults = (;))

Build an in-place Petri net vectorfield from a rate function plus optional fixed defaults.

`rates(latent, hyperparams, t) -> NamedTuple` returns the *dynamic* transition rates,
keyed by transition name. It reads constrained latent values (e.g. `latent.Rt`),
hyperparameters (e.g. `hyperparams.beta`), and time `t`.

`defaults` is a NamedTuple of *fixed* transition rates — typically loaded from a
TOML/YAML spec — so the rate function only has to express the transitions that are
actually dynamic, keeping it intentional rather than a wall of constants. At each step
the two are combined with `merge(defaults, rates(...))`, so **anything the rate function
returns overrides the matching default**. The union of `defaults`' keys and the rate
function's keys must cover every transition of `pn` (extra/unused keys are ignored).

The returned `petri_vf!(du, u, (hyperparams, latent), t)` is consumed by
`build_unified_vf` / `build_full_dynamics`, which thread `hyperparams` and the
constrained `latent` NamedTuple through as the `p` argument.

# Example
```julia
pn = dom(create_model(OnePopulationSchema(), SEIRS()))

# Fixed structural rates from a spec; only transmission is dynamic.
defaults = (E_to_I = 1.0/3.0, I_to_R = 1.0/7.0, R_to_S = 1.0/180.0)
rates(latent, hyper, t) = (transmission_S_I = hyper.beta * latent.Rt,)
petri_vf! = build_petri_vf(pn, rates; defaults = defaults)
```
"""
function build_petri_vf(pn, rates; defaults::NamedTuple = (;))
    petri_vf_raw! = vectorfield_flat(pn)
    if !isempty(defaults)
        transition_names = Set(flatten_symbols(n) for n in AlgebraicPetri.tnames(pn))
        for k in keys(defaults)
            k in transition_names || throw(
                ArgumentError(
                    "rate default `$k` is not a transition of the Petri net; " *
                        "transitions are $(sort!(collect(transition_names)))"
                )
            )
        end
    end
    function petri_vf!(du, u, p, t)
        hyperparams, latent = p
        return petri_vf_raw!(du, u, merge(defaults, rates(latent, hyperparams, t)), t)
    end
    return petri_vf!
end

"""
    build_unified_vf(petri_vf!, layout::StateLayout{N, M, K, L}) where {N, M, K, L}

Build a vectorfield for a Petri net that already includes observation dynamics.

When observation delay states are part of the Petri net (attached by
`AlgebraicEpiMech.attach_observation`), `vectorfield_flat(pn)` handles all ODE dynamics in one
pass. This function wraps that in-place vectorfield into the SeeToDee-compatible
`(x, u, p, t) -> dx` form.

# Arguments
- `petri_vf!`: In-place vectorfield from `build_petri_vf(augmented_pn, rates)`
  that covers both core and observation dynamics
- `layout`: StateLayout defining state vector structure

# Returns
Closure `(x, u, p, t) -> dx` compatible with SeeToDee.jl discretization.

# Example
```julia
# Build the model, then attach observation to the infection event
pn = attach_observation(
    dom(create_model(OnePopulationSchema(), SEIR())),
    AtEvent(:transmission); n_stages = 2,
)

# Rate function covers core + observation-chain transitions
rates(latent, hyper, t) = (
    transmission_S_I = hyper.beta * latent.Rt,
    obs_inflow = 1.0 / 3.0,
    obs_delay = 1.0 / 2.0,
)
petri_vf! = build_petri_vf(pn, rates)

# Build unified vectorfield (no separate obs_vf! needed)
vf = build_unified_vf(petri_vf!, layout)
ode_stepper = Rk4(vf, dt; supersample=2)
```
"""
function build_unified_vf(
        petri_vf!,
        layout::StateLayout{N, M, K, L}
    ) where {N, M, K, L}

    # Capture constants at build time
    ode_state_names = ode_names(layout)
    ode_state_names_val = Val(ode_state_names)
    n_ode = N + M
    n_ode_val = Val(n_ode)

    # SeeToDee-compatible out-of-place vectorfield
    @inline function unified_vf(x_flat, u, p, t)
        # Wrap flat state in LVector for named access
        x = _named_lvector(ode_state_names_val, n_ode_val, x_flat)

        # Initialize du with zeros (same structure)
        Ty = eltype(x_flat)
        du = _zero_named_lvector(ode_state_names_val, n_ode_val, Ty)

        # Apply all dynamics in one pass (core + obs)
        petri_vf!(du, x, p, t)

        return du
    end

    return unified_vf
end

# ============================================================================
# HELPER: LVector constructor factory
# ============================================================================

"""
    make_lvector_constructor(names::NTuple{N,Symbol}) where N

Create an LVector constructor for the given names.

Returns a closure that wraps a flat vector in an LVector with named access.
Useful for creating state/derivative vectors from flat arrays.

# Example
```julia
to_lvec = make_lvector_constructor((:S, :I, :R))
x = to_lvec([990.0, 10.0, 0.0])
x.S  # 990.0
x[:I]  # 10.0
```
"""
function make_lvector_constructor(names::NTuple{N, Symbol}) where {N}
    names_val = Val(names)
    n_val = Val(N)

    # Copies, so the result owns its storage (the internal wrapper aliases its argument).
    @inline function to_lvector(x)
        return _named_lvector(names_val, n_val, collect(x))
    end
    return to_lvector
end

# Alias for backward compatibility
const make_slvector_constructor = make_lvector_constructor
