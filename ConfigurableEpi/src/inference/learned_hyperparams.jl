# Hyperparameters learned online by the particle filter: appended to each particle in
# unconstrained space, read back constrained, and refreshed between correct! and predict! by the
# Liu-West shrink-jitter kernel.

"""
    LearnedHyperparams{H}

A block of `H` hyperparameters carried per particle in the tail after the model state
(`offset == layout.total_dim`). `extract(x)` reads them constrained as a NamedTuple;
`to_unconstrained(nt)` maps a constrained NamedTuple to the tail's `SVector{H}`.
"""
struct LearnedHyperparams{H, Names, Priors, E, U}
    names::Names
    priors::Priors
    offset::Int
    extract::E
    to_unconstrained::U
end

n_learned(::LearnedHyperparams{H}) where {H} = H
hyper_range(b::LearnedHyperparams{H}) where {H} = (b.offset + 1):(b.offset + H)
hyper_slots(b::LearnedHyperparams{H}) where {H} = ntuple(i -> b.offset + i, Val(H))

"""
    build_learned_hyperparams(priors, layout) -> LearnedHyperparams

`priors` is a NamedTuple (keys equal to the prior names), a tuple of priors, or one prior.
"""
function build_learned_hyperparams(priors::NamedTuple, layout::StateLayout)
    isempty(priors) && throw(ArgumentError("at least one prior is required"))
    for (key, prior) in pairs(priors)
        prior isa ParameterDistribution || throw(ArgumentError("prior $key is not a ParameterDistribution"))
        prior_name(prior) == key || throw(ArgumentError("prior key $key does not match its name $(prior_name(prior))"))
    end
    return _build_learned_hyperparams(Val(length(priors)), priors, layout.total_dim)
end

function build_learned_hyperparams(priors::Tuple, layout::StateLayout)
    names = map(prior_name, priors)
    allunique(names) || throw(ArgumentError("learned prior names must be unique, got $names"))
    return build_learned_hyperparams(NamedTuple{names}(priors), layout)
end
build_learned_hyperparams(prior::ParameterDistribution, layout::StateLayout) =
    build_learned_hyperparams((prior,), layout)

function _build_learned_hyperparams(::Val{H}, priors::NamedTuple, offset::Int) where {H}
    names = keys(priors)
    constraints = map(ScalarConstraint, values(priors))
    names_val = Val(names)
    @inline extract(x::AbstractVector) =
        _named_tuple(names_val, ntuple(i -> _to_constrained(constraints[i], x[offset + i]), Val(H)))
    @inline to_unconstrained(nt::NamedTuple) =
        SVector{H}(ntuple(i -> _to_unconstrained(constraints[i], nt[names[i]]), Val(H)))
    return LearnedHyperparams{H, typeof(names), typeof(values(priors)), typeof(extract), typeof(to_unconstrained)}(
        names, values(priors), offset, extract, to_unconstrained
    )
end

# Replace one slot of a particle: functional for an SVector, in place for a mutable vector.
@inline _set_slot(p::SVector, val, idx) = setindex(p, val, idx)
@inline function _set_slot(p::AbstractVector, val, idx)
    p[idx] = val
    return p
end

# The learned block of every particle as an `SVector{H}`, and its weighted mean and covariance.
function _theta_block(particles::AbstractVector, learned::LearnedHyperparams{H}) where {H}
    slots = hyper_slots(learned)
    return [SVector{H}(ntuple(k -> particles[i][slots[k]], Val(H))) for i in eachindex(particles)]
end

function _weighted_theta_moments(θs::AbstractVector{<:SVector{H}}, weights::AbstractVector) where {H}
    wsum = sum(weights)
    θ̄ = sum(weights[i] .* θs[i] for i in eachindex(θs)) ./ wsum
    V = sum(weights[i] .* ((θs[i] - θ̄) * (θs[i] - θ̄)') for i in eachindex(θs)) ./ wsum
    return θ̄, V
end

"""
    build_hyperparam_updater(learned; discount = 0.95, jitter_floor_fraction = DEFAULT_JITTER_FLOOR_FRACTION,
                             forgetting_memory_days = (;), rng = Random.default_rng(), dt = 1.0) -> update!

`update!(particles, weights)` refreshes every particle's learned slots in place by the Liu-West
kernel `θᵢ ← a θᵢ + (1 - a) θ̄ + ε`, `ε ~ N(0, (1 - a²) V)` with `a = (3δ - 1) / 2δ` and `θ̄, V` the
weighted cloud moments, which preserves the weighted mean and variance. The jitter variance is
floored at `jitter_floor_fraction` of each prior's variance so a collapsed cloud can re-expand.
Parameters named in `forgetting_memory_days` are additionally pulled toward their prior by Kulhavý
forgetting with `λ = exp(-dt / memory)`: the marginal `N(m, v)` becomes the geometric mean of
itself and the prior. Call between `correct!` and `predict!` on `state(pf).xprev` with
`expweights(pf)`.
"""
function build_hyperparam_updater(
        learned::LearnedHyperparams{H}; discount::Real = 0.95,
        jitter_floor_fraction::Real = DEFAULT_JITTER_FLOOR_FRACTION, forgetting_memory_days = (;),
        rng = Random.default_rng(), dt::Real = 1.0,
    ) where {H}
    1 / 3 <= discount <= 1 ||
        throw(ArgumentError("discount must be in [1/3, 1] so the shrinkage (3δ - 1) / 2δ lies in [0, 1], got $discount"))
    jitter_floor_fraction >= 0 || throw(ArgumentError("jitter_floor_fraction must be non-negative, got $jitter_floor_fraction"))
    memory = Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in pairs(forgetting_memory_days))
    unknown = setdiff(keys(memory), learned.names)
    isempty(unknown) || throw(
        ArgumentError("forgetting_memory_days names parameters outside the learned block: $(sort!(collect(unknown)))")
    )
    all(>(0), values(memory)) || throw(ArgumentError("forgetting memories must be positive (Inf disables), got $memory"))

    a = (3 * discount - 1) / (2 * discount)
    h2 = 1 - a^2
    slots = hyper_slots(learned)
    hv = Val(H)
    prior_var = SVector{H}(ntuple(k -> prior_unconstrained_variance(learned.priors[k]), hv))
    prior_mean = SVector{H}(ntuple(k -> prior_unconstrained_mean(learned.priors[k]), hv))
    floor_diag = max.(jitter_floor_fraction .* prior_var, 1.0e-10)
    ridge = SMatrix{H, H, Float64}(I) .* 1.0e-10   # keeps the jitter Cholesky positive definite
    forget = SVector{H}(
        ntuple(k -> (days = get(memory, learned.names[k], Inf); isfinite(days) ? -expm1(-dt / days) : 0.0), hv)
    )
    has_forgetting = any(>(0), forget)

    function update!(particles::AbstractVector, weights::AbstractVector)
        θs = _theta_block(particles, learned)
        θ̄, V = _weighted_theta_moments(θs, weights)
        shift, widen, scale = zero(θ̄), zero(θ̄), one.(θ̄)
        if has_forgetting
            v = diag(V)
            denom = (1 .- forget) .* prior_var .+ forget .* v
            v_new = v .* prior_var ./ denom
            shift = forget .* v ./ denom .* (prior_mean .- θ̄)
            widen = max.(v_new .- v, 0.0)
            scale = SVector{H}(ntuple(k -> v[k] > 0 ? sqrt(min(v_new[k] / v[k], 1.0)) : 1.0, hv))
            V = V .* (scale * scale')
        end
        jittered = h2 .* V
        lift = Diagonal(max.(zero(floor_diag), floor_diag .- diag(jittered)) .+ widen)
        Ljit = cholesky(Symmetric(jittered + lift + ridge)).L
        for i in eachindex(particles)
            θ_new = a .* θs[i] .+ (1 - a) .* θ̄ .+ Ljit * SVector{H}(randn(rng, H))
            has_forgetting && (θ_new = θ_new .+ shift .+ a .* (scale .- 1) .* (θs[i] .- θ̄))
            p = particles[i]
            for k in 1:H
                p = _set_slot(p, θ_new[k], slots[k])
            end
            particles[i] = p
        end
        return particles
    end
    return update!
end
