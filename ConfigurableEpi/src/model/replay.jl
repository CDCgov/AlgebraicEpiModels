# Replay initialisation: the model's own net integrated over an incidence history with the
# infection loop opened. Every other transition runs as it is; infections enter each cell at a
# prescribed rate, shared over its susceptible classes by relative susceptibility.

is_infection_transition(name) = startswith(string(name), "transmission") || startswith(string(name), "importation")

"""
    InfectionRoutes(cells, source, targets, cell, weight)

The distinct ways a host enters infection: for route `k` its `source` compartment, the
compartments it feeds (`targets[k]`, the first infected stage and any observation chain), the
index `cell[k]` into `cells` of the incidence that drives it, and its relative susceptibility
`weight[k]`. Read off a net by [`infection_routes`](@ref).
"""
struct InfectionRoutes{K}
    cells::Vector{K}
    source::Vector{Int}
    targets::Vector{Vector{Int}}
    cell::Vector{Int}
    weight::Vector{Float64}
end

"""
    infection_routes(pn, layout; cells, cell, weight = Returns(1.0), is_infection = is_infection_transition)
        -> InfectionRoutes

Each infection transition (by default a `transmission…` or `importation…` one) moves one host from
a source compartment into its products, leaving the infector out; transitions sharing a source and
products are one route. `cell(name)`, one of `cells`, and `weight(name)` must agree across a route.
"""
function infection_routes(
        pn, layout::StateLayout; cells, cell, weight = Returns(1.0), is_infection = is_infection_transition,
    )
    cell_keys = collect(cells)
    allunique(cell_keys) || throw(ArgumentError("replay cells must be distinct; got $cell_keys"))
    position = Dict(name => k for (k, name) in enumerate(ode_names(layout)))
    index(j) = position[flatten_symbols(sname(pn, j))]
    found = Dict{Tuple{Int, Vector{Int}}, Tuple{Int, Float64}}()
    order = Tuple{Int, Vector{Int}}[]
    for i in 1:nt(pn)
        name = flatten_symbols(tname(pn, i))
        is_infection(name) || continue
        ins, outs = inputs(pn, i), outputs(pn, i)
        hosts = setdiff(ins, outs)
        length(hosts) == 1 ||
            throw(ArgumentError("infection transition `$name` does not move one host out of one compartment"))
        route = (index(only(hosts)), sort!([index(j) for j in setdiff(outs, ins)]))
        c = findfirst(==(cell(name)), cell_keys)
        c === nothing && throw(ArgumentError("infection transition `$name` is in cell $(cell(name)), not one of $cell_keys"))
        w = Float64(weight(name))
        isfinite(w) && w >= 0 || throw(ArgumentError("infection transition `$name` has weight $w"))
        if !haskey(found, route)
            found[route] = (c, w)
            push!(order, route)
        elseif found[route] != (c, w)
            throw(ArgumentError("infection transition `$name` disagrees with its route's cell or weight"))
        end
    end
    isempty(order) && throw(ArgumentError("the net has no infection transition"))
    cell_of, weight_of = [first(found[r]) for r in order], [last(found[r]) for r in order]
    unserved = [cell_keys[c] for c in eachindex(cell_keys) if !any(cell_of .== c .&& weight_of .> 0)]
    isempty(unserved) || throw(ArgumentError("no infection route with a positive weight enters cells $unserved"))
    return InfectionRoutes(cell_keys, first.(order), last.(order), cell_of, weight_of)
end

"""
    BinnedIncidence(edges, rates)

Infections per day into each cell, piecewise constant: `rates[c, b]` holds over
`[edges[b], edges[b + 1])` and the rate is zero outside the bins.
"""
struct BinnedIncidence
    edges::Vector{Float64}
    rates::Matrix{Float64}
    function BinnedIncidence(edges::AbstractVector{<:Real}, rates::AbstractMatrix{<:Real})
        length(edges) >= 2 && all(diff(edges) .> 0) ||
            throw(ArgumentError("incidence bin edges must increase and bound at least one bin"))
        size(rates, 2) == length(edges) - 1 ||
            throw(DimensionMismatch("$(size(rates, 2)) incidence bins for $(length(edges)) edges"))
        all(r -> isfinite(r) && r >= 0, rates) || throw(ArgumentError("incidence rates must be finite and non-negative"))
        return new(Float64.(edges), Float64.(rates))
    end
end

@inline function _incidence_at(incidence::BinnedIncidence, c::Integer, t::Real)
    b = searchsortedlast(incidence.edges, t)
    return 1 <= b < length(incidence.edges) ? incidence.rates[c, b] : 0.0
end

"""
    build_replay_vf(pn, rates; is_infection = is_infection_transition, defaults = (;), structure_kwargs...)
        -> petri_vf!

[`build_petri_vf`](@ref) with the rate keys of every infection transition set to zero. A key that
also serves another transition, or is fixed, cannot be switched off and is an error.
"""
function build_replay_vf(pn, rates; is_infection = is_infection_transition, defaults::NamedTuple = (;), structure_kwargs...)
    rate_class = get(structure_kwargs, :rate_class, nothing)
    key_of(name) = flatten_symbols(rate_class === nothing ? name : first(rate_class(name)))
    infection, other = Symbol[], Set{Symbol}()
    for name in tnames(pn)
        key = key_of(name)
        is_infection(flatten_symbols(name)) ? (key in infection || push!(infection, key)) : push!(other, key)
    end
    stuck = union(intersect(infection, other), intersect(infection, Symbol.(keys(get(structure_kwargs, :fixed, (;))))))
    isempty(stuck) ||
        throw(ArgumentError("rate key(s) $stuck also serve other transitions or are fixed, so the replay cannot switch them off"))
    off = NamedTuple{Tuple(infection)}(ntuple(_ -> 0.0, length(infection)))
    return build_petri_vf(pn, (latent, hyper, t) -> merge(rates(latent, hyper, t), off); defaults, structure_kwargs...)
end

"""
    ReplayModel(vf!, replay_vf!, layout, routes)
    ReplayModel(pn, layout, rates; cells, cell, weight = Returns(1.0), is_infection = is_infection_transition,
                defaults = (;), structure_kwargs...)

A model's vector field, the same field with its infection transitions off, its layout and its
[`InfectionRoutes`](@ref). The second form builds them from the net with [`build_petri_vf`](@ref),
[`build_replay_vf`](@ref) and [`infection_routes`](@ref).
"""
struct ReplayModel{F, G, L <: StateLayout, K}
    vf!::F
    replay_vf!::G
    layout::L
    routes::InfectionRoutes{K}
end

function ReplayModel(
        pn, layout::StateLayout, rates; cells, cell, weight = Returns(1.0), is_infection = is_infection_transition,
        defaults::NamedTuple = (;), structure_kwargs...,
    )
    return ReplayModel(
        build_petri_vf(pn, rates; defaults, structure_kwargs...),
        build_replay_vf(pn, rates; is_infection, defaults, structure_kwargs...),
        layout, infection_routes(pn, layout; cells, cell, weight, is_infection),
    )
end

function _add_infections!(dx, x, routes::InfectionRoutes, inflow)
    pool = zeros(eltype(x), length(inflow))
    @inbounds for k in eachindex(routes.source)
        pool[routes.cell[k]] += routes.weight[k] * x[routes.source[k]]
    end
    @inbounds for k in eachindex(routes.source)
        c = routes.cell[k]
        pool[c] > 0 || continue
        flux = inflow[c] * routes.weight[k] * x[routes.source[k]] / pool[c]
        dx[routes.source[k]] -= flux
        for j in routes.targets[k]
            dx[j] += flux
        end
    end
    return dx
end

function _assert_supply(x, routes::InfectionRoutes, inflow, h, t)
    for c in eachindex(inflow)
        need = inflow[c] * h
        need > 0 || continue
        have = sum((x[routes.source[k]] for k in eachindex(routes.source) if routes.cell[k] == c && routes.weight[k] > 0); init = zero(eltype(x)))
        need <= have || throw(
            ArgumentError(
                "the incidence prescribed for cell $(routes.cells[c]) at t = $t needs $need hosts and its " *
                    "susceptible classes hold $have; the incidence level or the waning is inconsistent with the model",
            )
        )
    end
    return nothing
end

"""
    replay_state(model::ReplayModel, incidence, x_start, hyper, latent_at, t_from, t_to;
                 scale = 1.0, dt = 1.0, supersample = 2, integrator = Rk4, keep_accumulators = false) -> Vector

The ODE state at `t_to` from `x_start` at `t_from`, integrated as [`deterministic_spinup`](@ref)
does but with `model.replay_vf!`: infections enter cell `c` at `scale × incidence`, held over each
substep, and leave its susceptible classes in proportion to `weight × occupancy`. An incidence
the susceptible classes cannot supply is an error. The state inherits no transmission parameter;
the influence of `x_start` decays at the slowest waning rate.
"""
function replay_state(
        model::ReplayModel, incidence::BinnedIncidence, x_start::AbstractVector, hyper, latent_at, t_from::Real,
        t_to::Real; scale::Real = 1.0, dt::Real = 1.0, supersample::Integer = 2, integrator::Type{<:AbstractIntegrator} = Rk4,
        keep_accumulators::Bool = false,
    )
    (; layout, routes) = model
    n_cells = length(routes.cells)
    size(incidence.rates, 1) == n_cells ||
        throw(DimensionMismatch("incidence has $(size(incidence.rates, 1)) rows for $n_cells cells"))
    length(x_start) == n_ode_states(layout) ||
        throw(DimensionMismatch("x_start has $(length(x_start)) entries for $(n_ode_states(layout)) ODE states"))
    scale >= 0 || throw(ArgumentError("the incidence scale must be non-negative, got $scale"))
    T = promote_type(eltype(x_start), typeof(float(scale)))
    inflow = zeros(T, n_cells)
    unified = build_unified_vf(model.replay_vf!, layout)
    f(x, u, p, t) = _add_infections!(unified(x, u, (p, latent_at(t)), t), x, routes, inflow)
    function prescribe!(x, t, h)
        for c in 1:n_cells
            inflow[c] = scale * _incidence_at(incidence, c, t + h / 2)
        end
        return _assert_supply(x, routes, inflow, h, t)
    end
    return _integrate(
        f, layout, T.(x_start), hyper, t_from, t_to; dt, supersample, integrator, keep_accumulators, before_substep = prescribe!,
    )
end

"""
    replay_scale_moments(replay_at, logsd; relative_step = 1.0e-3) -> (; mean, sensitivity, covariance)

Moments of a replayed state when the history is `1 / m` times the central one, `m` log-normal with
median 1 and log-sd `logsd` (the `historical_multiplier_logsd` of [`SlowModeInitialisation`](@ref)).
`replay_at(scale)` is the state at one scale; linear about its mean, the state has a rank-one
covariance along `sensitivity`, its central-difference derivative in the scale.
"""
function replay_scale_moments(replay_at, logsd::Real; relative_step::Real = 1.0e-3)
    logsd >= 0 || throw(ArgumentError("logsd must be non-negative, got $logsd"))
    scale = exp(logsd^2 / 2)
    state = replay_at(scale)
    iszero(logsd) && return (; mean = state, sensitivity = zero(state), covariance = zeros(length(state), length(state)))
    sensitivity = (replay_at(scale * (1 + relative_step)) .- replay_at(scale * (1 - relative_step))) ./ (2 * relative_step * scale)
    variance = exp(2 * logsd^2) - exp(logsd^2)
    return (; mean = state, sensitivity, covariance = variance .* (sensitivity * sensitivity'))
end
