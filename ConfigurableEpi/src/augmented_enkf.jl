# ============================================================================
# Augmented-noise ensemble Kalman filter
# ============================================================================
#
# `AugmentedEnsembleKalmanFilter` extends `LowLevelParticleFilters.EnsembleKalmanFilter`
# to work with ConfigurableEpi's shared dynamics and measurement functions which are augmented
# e.g `dynamics(x, u, p, t, w)` and `measurement(x, u, p, t, v)`.
#

"""
    AugmentedEnsembleKalmanFilter

Ensemble Kalman filter for augmented-noise dynamics and measurement functions:

    x⁺ = dynamics(x, u, p, t, w),   w ~ N(0, R1)
    y  = measurement(x, u, p, t, v),  v ~ N(0, R2)

The noise dimensions are free: `nw = size(R1, 1)` need not equal `nx`, and `nv = size(R2, 1)`
need not equal `ny`. Subtypes `LowLevelParticleFilters.AbstractKalmanFilter`, so
`forward_trajectory`, `KalmanFilteringSolution` and `smooth` work on it directly.

Construct with [`AugmentedEnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; …)`](@ref).
"""
mutable struct AugmentedEnsembleKalmanFilter{DT, MT, R1T, R2T, D0T, ET, XT, RT, P, RNGT} <:
    AbstractKalmanFilter
    dynamics::DT
    measurement::MT
    R1::R1T          # nw × nw process-noise covariance
    R2::R2T          # nv × nv measurement-noise covariance
    d0::D0T          # initial state distribution
    ensemble::ET     # N members
    x::XT            # cached ensemble mean
    R::RT            # cached ensemble sample covariance
    t::Int
    Ts::Float64
    ny::Int
    nu::Int
    nx::Int
    nw::Int
    nv::Int
    p::P
    rng::RNGT
    inflation::Float64
    threads::Bool
    names::SignalNames
end

"""
    AugmentedEnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny, kwargs...)

Build an [`AugmentedEnsembleKalmanFilter`](@ref) with `N` members drawn from `d0`.

# Arguments
- `dynamics`: `f(x, u, p, t, w) -> x⁺`, the noise applied *before* the deterministic flow.
- `measurement`: `h(x, u, p, t, v) -> y`, the noise scale free to depend on `x`.
- `R1`: `nw × nw` process-noise covariance. `nw` need not equal `nx`.
- `R2`: `nv × nv` measurement-noise covariance. `nv` need not equal `ny`.
- `d0`: initial state distribution (anything `rand(rng, d0)` accepts).
- `N`: ensemble size.

# Keywords
- `nu`: input dimension (required; `0` for this codebase's models).
- `ny`: observation dimension. Defaults to `size(R2, 1)`, which is only right when
  `nv == ny` — pass it explicitly for augmented measurements.
- `p`, `Ts`, `rng`, `threads`, `names`: as for LLPF's filters.
- `inflation`: multiplicative ensemble-spread inflation applied after propagation. `1.0`
  disables it.
"""
function AugmentedEnsembleKalmanFilter(
        dynamics,
        measurement,
        R1::AbstractMatrix,
        R2::AbstractMatrix,
        d0,
        N::Integer;
        nu::Int,
        ny::Int = size(R2, 1),
        p = nothing,
        Ts::Real = 1.0,
        inflation::Real = 1.0,
        rng = Random.Xoshiro(),
        threads::Bool = false,
        names = nothing,
    )
    N > 1 || throw(ArgumentError("ensemble size must be at least 2, got $N"))
    size(R1, 1) == size(R1, 2) ||
        throw(ArgumentError("R1 must be square, got $(size(R1))"))
    size(R2, 1) == size(R2, 2) ||
        throw(ArgumentError("R2 must be square, got $(size(R2))"))
    inflation >= 1.0 || throw(
        ArgumentError(
            "inflation must be >= 1.0 (values below 1 shrink the ensemble and " *
                "drive the filter to divergence), got $inflation",
        )
    )

    ensemble = [rand(rng, d0) for _ in 1:N]
    nx = length(ensemble[1])
    x0 = _ensemble_mean(ensemble)
    R0 = _ensemble_cov(ensemble, x0)
    signal_names = names === nothing ?
        SignalNames(
            x = fill("", nx), u = fill("", max(nu, 1)), y = fill("", ny),
            name = "AugmentedEnKF",
        ) : names

    return AugmentedEnsembleKalmanFilter(
        dynamics, measurement, R1, R2, d0,
        ensemble, x0, R0,
        0, float(Ts),
        ny, nu, nx, size(R1, 1), size(R2, 1),
        p, rng, float(inflation), threads, signal_names,
    )
end

# --- ensemble statistics -----------------------------------------------------------
# Local rather than reused from LLPF: `_ensemble_mean`/`_ensemble_cov` there are internal
# (non-exported) and use the package's `@bangbang` macro.

function _ensemble_mean(ensemble)
    N = length(ensemble)
    x̄ = collect(float.(ensemble[1]))
    @inbounds for i in 2:N
        x̄ .+= ensemble[i]
    end
    x̄ ./= N
    return x̄
end

function _ensemble_cov(ensemble, x̄)
    N = length(ensemble)
    nx = length(x̄)
    R = zeros(eltype(x̄), nx, nx)
    @inbounds for i in 1:N
        δx = ensemble[i] .- x̄
        mul!(R, δx, δx', 1, 1)
    end
    R ./= (N - 1)
    return R
end

function _update_ensemble_stats!(f::AugmentedEnsembleKalmanFilter)
    f.x = _ensemble_mean(f.ensemble)
    f.R = _ensemble_cov(f.ensemble, f.x)
    return nothing
end

# --- noise draws -------------------------------------------------------------------

_noise_factor(R::Diagonal) = Diagonal(sqrt.(diag(R)))
_noise_factor(R::AbstractMatrix) = _is_diagonal(R) ? Diagonal(sqrt.(diag(R))) :
    cholesky(Symmetric(Matrix(R))).L

function _is_diagonal(R::AbstractMatrix)
    n = size(R, 1)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        iszero(R[i, j]) || return false
    end
    return true
end

_draw_noise(rng, L, n::Int, N::Int) = [L * randn(rng, n) for _ in 1:N]

# --- interface accessors -----------------------------------------------------------


LLPF.num_particles(f::AugmentedEnsembleKalmanFilter) = length(f.ensemble)
LLPF.particles(f::AugmentedEnsembleKalmanFilter) = f.ensemble
LLPF.parameters(f::AugmentedEnsembleKalmanFilter) = f.p
LLPF.index(f::AugmentedEnsembleKalmanFilter) = f.t
LLPF.dynamics(f::AugmentedEnsembleKalmanFilter) = f.dynamics
LLPF.measurement(f::AugmentedEnsembleKalmanFilter) = f.measurement

"""
    state(f::AugmentedEnsembleKalmanFilter)

Cached ensemble mean.
"""
LLPF.state(f::AugmentedEnsembleKalmanFilter) = f.x

"""
    covariance(f::AugmentedEnsembleKalmanFilter)

Cached ensemble sample covariance.
"""
LLPF.covariance(f::AugmentedEnsembleKalmanFilter) = f.R

LLPF.particletype(f::AugmentedEnsembleKalmanFilter) = eltype(f.ensemble)
LLPF.covtype(f::AugmentedEnsembleKalmanFilter) = Matrix{eltype(eltype(f.ensemble))}

"""
    reset!(f::AugmentedEnsembleKalmanFilter; x0 = nothing, t = 0)

Resample the ensemble from `d0` (or from `d0`'s covariance around `x0`) and rewind the step
counter. Note that this CONSUMES `f.rng` — an evaluation meant to be reproducible across
candidate parameter vectors must re-seed `f.rng`, not merely call `reset!`.
"""
function LLPF.reset!(f::AugmentedEnsembleKalmanFilter; x0 = nothing, t = 0)
    N = LLPF.num_particles(f)
    d = x0 === nothing ? f.d0 : MvNormal(collect(x0), f.d0.Σ)
    @inbounds for i in 1:N
        f.ensemble[i] = rand(f.rng, d)
    end
    f.t = t
    _update_ensemble_stats!(f)
    return nothing
end

"""
    predict!(f::AugmentedEnsembleKalmanFilter, u, p = parameters(f), t = index(f) * f.Ts;
             R1 = f.R1, inflation = f.inflation)

Propagate every member through `dynamics(xᵢ, u, p, t, wᵢ)` with `wᵢ ~ N(0, R1)` pre-drawn for
the whole ensemble. The noise enters the dynamics rather than being added to its output, so a
model that applies its latent innovation before an internal flow keeps that ordering.
"""
function LLPF.predict!(
        f::AugmentedEnsembleKalmanFilter,
        u::AbstractVector,
        p = LLPF.parameters(f),
        t::Real = LLPF.index(f) * f.Ts;
        R1 = f.R1,
        inflation = f.inflation,
    )
    fd = f.dynamics
    N = LLPF.num_particles(f)
    W = _draw_noise(f.rng, _noise_factor(R1), size(R1, 1), N)

    if f.threads
        Threads.@threads for i in 1:N
            @inbounds f.ensemble[i] = fd(f.ensemble[i], u, p, t, W[i])
        end
    else
        @inbounds for i in 1:N
            f.ensemble[i] = fd(f.ensemble[i], u, p, t, W[i])
        end
    end

    if inflation > 1.0
        x̄ = _ensemble_mean(f.ensemble)
        @inbounds for i in 1:N
            f.ensemble[i] = x̄ .+ inflation .* (f.ensemble[i] .- x̄)
        end
    end

    f.t += 1
    _update_ensemble_stats!(f)
    return nothing
end

"""
    (; ll, e, S, Sᵪ, K) = correct!(f::AugmentedEnsembleKalmanFilter, u, y,
                                   p = parameters(f), t = index(f) * f.Ts; R2 = f.R2)

Stochastic-EnKF measurement update in its noisy-predicted-observation form: each member's
predicted observation carries its own `vᵢ ~ N(0, R2)`, so the innovation covariance is the
sample covariance of that noisy ensemble with **no** second `R2` term, and members are updated
against the unperturbed `y`.

Returns the log-likelihood, innovation, innovation covariance, its Cholesky factor and the
Kalman gain — the same five-element shape `forward_trajectory` and `marginal_loglik` destructure.
"""
function LLPF.correct!(
        f::AugmentedEnsembleKalmanFilter,
        u::AbstractVector,
        y::AbstractVector,
        p = LLPF.parameters(f),
        t::Real = LLPF.index(f) * f.Ts;
        R2 = f.R2,
    )
    h = f.measurement
    N = LLPF.num_particles(f)
    nx, ny = f.nx, f.ny
    X = f.ensemble
    T = promote_type(eltype(y), eltype(eltype(X)), Float64)

    V = _draw_noise(f.rng, _noise_factor(R2), size(R2, 1), N)
    Y = Matrix{T}(undef, ny, N)
    if f.threads
        Threads.@threads for i in 1:N
            @inbounds Y[:, i] = h(X[i], u, p, t, V[i])
        end
    else
        @inbounds for i in 1:N
            Y[:, i] = h(X[i], u, p, t, V[i])
        end
    end

    x̄ = _ensemble_mean(X)
    ȳ = vec(sum(Y, dims = 2)) ./ N

    Xa = Matrix{T}(undef, nx, N)
    @inbounds for i in 1:N
        Xa[:, i] = X[i] .- x̄
    end
    Ya = Y .- ȳ

    # The measurement noise is already inside `Y`, so it is NOT added again here.
    S = (Ya * Ya') ./ (N - 1)
    S = (S .+ S') ./ 2
    Sᵪ = cholesky(Symmetric(S); check = false)
    issuccess(Sᵪ) || error(
        "Cholesky factorization of the innovation covariance failed at step $(f.t); " *
            "S = $S. An ensemble collapsed in observation space — raise n_ensemble or " *
            "the inflation factor."
    )

    K = ((Xa * Ya') ./ (N - 1)) / Sᵪ
    e = y .- ȳ

    # Unperturbed `y`: the perturbation lives on the predicted-observation side.
    @inbounds for i in 1:N
        innov = K * (y .- view(Y, :, i))
        X[i] = X[i] isa SVector ? X[i] + innov : X[i] .+ innov
    end

    ll = -0.5 * (ny * log(2π) + logdet(Sᵪ) + dot(e, Sᵪ \ e))

    _update_ensemble_stats!(f)
    return (; ll, e, S, Sᵪ, K)
end

"""
    update!(f::AugmentedEnsembleKalmanFilter, u, y, p = parameters(f), t = index(f) * f.Ts)

One filtering step: `correct!` then `predict!`.
"""
function LLPF.update!(
        f::AugmentedEnsembleKalmanFilter, u::AbstractVector, y::AbstractVector,
        p = LLPF.parameters(f), t::Real = LLPF.index(f) * f.Ts,
    )
    ret = LLPF.correct!(f, u, y, p, t)
    LLPF.predict!(f, u, p, t)
    return ret
end

function (f::AugmentedEnsembleKalmanFilter)(
        u::AbstractVector, y::AbstractVector,
        p = LLPF.parameters(f), t = LLPF.index(f) * f.Ts,
    )
    return LLPF.update!(f, u, y, p, t)
end

# --- simulation hooks --------------------------------------------------------------
# Augmented forms: the noise goes INTO the function, mirroring
# `LowLevelParticleFilters.sample_state(::UnscentedKalmanFilter{false,<:Any,true,<:Any}, …)`.

LLPF.sample_state(f::AugmentedEnsembleKalmanFilter, p = LLPF.parameters(f); noise = true) =
    noise ? rand(f.rng, f.d0) : collect(mean(f.d0))

function LLPF.sample_state(
        f::AugmentedEnsembleKalmanFilter, x, u, p = LLPF.parameters(f), t = 0; noise = true,
    )
    w = noise ? only(_draw_noise(f.rng, _noise_factor(f.R1), f.nw, 1)) : zeros(f.nw)
    return f.dynamics(x, u, p, t, w)
end

function LLPF.sample_measurement(
        f::AugmentedEnsembleKalmanFilter, x, u, p = LLPF.parameters(f), t = 0; noise = true,
    )
    v = noise ? only(_draw_noise(f.rng, _noise_factor(f.R2), f.nv, 1)) : zeros(f.nv)
    return f.measurement(x, u, p, t, v)
end
