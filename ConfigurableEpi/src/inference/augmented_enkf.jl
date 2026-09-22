# An ensemble Kalman filter for the augmented-noise dynamics `x⁺ = f(x, u, p, t, w)` and
# measurement `y = h(x, u, p, t, v)` this package builds, where `nw` and `nv` need not equal `nx`
# and `ny`. It adds methods to LowLevelParticleFilters' generics, so `forward_trajectory` works.

"""
    AugmentedEnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny = size(R2, 1),
                                  p = nothing, Ts = 1.0, inflation = 1.0, rng = Xoshiro(),
                                  threads = false, names = nothing)

Ensemble Kalman filter with `N` members drawn from `d0`, process noise `w ~ N(0, R1)` applied
inside `dynamics` and measurement noise `v ~ N(0, R2)` inside `measurement`. `inflation >= 1`
multiplies the ensemble spread after each propagation; `threads` parallelises member propagation.
"""
mutable struct AugmentedEnsembleKalmanFilter{DT, MT, R1T, R2T, D0T, ET, XT, RT, P, RNGT} <: AbstractKalmanFilter
    dynamics::DT
    measurement::MT
    R1::R1T
    R2::R2T
    d0::D0T
    ensemble::ET
    x::XT          # cached ensemble mean
    R::RT          # cached ensemble covariance
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

function AugmentedEnsembleKalmanFilter(
        dynamics, measurement, R1::AbstractMatrix, R2::AbstractMatrix, d0, N::Integer;
        nu::Int, ny::Int = size(R2, 1), p = nothing, Ts::Real = 1.0, inflation::Real = 1.0,
        rng = Random.Xoshiro(), threads::Bool = false, names = nothing,
    )
    N > 1 || throw(ArgumentError("ensemble size must be at least 2, got $N"))
    size(R1, 1) == size(R1, 2) || throw(ArgumentError("R1 must be square, got $(size(R1))"))
    size(R2, 1) == size(R2, 2) || throw(ArgumentError("R2 must be square, got $(size(R2))"))
    inflation >= 1.0 || throw(ArgumentError("inflation must be >= 1.0 (below 1 shrinks the ensemble), got $inflation"))
    ensemble = [rand(rng, d0) for _ in 1:N]
    nx = length(ensemble[1])
    x0 = _ensemble_mean(ensemble)
    signal_names = names === nothing ?
        SignalNames(x = fill("", nx), u = fill("", max(nu, 1)), y = fill("", ny), name = "AugmentedEnKF") : names
    return AugmentedEnsembleKalmanFilter(
        dynamics, measurement, R1, R2, d0, ensemble, x0, _ensemble_cov(ensemble, x0), 0, float(Ts),
        ny, nu, nx, size(R1, 1), size(R2, 1), p, rng, float(inflation), threads, signal_names,
    )
end

function _ensemble_mean(ensemble)
    x̄ = collect(float.(ensemble[1]))
    @inbounds for i in 2:length(ensemble)
        x̄ .+= ensemble[i]
    end
    return x̄ ./= length(ensemble)
end

function _ensemble_cov(ensemble, x̄)
    R = zeros(eltype(x̄), length(x̄), length(x̄))
    @inbounds for member in ensemble
        δx = member .- x̄
        mul!(R, δx, δx', 1, 1)
    end
    return R ./= (length(ensemble) - 1)
end

function _update_ensemble_stats!(f::AugmentedEnsembleKalmanFilter)
    f.x = _ensemble_mean(f.ensemble)
    f.R = _ensemble_cov(f.ensemble, f.x)
    return nothing
end

_noise_factor(R::Diagonal) = Diagonal(sqrt.(diag(R)))
_noise_factor(R::AbstractMatrix) = isdiag(R) ? Diagonal(sqrt.(diag(R))) : cholesky(Symmetric(Matrix(R))).L
_draw_noise(rng, L, n::Int, N::Int) = [L * randn(rng, n) for _ in 1:N]

LLPF.num_particles(f::AugmentedEnsembleKalmanFilter) = length(f.ensemble)
LLPF.particles(f::AugmentedEnsembleKalmanFilter) = f.ensemble
LLPF.parameters(f::AugmentedEnsembleKalmanFilter) = f.p
LLPF.index(f::AugmentedEnsembleKalmanFilter) = f.t
LLPF.dynamics(f::AugmentedEnsembleKalmanFilter) = f.dynamics
LLPF.measurement(f::AugmentedEnsembleKalmanFilter) = f.measurement
LLPF.state(f::AugmentedEnsembleKalmanFilter) = f.x
LLPF.covariance(f::AugmentedEnsembleKalmanFilter) = f.R
LLPF.particletype(f::AugmentedEnsembleKalmanFilter) = eltype(f.ensemble)
LLPF.covtype(f::AugmentedEnsembleKalmanFilter) = Matrix{eltype(eltype(f.ensemble))}

"""
    reset!(f::AugmentedEnsembleKalmanFilter; x0 = nothing, t = 0)

Resample the ensemble from `d0` (or from `d0`'s covariance around `x0`) and rewind the step
counter. Consumes `f.rng`: reproducible candidate evaluations must re-seed it.
"""
function LLPF.reset!(f::AugmentedEnsembleKalmanFilter; x0 = nothing, t = 0)
    d = x0 === nothing ? f.d0 : MvNormal(collect(x0), f.d0.Σ)
    @inbounds for i in eachindex(f.ensemble)
        f.ensemble[i] = rand(f.rng, d)
    end
    f.t = t
    _update_ensemble_stats!(f)
    return nothing
end

"""
    predict!(f::AugmentedEnsembleKalmanFilter, u, p = parameters(f), t = index(f) * f.Ts; R1 = f.R1, inflation = f.inflation)

Propagate every member through `dynamics(xᵢ, u, p, t, wᵢ)` with `wᵢ ~ N(0, R1)` pre-drawn for the
whole ensemble, then inflate the spread.
"""
function LLPF.predict!(
        f::AugmentedEnsembleKalmanFilter, u::AbstractVector, p = LLPF.parameters(f), t::Real = LLPF.index(f) * f.Ts;
        R1 = f.R1, inflation = f.inflation,
    )
    N = length(f.ensemble)
    W = _draw_noise(f.rng, _noise_factor(R1), size(R1, 1), N)
    if f.threads
        Threads.@threads for i in 1:N
            @inbounds f.ensemble[i] = f.dynamics(f.ensemble[i], u, p, t, W[i])
        end
    else
        @inbounds for i in 1:N
            f.ensemble[i] = f.dynamics(f.ensemble[i], u, p, t, W[i])
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
    (; ll, e, S, Sᵪ, K) = correct!(f::AugmentedEnsembleKalmanFilter, u, y, p = parameters(f), t = index(f) * f.Ts; R2 = f.R2)

Stochastic EnKF update in noisy-predicted-observation form: each member's predicted observation
carries its own `vᵢ ~ N(0, R2)`, so the innovation covariance is the sample covariance of that
ensemble with no second `R2` term, and members are updated against the unperturbed `y`.
"""
function LLPF.correct!(
        f::AugmentedEnsembleKalmanFilter, u::AbstractVector, y::AbstractVector, p = LLPF.parameters(f),
        t::Real = LLPF.index(f) * f.Ts; R2 = f.R2,
    )
    N, nx, ny, X = length(f.ensemble), f.nx, f.ny, f.ensemble
    T = promote_type(eltype(y), eltype(eltype(X)), Float64)
    V = _draw_noise(f.rng, _noise_factor(R2), size(R2, 1), N)
    Y = Matrix{T}(undef, ny, N)
    if f.threads
        Threads.@threads for i in 1:N
            @inbounds Y[:, i] = f.measurement(X[i], u, p, t, V[i])
        end
    else
        @inbounds for i in 1:N
            Y[:, i] = f.measurement(X[i], u, p, t, V[i])
        end
    end
    x̄ = _ensemble_mean(X)
    ȳ = vec(sum(Y, dims = 2)) ./ N
    Xa = Matrix{T}(undef, nx, N)
    @inbounds for i in 1:N
        Xa[:, i] = X[i] .- x̄
    end
    Ya = Y .- ȳ
    S = (Ya * Ya') ./ (N - 1)
    S = (S .+ S') ./ 2
    Sᵪ = cholesky(Symmetric(S); check = false)
    issuccess(Sᵪ) || error(
        "Cholesky factorization of the innovation covariance failed at step $(f.t); S = $S. " *
            "An ensemble collapsed in observation space: raise n_ensemble or the inflation factor.",
    )
    K = ((Xa * Ya') ./ (N - 1)) / Sᵪ
    e = y .- ȳ
    @inbounds for i in 1:N
        innov = K * (y .- view(Y, :, i))
        X[i] = X[i] isa SVector ? X[i] + innov : X[i] .+ innov
    end
    ll = -0.5 * (ny * log(2π) + logdet(Sᵪ) + dot(e, Sᵪ \ e))
    _update_ensemble_stats!(f)
    return (; ll, e, S, Sᵪ, K)
end

function LLPF.update!(
        f::AugmentedEnsembleKalmanFilter, u::AbstractVector, y::AbstractVector, p = LLPF.parameters(f),
        t::Real = LLPF.index(f) * f.Ts,
    )
    ret = LLPF.correct!(f, u, y, p, t)
    LLPF.predict!(f, u, p, t)
    return ret
end

(f::AugmentedEnsembleKalmanFilter)(u::AbstractVector, y::AbstractVector, p = LLPF.parameters(f), t = LLPF.index(f) * f.Ts) =
    LLPF.update!(f, u, y, p, t)

LLPF.sample_state(f::AugmentedEnsembleKalmanFilter, p = LLPF.parameters(f); noise = true) =
    noise ? rand(f.rng, f.d0) : collect(mean(f.d0))

function LLPF.sample_state(f::AugmentedEnsembleKalmanFilter, x, u, p = LLPF.parameters(f), t = 0; noise = true)
    w = noise ? only(_draw_noise(f.rng, _noise_factor(f.R1), f.nw, 1)) : zeros(f.nw)
    return f.dynamics(x, u, p, t, w)
end

function LLPF.sample_measurement(f::AugmentedEnsembleKalmanFilter, x, u, p = LLPF.parameters(f), t = 0; noise = true)
    v = noise ? only(_draw_noise(f.rng, _noise_factor(f.R2), f.nv, 1)) : zeros(f.nv)
    return f.measurement(x, u, p, t, v)
end
