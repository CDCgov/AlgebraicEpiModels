# ============================================================================
# State Filter Method Types
# ============================================================================

"""
    StateFilterMethod

Abstract base type for state filtering methods used in sequential inference.
"""
abstract type StateFilterMethod end

"""
    UKF(; dt=1.0, supersample=2, obs_jitter=1.0) <: StateFilterMethod

Configure an unscented Kalman filter. `dt` is the observation interval in model
time units, `supersample` is the number of ODE substeps per interval, and
`obs_jitter` is the accumulator process-noise scale used by the Gaussian filter.
All three numerical settings must be positive.
"""
struct UKF{F <: AbstractFloat} <: StateFilterMethod
    dt::F
    supersample::Int
    obs_jitter::F

    function UKF(dt::Real, supersample::Integer, obs_jitter::Real)
        dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
        supersample > 0 || throw(
            ArgumentError("supersample must be positive, got $supersample")
        )
        obs_jitter > 0 || throw(
            ArgumentError("obs_jitter must be positive for UKF, got $obs_jitter")
        )
        dt_float, jitter_float = promote(float(dt), float(obs_jitter))
        return new{typeof(dt_float)}(dt_float, Int(supersample), jitter_float)
    end
end

UKF(; dt::Real = 1.0, supersample::Integer = 2, obs_jitter::Real = 1.0) =
    UKF(dt, supersample, obs_jitter)

"""
    EnKF(n_ensemble; dt=1.0, supersample=2, obs_jitter=0.0, inflation=1.0, threads=false)
        <: StateFilterMethod

Ensemble Kalman filter over an ensemble of model realizations, built as an
`AugmentedEnsembleKalmanFilter` so it shares the UKF's augmented-noise model
semantics — the latent innovation lands before the weekly flow and the observation-noise
scale stays a function of the state.

`dt` and `supersample` configure the shared model dynamics, as for [`UKF`](@ref) and
[`PF`](@ref). `obs_jitter` defaults to zero: the accumulator whisker is a UKF-smoother
regularizer that an ensemble filter does not need. `inflation` multiplies the ensemble spread
after each propagation (`1.0` disables it) and must not be below 1, since shrinking the spread
drives the filter to divergence. `threads` parallelizes the per-member propagation; the noise
is pre-drawn for the whole ensemble, so a seeded run reproduces either way.
"""
struct EnKF{F <: AbstractFloat} <: StateFilterMethod
    n_ensemble::Int
    dt::F
    supersample::Int
    obs_jitter::F
    inflation::F
    threads::Bool

    function EnKF(
            n_ensemble::Integer, dt::Real, supersample::Integer, obs_jitter::Real,
            inflation::Real, threads::Bool,
        )
        n_ensemble > 1 || throw(
            ArgumentError("n_ensemble must be at least 2, got $n_ensemble")
        )
        dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
        supersample > 0 || throw(
            ArgumentError("supersample must be positive, got $supersample")
        )
        obs_jitter >= 0 || throw(
            ArgumentError("obs_jitter must be non-negative for EnKF, got $obs_jitter")
        )
        inflation >= 1 || throw(
            ArgumentError("inflation must be at least 1.0, got $inflation")
        )
        dt_float, jitter_float, inflation_float =
            promote(float(dt), float(obs_jitter), float(inflation))
        return new{typeof(dt_float)}(
            Int(n_ensemble), dt_float, Int(supersample), jitter_float, inflation_float,
            threads,
        )
    end
end

EnKF(
    n_ensemble::Integer; dt::Real = 1.0, supersample::Integer = 2, obs_jitter::Real = 0.0,
    inflation::Real = 1.0, threads::Bool = false,
) = EnKF(n_ensemble, dt, supersample, obs_jitter, inflation, threads)

"""
    PF(n_particles; dt=1.0, supersample=2, obs_jitter=0.0, threads=true) <: StateFilterMethod

Bootstrap particle filter (LowLevelParticleFilters' `AdvancedParticleFilter`) for
nonlinear, non-Gaussian state estimation — in particular discrete count
observations scored by their exact log-likelihood rather than a Gaussian
approximation.

`dt` and `supersample` configure the shared model dynamics. `obs_jitter`
defaults to zero because the PF's exact observation likelihood already models
observation noise and does not use the UKF smoothing whisker.

`threads` (default `true`) parallelizes the per-particle propagation and weighting
across Julia's threads (LLPF's `Threads.@threads :static` loops); in a
single-threaded session the loops degenerate to serial execution. Propagation
noise is then drawn from a per-thread RNG pool seeded from the filter `rng` (see
`build_pf_dynamics`), so a seeded run reproduces exactly for a fixed
`Threads.nthreads()` — a different thread count is a different draw sequence.
"""
struct PF{F <: AbstractFloat} <: StateFilterMethod
    n_particles::Int
    dt::F
    supersample::Int
    obs_jitter::F
    threads::Bool

    function PF(
            n_particles::Integer, dt::Real, supersample::Integer, obs_jitter::Real,
            threads::Bool = true
        )
        n_particles > 0 || throw(
            ArgumentError("n_particles must be positive, got $n_particles")
        )
        dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
        supersample > 0 || throw(
            ArgumentError("supersample must be positive, got $supersample")
        )
        obs_jitter >= 0 || throw(
            ArgumentError("obs_jitter must be non-negative for PF, got $obs_jitter")
        )
        dt_float, jitter_float = promote(float(dt), float(obs_jitter))
        return new{typeof(dt_float)}(
            Int(n_particles), dt_float, Int(supersample), jitter_float, threads
        )
    end
end

PF(
    n_particles::Integer;
    dt::Real = 1.0,
    supersample::Integer = 2,
    obs_jitter::Real = 0.0,
    threads::Bool = true,
) = PF(n_particles, dt, supersample, obs_jitter, threads)
