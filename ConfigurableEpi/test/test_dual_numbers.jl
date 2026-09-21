using Test
using AlgebraicPetri: LabelledPetriNet
using Dates: Date
using ForwardDiff
using JET
using LabelledArrays: LVector
using StaticArrays: SVector
using ConfigurableEpi

# ----------------------------------------------------------------------------
# Dual number compatibility checks
#
# The closure-boxing tests use Float64 inputs only. This file is the
# AD-flavoured complement: it runs `JET.@test_opt` with `ForwardDiff.Dual`
# state, so a regression that breaks inference-on-Duals is caught even when
# Float64 inference is fine.
#
# Scope: the UKF predict/correct closures that do **not** go through
# EnsembleKalmanProcesses transforms. The four latent-dynamics closures,
# the latent-aware measurement, and the full-dynamics composition all call
# `_to_constrained` / `_to_unconstrained` → `transform_*_unconstrained` in EKP.
#
# NOTE (corrected): an earlier version of this comment claimed "EKP is not
# AD-compatible as currently used". That is **wrong**, and it misattributed the
# real blocker. Measured directly, ForwardDiff differentiates every EKP path this
# package uses, correctly and JET-clean:
#
#   d/du _to_constrained(positive_gaussian, 0.3)      = 1.34986  ( = exp'(0.3) ✓)
#   d/du _to_constrained(unit_interval, 0.3)          = 0.24446  ( = logistic'(0.3) ✓)
#   d/dc _to_unconstrained(positive_gaussian, 1.7)    = 0.58824  ( = 1/1.7 ✓)
#   gradient(constrained_values), gradient(prior_logpdf)         both fine
#   d/dρ update_single(::AR1ParamSpec)                = log 2 ✓
#   @report_opt _to_constrained(prior, ::Dual)        "No errors detected"
#
# The actual blockers on differentiating the *hyperparameter objective* are all
# concrete-`Float64` buffers one level out, and none of them involve priors. All three
# were measured, and a prototype gradient of the real objective (the one
# `build_inference` optimises) then matched central finite differences to 1.8e-9
# relative at ∇f/f = 1.05x — i.e. exact and effectively free:
#
#   1. `PositiveFactorizations.cholesky!` is signature-bounded to
#      `AbstractMatrix{T} where T<:AbstractFloat`, and `Dual <: Real` but NOT
#      `<: AbstractFloat`. Pure signature restriction: the underlying `ldlt!` is
#      already `where {T}` and self-checks `eltype(A)<:Real`
#      (PositiveFactorizations/src/cholesky.jl:21-23), so widening the two wrapper
#      methods at cholesky.jl:17-18 to `T<:Real` is enough. Verified: derivative
#      2.0791561976 vs FD 2.0791561979.
#   2. `forward_trajectory` types its own buffers off the OBSERVATIONS —
#      `e = similar(y)`, `ll = zero(eltype(particletype(kf)))`
#      (LowLevelParticleFilters/src/filtering.jl:290-291) — which are Float64.
#      Sidestepped without touching upstream: the objective only wants `.ll`, so run
#      `correct!`/`predict!` directly and accumulate at the promoted type.
#   3. `_ukf_backend` builds the filter once at Float64 (`R1`, `R2`, `P0`, `x0`).
#      Needs to be parameterised on the element type and rebuilt per gradient
#      evaluation.
#
# So this is an element-type problem end to end, NOT a missing frule/rrule — and not
# an argument for swapping to a source-based AD. Zygote in particular is a non-starter
# here because `correct!`/`predict!` mutate in place, which is precisely what it cannot
# differentiate; and at 2-4 learned hyperparameters forward mode is the right mode
# anyway (see the ∇f/f = 1.05x above — no reverse-mode constant factor beats that).
#
# All three fixes have since landed: `positive_cholesky!` calls the generic `ldlt!`
# (verified bit-identical to the old hook on Float64), `marginal_loglik` replaces
# `forward_trajectory` in the objective, and `_ukf_backend` is parameterised on the element
# type and rebuilt per gradient evaluation. The UKF hyperparameter optimizer therefore runs on
# exact AD gradients (`DEFAULT_OPTIMISER_STAGES`). These closures still stay untested *here*
# for a mundane reason: they are reached through the filter object rather than called directly,
# so the end-to-end coverage lives in `test_build_inference.jl` / `test_run_model.jl`.
#
# Capture inspection (Core.Box, mutable config) depends on closure type, not
# input type — already covered by `test_closure_boxing.jl`. Allocation under
# Duals has its own profile (each Dual carries partials) and is left to a
# future round once JET coverage is settled.
# ----------------------------------------------------------------------------

# A single-partial Dual is enough for an inference check — type stability
# either holds for `Dual{Tag, Float64, N}` or it doesn't, and one partial keeps
# the test args small.
_seed(x::Float64) = ForwardDiff.Dual(x, 1.0)

_seed_vec(xs::Vector{Float64}) = _seed.(xs)

function _run_dual_jet(label::AbstractString, closure, args::Tuple)
    return @testset "$label" begin
        JET.@test_opt target_modules = (ConfigurableEpi,) closure(args...)
    end
end

# ----------------------------------------------------------------------------
# Build dual-number cases
# ----------------------------------------------------------------------------

function _build_dual_cases()
    cases = []

    # --- seasonality forcings ---
    # The UKF objective differentiates w.r.t. the learned hyperparameters, so each forcing must
    # stay inferable when the parameter it reads carries a Dual. The model clock `t` is threaded
    # separately from state and params and is always Float64.
    push!(
        cases,
        (
            label = "CosineForcing (Dual phase/amp)",
            closure = CosineForcing(247.0),
            args = (
                (
                    seasonal_amp = _seed(0.1), seasonal_phase = _seed(15.0),
                    seasonal_kappa = 1.0,
                ),
                100.0,
            ),
        )
    )
    push!(
        cases,
        (
            label = "IndoorActivityForcing (Dual kappa)",
            # The package ships no data, so the curve is injected: a synthetic unit-mean one.
            closure = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 9, 3);
                climatology = Dict("ny" => [1 + 0.3 * cospi(2 * (k - 0.5) / 52) for k in 1:52]),
            ),
            args = (
                (
                    seasonal_amp = 0.1, seasonal_phase = 15.0,
                    seasonal_kappa = _seed(1.0),
                ),
                100.0,
            ),
        )
    )

    # --- make_lvector_constructor ---
    push!(
        cases,
        (
            label = "make_lvector_constructor (Dual)",
            closure = make_lvector_constructor((:S, :I, :R)),
            args = (_seed_vec([990.0, 10.0, 0.0]),),
        )
    )

    # --- petri_vf! / unified_vf shared setup ---
    pn = LabelledPetriNet([:S, :I], :infection => ((:S, :I) => (:I, :I)))
    rates(latent, hyper, t) = (infection = 0.001 * latent.Rt,)
    petri_vf! = build_petri_vf(pn, rates)

    du_dual = LVector(S = _seed(0.0), I = _seed(0.0))
    u_dual = LVector(S = _seed(990.0), I = _seed(10.0))
    push!(
        cases,
        (
            label = "build_petri_vf (Dual)",
            closure = petri_vf!,
            args = (du_dual, u_dual, ((;), (Rt = 1.2,)), 0.0),
        )
    )

    # --- build_unified_vf ---
    core_layout = StateLayout((:S, :I), (), (); signal_names = ())
    unified_vf = build_unified_vf(petri_vf!, core_layout)
    push!(
        cases,
        (
            label = "build_unified_vf (Dual)",
            closure = unified_vf,
            args = (_seed_vec([990.0, 10.0]), nothing, ((;), (Rt = 1.2,)), 0.0),
        )
    )

    # --- measurement: single noise (no latent_dynamics arg) ---
    # The non-latent_dynamics overload of build_measurement_model uses the
    # raw-state extract (`_extract_latent_named`), which does **not** call
    # the EKP transforms. Safe to test under Duals.
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    single_noise = NegBinomialNoise(0.0, 100.0)
    measurement_single = build_measurement_model(layout, single_noise)
    x_dual = _seed_vec([990.0, 10.0, 5.0, 0.0])
    push!(
        cases,
        (
            label = "single measurement (Dual)",
            closure = measurement_single.measure,
            args = (x_dual, SVector(50.0), NamedTuple(), 0.0, SVector(0.0, 0.0)),
        )
    )

    # --- measurement: declining ascertainment path with the level and rate as Duals, which is
    # what the UKF hyperparameter objective differentiates through when the rate is learned ---
    path_measurement = build_measurement_model(
        layout,
        (SignalObservationSpec(1, single_noise; mean_modifier = AscertainmentPath(0.2, 100.0)),),
    )
    push!(
        cases,
        (
            label = "ascertainment-path measurement (Dual)",
            closure = path_measurement.measure,
            args = (
                x_dual, SVector(50.0),
                (ascertainment = _seed(0.005), ascertainment_decline_rate = _seed(0.2)),
                0.0, SVector(0.0, 0.0),
            ),
        )
    )

    # --- measurement: multi-signal ---
    layout_multi = StateLayout(
        (:S, :I, :R),
        (:O_y1, :O_y2),
        (:Rt,);
        signal_names = (:y1, :y2)
    )
    obs_specs_multi = (
        SignalObservationSpec(
            1,
            NegBinomialNoise(0.0, 100.0)
        ),
        SignalObservationSpec(2, PoissonNoise()),
    )
    measurement_multi = build_measurement_model(layout_multi, obs_specs_multi)
    x_multi_dual = _seed_vec([990.0, 10.0, 0.0, 100.0, 40.0, 0.0])
    push!(
        cases,
        (
            label = "multi measurement (Dual)",
            closure = measurement_multi.measure,
            args = (
                x_multi_dual,
                SVector(50.0, 20.0),
                NamedTuple(),
                0.0,
                SVector(0.0, 0.0, 0.0),
            ),
        )
    )

    return cases
end

# ----------------------------------------------------------------------------
# Run the suite
# ----------------------------------------------------------------------------

@testset "Dual Number Compatibility" begin
    for case in _build_dual_cases()
        _run_dual_jet(case.label, case.closure, case.args)
    end
end
