using Test
using AlgebraicPetri: LabelledPetriNet
using Dates: Date
using JET
using LabelledArrays: LVector
using StaticArrays: SVector
using ConfigurableEpi

# Latents stored under an identity constraint, so constrained == unconstrained in these checks.
_sto(layout) = build_stochastic_update(
    layout, Tuple(RWParamSpec(n; init = unconstrained_gaussian(n, 0.0, 1.0), sigma_rate = 0.1) for n in layout.latent_names)
)


# ----------------------------------------------------------------------------
# Capture / inference / allocation helpers
# ----------------------------------------------------------------------------

function _direct_weak_capture_types(f)
    return filter(fieldtypes(typeof(f))) do T
        T === Core.Box ||
            T === Any ||
            (T <: Function && !isconcretetype(T))
    end
end

function _mutable_capture_types(f; allowed = Type[])
    return filter(fieldtypes(typeof(f))) do T
        isconcretetype(T) &&
            ismutabletype(T) &&
            !any(A -> T <: A, allowed)
    end
end

function _test_no_direct_weak_captures(label::AbstractString, f)
    bad = _direct_weak_capture_types(f)
    if !isempty(bad)
        @info "weak closure captures found" label bad closure_type = typeof(f)
    end
    return @test isempty(bad)
end

function _test_no_mutable_config_captures(label::AbstractString, f; allowed = Type[])
    bad = _mutable_capture_types(f; allowed = allowed)
    if !isempty(bad)
        @info "mutable closure configuration captures found" label bad closure_type = typeof(f)
    end
    return @test isempty(bad)
end

function _test_budget_alloc(label::AbstractString, budget, f, args...; kwargs...)
    f(args...; kwargs...) # warmup
    allocs = @allocated(f(args...; kwargs...))
    if allocs > budget
        @info "More allocations $allocs found than budget $budget" label closure_type = typeof(f)
    end
    return @test allocs <= budget
end

function _test_zero_alloc(label::AbstractString, f, args...; kwargs...)
    return _test_budget_alloc(label, 0, f, args...; kwargs...)
end

# ----------------------------------------------------------------------------
# Per-factory check runner
#
# Each case (built by `_build_cases`) is a NamedTuple with:
#   label         :: String              — testset name and log label
#   closure       :: Function             — the closure under test
#   args          :: Tuple                — hot-call positional arguments
#   check_mutable :: Bool                 — run the mutable-capture check?
#   budget        :: Union{Int, Nothing}  — max bytes per call;
#                                            0 = strict zero alloc,
#                                            nothing = skip the alloc check
#
# `closure` and `args` are passed positionally so Julia specializes the helper
# on their concrete types; JET and @allocated then see the real closure rather
# than `Any`.
# ----------------------------------------------------------------------------

function _run_factory_checks(
        label::AbstractString,
        closure,
        args::Tuple;
        check_mutable::Bool = true,
        budget::Union{Int, Nothing} = nothing
    )
    return @testset "$label" begin
        _test_no_direct_weak_captures(label, closure)
        if check_mutable
            _test_no_mutable_config_captures(label, closure)
        end
        JET.@test_opt target_modules = (ConfigurableEpi,) closure(args...)
        if budget !== nothing
            _test_budget_alloc(label, budget, closure, args...)
        end
    end
end

# ----------------------------------------------------------------------------
# Mock vectorfield for full-dynamics cases
# ----------------------------------------------------------------------------

function _mock_petri_vf!(du, u, p, _t)
    hyper, _latent = p  # full_dynamics threads (hyperparams, latent)
    du[:S] = -0.1 * u[:S] * u[:I] / 1000.0
    du[:I] = 0.1 * u[:S] * u[:I] / 1000.0
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

# ----------------------------------------------------------------------------
# Build the list of factory cases
#
# To add a new factory:
#   1. Build its closure (and any layout/spec it depends on).
#   2. Push a NamedTuple with label, closure, args, and optional flags.
#   3. Set `budget` once you've measured the steady-state allocation.
# ----------------------------------------------------------------------------

function _build_cases()
    cases = []

    # --- seasonality forcings (evaluated inside the ODE RHS) ---
    # These are callable structs rather than closures, but the same capture rules apply: an
    # abstract or mutable field would make the vectorfield dynamically dispatched. That capture
    # check is the point of listing them here.
    #
    # Budget 16, not 0: unlike the in-place `vf!`s, these RETURN a Float64, and this harness
    # invokes the callable through a dynamically dispatched `f(args...)`, which boxes the
    # result. It is a fixed 16 bytes per *measurement*, not per call — `test_seasonality.jl`
    # pins the actual steady-state allocation at zero over a million calls. (`UnitForcing`
    # measures 0 only because its constant return is folded away.)
    let hp = (seasonal_amp = 0.1, seasonal_phase = 15.0, seasonal_kappa = 1.0)
        for (label, forcing) in (
                "UnitForcing" => UnitForcing(),
                "CosineForcing" => CosineForcing(247.0),
                # The package ships no data, so the curve is injected: a synthetic unit-mean one.
                "IndoorActivityForcing" => build_seasonal_forcing(
                    SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 9, 3);
                    climatology = Dict("ny" => [1 + 0.3 * cospi(2 * (k - 0.5) / 52) for k in 1:52]),
                ),
            )
            push!(
                cases,
                (
                    label = label,
                    closure = forcing,
                    args = (hp, 100.0),
                    check_mutable = true,
                    budget = 16,
                )
            )
        end
    end

    # --- make_lvector_constructor ---
    push!(
        cases,
        (
            label = "make_lvector_constructor",
            closure = make_lvector_constructor((:S, :I, :R)),
            args = ([990.0, 10.0, 0.0],),
            check_mutable = true,
            budget = 240,
        )
    )

    # --- build_petri_vf (in-place; should target zero) ---
    pn = LabelledPetriNet([:S, :I], :infection => ((:S, :I) => (:I, :I)))
    rates(latent, hyper, t) = (infection = 0.001 * latent.Rt,)
    petri_vf! = build_petri_vf(pn, rates)
    push!(
        cases,
        (
            label = "build_petri_vf",
            closure = petri_vf!,
            args = (
                LVector(S = 0.0, I = 0.0),
                LVector(S = 990.0, I = 10.0),
                ((;), (Rt = 1.2,)),
                0.0,
            ),
            check_mutable = true,
            budget = 2000,
        )
    )

    # --- build_unified_vf (out-of-place; needs a budget) ---
    core_layout = StateLayout((:S, :I), (), (); signal_names = ())
    unified_vf = build_unified_vf(petri_vf!, core_layout)
    push!(
        cases,
        (
            label = "build_unified_vf",
            closure = unified_vf,
            args = ([990.0, 10.0], nothing, ((;), (Rt = 1.2,)), 0.0),
            check_mutable = true,
            budget = 2000,
        )
    )

    # --- latent dynamics shared setup ---
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.25), sigma_rate = FixedParam(:sigma_Rt, 0.1))
    x = [990.0, 10.0, 5.0, 0.0]
    params = (sigma_Rt = 0.1, obs_scale = 0.1)

    # --- latent dynamics with explicit process noise (4 closures) ---
    latent_aug = build_stochastic_update(layout, (rt_spec,))
    push!(
        cases,
        (
            label = "stochastic advance",
            closure = latent_aug.advance,
            args = (x, params, SVector(0.0), nothing, 0.0, 1.0),
            check_mutable = true,
            budget = 8000,
        )
    )
    push!(
        cases,
        (
            label = "latent extract",
            closure = latent_aug.extract,
            args = (x,),
            check_mutable = true,
            budget = 8000,
        )
    )
    push!(
        cases,
        (
            label = "latent extract_params",
            closure = latent_aug.extract_params,
            args = (x, params),
            check_mutable = true,
            budget = 8000,
        )
    )
    push!(
        cases,
        (
            label = "latent to_unconstrained",
            closure = latent_aug.to_unconstrained,
            args = ((Rt = 1.2,),),
            check_mutable = true,
            budget = 8000,
        )
    )


    # --- full dynamics ---
    dynamics_aug = build_full_dynamics(
        _mock_petri_vf!,
        latent_aug,
        layout;
        dt = 1.0,
        supersample = 2
    )
    push!(
        cases,
        (
            label = "full dynamics",
            closure = dynamics_aug,
            # augmented noise w is sized to L + S (latent + accumulator whiskers),
            # not the full state vector
            args = (x, nothing, params, 0.0, zeros(n_latent(layout) + n_signals(layout))),
            check_mutable = false,
            budget = 24_000,
        )
    )


    # --- measurement: single noise ---
    single_noise = NegBinomialNoise(0.0, 100.0)
    measurement_single = build_measurement_model(layout, single_noise, _sto(layout))
    push!(
        cases,
        (
            label = "single measurement",
            closure = measurement_single.measure,
            args = (x, SVector(50.0), NamedTuple(), 0.0, SVector(0.0, 0.0)),
            check_mutable = true,
            budget = 32,
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
    measurement_multi = build_measurement_model(layout_multi, obs_specs_multi, _sto(layout_multi))
    x_multi = [990.0, 10.0, 0.0, 100.0, 40.0, 0.0]
    push!(
        cases,
        (
            label = "multi measurement",
            closure = measurement_multi.measure,
            args = (
                x_multi,
                SVector(50.0, 20.0),
                NamedTuple(),
                0.0,
                SVector(0.0, 0.0, 0.0),
            ),
            check_mutable = true,
            budget = 600,
        )
    )

    # --- measurement: latent-aware ---
    latent_measurement = build_measurement_model(
        layout,
        (
            SignalObservationSpec(
                1,
                single_noise;
                mean_modifier = (latent, hyper, t) -> latent.Rt
            ),
        ),
        latent_aug
    )
    push!(
        cases,
        (
            label = "latent-aware measurement",
            closure = latent_measurement.measure,
            args = (x, SVector(50.0), NamedTuple(), 0.0, SVector(0.0, 0.0)),
            check_mutable = true,
            budget = 9000,
        )
    )

    # --- measurement: the declining ascertainment path as a callable-struct modifier ---
    path_measurement = build_measurement_model(
        layout,
        (
            SignalObservationSpec(
                1,
                single_noise;
                mean_modifier = AscertainmentPath(0.2, 100.0)
            ),
        ),
        latent_aug
    )
    push!(
        cases,
        (
            label = "ascertainment-path measurement",
            closure = path_measurement.measure,
            args = (
                x, SVector(50.0),
                (ascertainment = 0.005, ascertainment_decline_rate = 0.2),
                0.0, SVector(0.0, 0.0),
            ),
            check_mutable = true,
            budget = 9000,
        )
    )

    # --- measurement: aggregated (single AggregatedSignalSpec) ---
    layout_agg = StateLayout(
        (:S, :I, :R),
        (:O_child, :O_adult, :O_elderly),
        (:Rt,);
        signal_names = (:child, :adult, :elderly)
    )
    agg_spec = AggregatedSignalSpec(
        [1, 2, 3],
        NegBinomialNoise(0.0, 100.0);
        name = :total_hosp
    )
    measurement_agg = build_measurement_model(layout_agg, (agg_spec,), _sto(layout_agg))
    x_agg = [990.0, 10.0, 0.0, 5.0, 5.0, 5.0, 0.0]
    push!(
        cases,
        (
            label = "aggregated measurement",
            closure = measurement_agg.measure,
            args = (
                x_agg,
                SVector(0.0, 0.0, 0.0),
                NamedTuple(),
                0.0,
                SVector(0.0, 0.0),
            ),
            check_mutable = true,
            # AggregatedSignalSpec.signal_indices is still a Vector{Int}
            # internally; the captured tuple-of-specs is type-stable at the
            # top level, but JET may surface the runtime length-dependent
            # iteration. Set a budget once measured.
            budget = nothing,
        )
    )

    # The generic redistribution primitives `pool_redistribute!` / `pro_rata_move!` — which run per
    # particle inside a relabeling arrival's jump — are allocation-tested directly in
    # `test_redistribute_jumps.jl` (they are plain functions over NTuple indices, not closure
    # factories, so they do not fit this closure-oriented harness). The two-strain escape closures
    # that compose them live in `submodels/two_strain_escape.jl`, alongside the `rates` closures,
    # which are likewise script-level and outside the package's JET gate.

    return cases
end

# ----------------------------------------------------------------------------
# Run the suite
# ----------------------------------------------------------------------------

@testset "Closure Boxing" begin
    for case in _build_cases()
        _run_factory_checks(
            case.label,
            case.closure,
            case.args;
            check_mutable = case.check_mutable,
            budget = case.budget,
        )
    end
end
