using Test
using AlgebraicPetri: LabelledPetriNet
using Dates: Date
using ForwardDiff
using JET
using LabelledArrays: LVector
using StaticArrays: SVector
using ConfigurableEpi

# The AD-flavoured complement of test_closure_boxing.jl: the same closures under `JET.@test_opt`
# with `ForwardDiff.Dual` state, so a regression that breaks inference on Duals is caught even when
# Float64 inference is fine. The UKF hyperparameter objective differentiates through all of them.

_seed(x::Float64) = ForwardDiff.Dual(x, 1.0)
_seed_vec(xs::Vector{Float64}) = _seed.(xs)

_sto(layout) = build_stochastic_update(
    layout, Tuple(RWParamSpec(n; init = unconstrained_gaussian(n, 0.0, 1.0), sigma_rate = 0.1) for n in layout.latent_names)
)

function _build_dual_cases()
    cases = []
    push!(
        cases, (
            label = "CosineForcing (Dual phase/amp)",
            closure = CosineForcing(247.0),
            args = ((seasonal_amp = _seed(0.1), seasonal_phase = _seed(15.0), seasonal_kappa = 1.0), 100.0),
        )
    )
    push!(
        cases, (
            label = "IndoorActivityForcing (Dual kappa)",
            closure = build_seasonal_forcing(
                SeasonalityConfig(mode = "indoor_activity"), "ny", Date(2023, 9, 3);
                climatology = Dict("ny" => [1 + 0.3 * cospi(2 * (k - 0.5) / 52) for k in 1:52]),
            ),
            args = ((seasonal_amp = 0.1, seasonal_phase = 15.0, seasonal_kappa = _seed(1.0)), 100.0),
        )
    )
    push!(
        cases, (
            label = "make_lvector_constructor (Dual)",
            closure = make_lvector_constructor((:S, :I, :R)),
            args = (_seed_vec([990.0, 10.0, 0.0]),),
        )
    )

    pn = LabelledPetriNet([:S, :I], :infection => ((:S, :I) => (:I, :I)))
    rates(latent, hyper, t) = (infection = 0.001 * latent.Rt,)
    petri_vf! = build_petri_vf(pn, rates)
    push!(
        cases, (
            label = "build_petri_vf (Dual)",
            closure = petri_vf!,
            args = (LVector(S = _seed(0.0), I = _seed(0.0)), LVector(S = _seed(990.0), I = _seed(10.0)), ((;), (Rt = 1.2,)), 0.0),
        )
    )
    core_layout = StateLayout((:S, :I), (), (); signal_names = ())
    push!(
        cases, (
            label = "build_unified_vf (Dual)",
            closure = build_unified_vf(petri_vf!, core_layout),
            args = (_seed_vec([990.0, 10.0]), nothing, ((;), (Rt = 1.2,)), 0.0),
        )
    )

    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    single_noise = NegBinomialNoise(0.0, 100.0)
    x_dual = _seed_vec([990.0, 10.0, 5.0, 0.0])
    push!(
        cases, (
            label = "single measurement (Dual)",
            closure = build_measurement_model(layout, single_noise, _sto(layout)).measure,
            args = (x_dual, SVector(50.0), NamedTuple(), 0.0, SVector(0.0, 0.0)),
        )
    )
    # The declining ascertainment path with the level and rate as Duals: what the UKF objective
    # differentiates through when the rate is learned.
    push!(
        cases, (
            label = "ascertainment-path measurement (Dual)",
            closure = build_measurement_model(
                layout, (SignalObservationSpec(1, single_noise; mean_modifier = AscertainmentPath(0.2, 100.0)),), _sto(layout)
            ).measure,
            args = (
                x_dual, SVector(50.0), (ascertainment = _seed(0.005), ascertainment_decline_rate = _seed(0.2)),
                0.0, SVector(0.0, 0.0),
            ),
        )
    )
    layout_multi = StateLayout((:S, :I, :R), (:O_y1, :O_y2), (:Rt,); signal_names = (:y1, :y2))
    obs_specs_multi = (SignalObservationSpec(1, NegBinomialNoise(0.0, 100.0)), SignalObservationSpec(2, PoissonNoise()))
    push!(
        cases, (
            label = "multi measurement (Dual)",
            closure = build_measurement_model(layout_multi, obs_specs_multi, _sto(layout_multi)).measure,
            args = (_seed_vec([990.0, 10.0, 0.0, 100.0, 40.0, 0.0]), SVector(50.0, 20.0), NamedTuple(), 0.0, SVector(0.0, 0.0, 0.0)),
        )
    )
    return cases
end

@testset "Dual Number Compatibility" begin
    for case in _build_dual_cases()
        @testset "$(case.label)" begin
            JET.@test_opt target_modules = (ConfigurableEpi,) case.closure(case.args...)
        end
    end
end
