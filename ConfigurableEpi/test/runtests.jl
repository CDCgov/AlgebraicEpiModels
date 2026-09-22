using ConfigurableEpi, Test

# `Pkg.test(; test_args = ["test_foo.jl"])` runs a subset.
const TEST_FILES = [
    "aqua.jl",
    "test_param_spec.jl",
    "test_ekp_parameter_distributions.jl",
    "test_config.jl",
    "test_seasonality.jl",
    "test_ascertainment.jl",
    "test_day_of_week.jl",
    "test_ascertainment_trend.jl",
    "test_state_layout.jl",
    "test_latent_dynamics.jl",
    "test_measurement_model.jl",
    "test_observation_logpdf.jl",
    "test_core_dynamics.jl",
    "test_full_dynamics.jl",
    "test_pf_builders.jl",
    "test_hyperparam_learning.jl",
    "test_arrival_process.jl",
    "test_redistribute_jumps.jl",
    "test_backtest.jl",
    "test_augmented_enkf.jl",
    "test_build_inference.jl",
    "test_data_linkage.jl",
    "test_closure_boxing.jl",
    "test_dual_numbers.jl",
]

@testset "ConfigurableEpi" begin
    for file in (isempty(ARGS) ? TEST_FILES : ARGS)
        @testset "$file" begin
            include(file)
        end
    end
end
