using Test
using ConfigurableEpi

const MINIMAL = Dict{String, Any}(
    "n_ahead" => 4,
    "step_days" => 7,
    "filter" => Dict("ukf" => Dict{String, Any}()),
    "hyper" => Dict("optimise" => Dict{String, Any}()),
)
config_with(changes...) = from_dict(InferenceConfig, merge(MINIMAL, Dict{String, Any}(changes...)))

@option struct ScriptRun
    data::String
    burnin_observations::Int
    inference::InferenceConfig
end

@testset "InferenceConfig" begin
    cfg = config_with()
    @test cfg.filter isa UKF && cfg.hyper isa Optimise
    @test cfg.filter isa StateFilter && cfg.hyper isa HyperMethod
    @test cfg.step_days === 7.0
    @test (cfg.n_draws, cfg.supersample, cfg.integrator) == (2000, 2, "Rk4") && isempty(cfg.learn_params) && isempty(cfg.priors)
    @test integrator_type(config_with("integrator" => "Heun").integrator) === ConfigurableEpi.Heun
    @test validate_inference_config(cfg) === cfg
    @test option_alias(cfg.filter) == "ukf" && option_alias(cfg.hyper) == "optimise"
    @test StateFilterConfig == Union{UKF, PF, EnKF} && all(T -> T <: StateFilter, (UKF, PF, EnKF))
    @test HyperMethodConfig == Union{Optimise, LiuWest, EKP} && all(T -> T <: HyperMethod, (Optimise, LiuWest, EKP))

    pf = config_with(
        "filter" => Dict("pf" => Dict("n_particles" => 500, "threads" => false)),
        "hyper" => Dict("liu_west" => Dict("discount" => 0.97, "forgetting_memory_days" => Dict("phi" => 90))),
    )
    @test pf.filter isa PF && pf.filter.n_particles == 500 && !pf.filter.threads
    @test pf.hyper.discount == 0.97 && pf.hyper.forgetting_memory_days == Dict("phi" => 90.0)
    @test pf.hyper.jitter_floor_fraction == DEFAULT_JITTER_FLOOR_FRACTION
    enkf = config_with(
        "filter" => Dict("enkf" => Dict("n_ensemble" => 200)),
        "hyper" => Dict("ekp" => Dict("n_ensemble" => 32, "iterations" => 3, "burnin_iterations" => 10)),
    )
    @test enkf.filter isa EnKF && enkf.hyper isa EKP
    @test enkf.hyper.reopt_interval == 1 && enkf.hyper.window_length === nothing

    # Strict reading: an unknown key anywhere is an error, not a silent no-op.
    @test_throws Exception config_with("nonsense" => 1)
    @test_throws Exception config_with("io" => Dict("data" => "triangle.csv"))
    @test_throws Exception config_with("filter" => Dict("ukf" => Dict("nonsense" => 1)))
    @test_throws Exception config_with("filter" => Dict("kalman" => Dict{String, Any}()))

    @testset "semantic validation" begin
        rejected(changes...) = @test_throws ArgumentError validate_inference_config(config_with(changes...))
        rejected("step_days" => 0.5)
        rejected("step_days" => 0)
        rejected("n_ahead" => 0)
        rejected("n_draws" => 0)
        rejected("supersample" => 0)
        rejected("integrator" => "Rk5")
        rejected("integrator" => "AbstractIntegrator")
        rejected("integrator" => "SeeToDee")
        rejected("filter" => Dict("ukf" => Dict("obs_jitter" => 0.0)))
        rejected("filter" => Dict("pf" => Dict("n_particles" => 0)), "hyper" => Dict("liu_west" => Dict{String, Any}()))
        rejected("hyper" => Dict("liu_west" => Dict("discount" => 1.5)))
        rejected("hyper" => Dict("liu_west" => Dict("discount" => 0.3)))
        rejected("hyper" => Dict("liu_west" => Dict("forgetting_memory_days" => Dict("phi" => 0))))
        rejected("hyper" => Dict("optimise" => Dict("window_length" => 0)))
        rejected(
            "filter" => Dict("enkf" => Dict("n_ensemble" => 1)),
            "hyper" => Dict("ekp" => Dict("n_ensemble" => 6, "iterations" => 1, "burnin_iterations" => 1)),
        )
        rejected(
            "filter" => Dict("enkf" => Dict("n_ensemble" => 10, "inflation" => 0.9)),
            "hyper" => Dict("ekp" => Dict("n_ensemble" => 6, "iterations" => 1, "burnin_iterations" => 1)),
        )
    end

    @testset "priors" begin
        defaults() = Dict(
            "R0_baseline" => PriorSpec(mean = 2.0, std = 0.5),
            "phi" => PriorSpec(mean = 100.0, std = 30.0),
            "kappa" => PriorSpec(mean = 0.8, std = 0.1, constraint = "unit_interval"),
        )
        overridden = config_with("priors" => Dict("phi" => Dict("mean" => 50.0, "std" => 10.0)))
        specs = resolve_prior_specs(overridden, defaults)
        @test specs["phi"].mean == 50.0 && specs["R0_baseline"].mean == 2.0
        priors = resolve_priors(overridden, defaults)
        @test Set(keys(priors)) == Set([:R0_baseline, :phi, :kappa])
        @test prior_name(priors.phi) == :phi && priors.kappa isa ParameterDistribution
        @test_throws ArgumentError build_prior(:x, PriorSpec(mean = 1.0, std = 1.0, constraint = "weird"))
        @test prior_upper(PriorSpec(mean = 1.0, std = 0.0)) ≈ 1.0
        @test prior_upper(PriorSpec(mean = 0.5, std = 10.0, constraint = "unit_interval")) == 1.0
        @test prior_upper(PriorSpec(mean = 0.0, std = 1.0, constraint = "unconstrained"); probability = 0.5) ≈ 0.0
        @test_throws ArgumentError prior_upper(PriorSpec(mean = 1.0, std = 1.0); probability = 1.0)
        @test prior_R_eff_bound(Dict("R0_baseline" => PriorSpec(mean = 2.0, std = 0.0)); chi_max = 1.5) ≈ 3.0
        @test prior_R_eff_bound(Dict{String, PriorSpec}(); chi_max = 1.0) == 1.0
        mktempdir() do dir
            path = joinpath(dir, "priors.toml")
            write(path, "[R0_baseline]\nmean = 2.0\nstd = 0.5\n\n[phi]\nmean = 100.0\nstd = 30.0\n")
            @test load_prior_specs(path)["phi"].std == 30.0
            @test Set(keys(load_priors(path))) == Set([:R0_baseline, :phi])
        end
    end

    @testset "from_toml round trip" begin
        mktempdir() do dir
            path = joinpath(dir, "inference.toml")
            write(
                path,
                """
                n_ahead = 4
                step_days = 7
                learn_params = ["R0_baseline"]

                [filter.enkf]
                n_ensemble = 100

                [hyper.ekp]
                n_ensemble = 32
                iterations = 3
                burnin_iterations = 10
                threads = true

                [priors.phi]
                mean = 50.0
                std = 10.0
                """,
            )
            cfg = validate_inference_config(from_toml(InferenceConfig, path))
            @test cfg.filter.n_ensemble == 100 && cfg.hyper.threads
            @test cfg.learn_params == ["R0_baseline"]
            @test cfg.priors["phi"].constraint == "positive"
        end
    end

    @testset "a run script embeds InferenceConfig" begin
        run = from_dict(ScriptRun, Dict("data" => "triangle.csv", "burnin_observations" => 12, "inference" => MINIMAL))
        @test run.inference == config_with()
    end
end
