using Test, ConfigurableEpi, Random, Statistics, LinearAlgebra
import ConfigurableEpi as CE
import ForwardDiff
import LowLevelParticleFilters as LLPF

const RTNB_WEIGHT = 13.96315018415381
const RTNB_PRIORS = (
    Rt_sigma_stat = positive_gaussian(:Rt_sigma_stat, 0.05, 0.02),
    R0_baseline = positive_gaussian(:R0_baseline, 1.2, 0.3),
    ascertainment_trend_wander = positive_gaussian(:ascertainment_trend_wander, 0.1, 0.05),
    phi = positive_gaussian(:phi, 140.0, 50.0),
)

function rtnb_test_model(; joint_prior = RtNBVariancePrior(transmission_weight = RTNB_WEIGHT), kwargs...)
    layout = StateLayout((:X,), (:reports,), ())
    vf!(du, x, p, t) = (fill!(du, 0); nothing)
    return EpiModel(;
        vectorfield! = vf!, layout, stochastic = build_stochastic_update(layout, ()),
        observation = (WeightedStateSpec([:X], [1.0], NegBinomialNoise(phi = HyperPhi()); name = :reports),),
        hyperparams = (Rt_sigma_stat = 0.05, R0_baseline = 1.2, ascertainment_trend_wander = 0.1, phi = 250.0),
        priors = RTNB_PRIORS, initial_state = hp -> [100hp.R0_baseline, 0.0], joint_prior, kwargs...,
    )
end

# `(log T, logit f)` of an unconstrained vector, with the remaining coordinates after them.
rtnb_chart(p, b, u) = (i = findfirst(==(p.rt_scale), b.names); j = findfirst(==(p.dispersion), b.names); CE._rtnb_summary(p, u[i], u[j]))

@testset "independent priors are the default coupling" begin
    b = ParameterPriorBundle(RTNB_PRIORS)
    @test b.joint === Independent()
    m, V = prior_moments(b)
    @test m == [prior_unconstrained_mean(p) for p in RTNB_PRIORS]
    @test V == Diagonal([prior_unconstrained_variance(p) for p in RTNB_PRIORS])
    u = [0.1, -0.2, 0.3, 4.0]
    @test prior_logpdf(b, u) ≈ sum(sum(CE.logpdf(p, [x])) for (p, x) in zip(RTNB_PRIORS, u))
    draws = reduce(hcat, [sample_prior(Xoshiro(3 + k), b) for k in 1:4000])
    @test vec(mean(draws; dims = 2)) ≈ m atol = 0.05
    @test rtnb_test_model(; joint_prior = Independent()).joint_prior === Independent()
end

@testset "Rt/NB prior geometry and elicited distribution" begin
    p = RtNBVariancePrior(transmission_weight = RTNB_WEIGHT)
    model = rtnb_test_model()
    b = ParameterPriorBundle(model)
    @test b.joint === p
    u = unconstrained_values(b, model.hyperparams)
    s, r = rtnb_chart(p, b, u)
    @test collect(CE._rtnb_inverse(p, s, r)) ≈ u[[1, 4]]
    expected = CE.logpdf(CE.Normal(2p.log_sd_mean, 2p.log_sd_std), s) + CE.logpdf(CE.Normal(p.logit_rt_mean, p.logit_rt_std), r) +
        log(2.0) + sum(sum(CE.logpdf(b.scalars[n], [u[k]])) for (k, n) in ((2, :R0_baseline), (3, :ascertainment_trend_wander)))
    @test prior_logpdf(b, u) ≈ expected
    @test all(isfinite, ForwardDiff.gradient(x -> prior_logpdf(b, x), u))
    @test prior_logpdf(b, [NaN, 0.0, 0.0, 0.0]) == -Inf
    for (extreme_s, extreme_r) in ((-100.0, 1000.0), (100.0, -1000.0))
        x, y = CE._rtnb_inverse(p, extreme_s, extreme_r)
        @test isfinite(x) && isfinite(y)
        @test collect(CE._rtnb_summary(p, x, y)) ≈ [extreme_s, extreme_r]
    end
    @test_throws "requires learned" ParameterPriorBundle((Rt_sigma_stat = b.scalars.Rt_sigma_stat,); joint = p)
    @test_throws "log-positive" ParameterPriorBundle(merge(b.scalars, (phi = unconstrained_gaussian(:phi, 140.0, 50.0),)); joint = p)
    @test_throws "positive" RtNBVariancePrior(transmission_weight = -1.0)
    @test_throws "finite" RtNBVariancePrior(transmission_weight = Inf)

    rng = Xoshiro(927)
    draws = reduce(hcat, [sample_prior(rng, b) for _ in 1:30000])
    qs = reduce(hcat, [collect(rtnb_chart(p, b, u)) for u in eachcol(draws)])
    @test mean(qs[1, :]) ≈ 2p.log_sd_mean atol = 0.02
    @test std(qs[1, :]) ≈ 2p.log_sd_std atol = 0.015
    @test mean(qs[2, :]) ≈ p.logit_rt_mean atol = 0.02
    @test std(qs[2, :]) ≈ p.logit_rt_std atol = 0.015
    @test abs(cor(qs[1, :], qs[2, :])) < 0.02
    @test mean(qs[2, :] .> log(4.0)) ≈ 0.7913 atol = 0.01
    @test quantile(exp.(qs[1, :] ./ 2), [0.05, 0.5, 0.95]) ≈ [0.1, 0.2, 0.4] atol = 0.01
    m, V = prior_moments(b)
    @test vec(mean(draws; dims = 2)) ≈ m atol = 0.025
    @test cov(draws; dims = 2) ≈ V atol = 0.03
    for n in (:R0_baseline, :ascertainment_trend_wander)
        k = findfirst(==(n), b.names)
        @test m[k] == prior_unconstrained_mean(b.scalars[n])
        @test V[k, k] == prior_unconstrained_variance(b.scalars[n])
        @test mean(draws[k, :]) ≈ m[k] atol = 0.01
    end
end

@testset "Rt/NB prior under another model's parameter names" begin
    p = RtNBVariancePrior(transmission_weight = RTNB_WEIGHT, rt_scale = "sigma", dispersion = :k)
    priors = (beta = positive_gaussian(:beta, 1.2, 0.3), k = positive_gaussian(:k, 140.0, 50.0), sigma = positive_gaussian(:sigma, 0.05, 0.02))
    b = ParameterPriorBundle(priors; joint = p)
    @test CE.joint_prior_parameters(p) == (:sigma, :k)
    @test isfinite(prior_logpdf(b, [0.1, 4.9, -3.0]))
    @test prior_moments(b)[1][1] == prior_unconstrained_mean(priors.beta)
    @test_throws "requires learned sigma and k" ParameterPriorBundle((beta = priors.beta, k = priors.k); joint = p)
    @test_throws "distinct" RtNBVariancePrior(transmission_weight = 1.0, rt_scale = :phi)
end

@testset "with_joint_prior couples and uncouples a model's priors" begin
    p = RtNBVariancePrior(transmission_weight = RTNB_WEIGHT)
    plain = rtnb_test_model(; joint_prior = Independent())
    coupled = with_joint_prior(plain, p)
    @test coupled.joint_prior === p
    @test ParameterPriorBundle(coupled).joint === p
    @test with_joint_prior(coupled, Independent()).joint_prior === Independent()
    @test_throws "requires learned" with_joint_prior(plain, RtNBVariancePrior(transmission_weight = 1.0, rt_scale = :beta))
end

@testset "Liu-West starts from and keeps the joint prior's moments" begin
    model = rtnb_test_model()
    lw(m = model; n = 100, hyper = LiuWest()) =
        build_inference(PF(n_particles = n, threads = false), hyper, m; dt = 1.0, n_ahead = 2, n_draws = 30, rng = Xoshiro(72))
    engine = lw(; n = 5000)
    U = reduce(hcat, [x[3:6] for x in LLPF.particles(engine.filter)])
    m, V = prior_moments(engine.core.bundle)
    @test vec(mean(U; dims = 2)) ≈ m atol = 0.05
    @test cov(U; dims = 2) ≈ V atol = 0.09
    # Each state is built from its own R0 draw, not a shared plug-in value.
    X = [x[1] for x in LLPF.particles(engine.filter)]
    @test cor(X, exp.(U[2, :])) > 0.95
    @test abs(mean(X .- 100exp.(U[2, :]))) < 0.3

    result = fit_forecast!(lw(), Union{Missing, Float64}[115, missing, 135, 120], 1)
    ratio = result.summary[(result.summary.parameter .== "phi") .& (result.summary.statistic .== "sd_ratio"), :value]
    @test only(ratio) > 0

    @test_throws "forgetting" lw(; hyper = LiuWest(forgetting_memory_days = Dict("phi" => 100.0)))
    @test lw(; hyper = LiuWest(forgetting_memory_days = Dict("R0_baseline" => 100.0))) isa PFLiuWestEngine
    @test lw(rtnb_test_model(; joint_prior = Independent(), initial_learned_variance = (phi = 0.1,))) isa PFLiuWestEngine
    @test_throws "initial_learned_variance" lw(rtnb_test_model(; initial_learned_variance = (R0_baseline = 0.1,)))
    @test_throws "joint prior" build_inference(
        EnKF(n_ensemble = 40), EKP(n_ensemble = 6, iterations = 1, burnin_iterations = 1), model; dt = 1.0, n_ahead = 2,
    )
end
