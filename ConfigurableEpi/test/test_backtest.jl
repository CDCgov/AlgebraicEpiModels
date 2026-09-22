using Test
using ConfigurableEpi
using StaticArrays: SVector
using LinearAlgebra: Diagonal, I, cholesky!
using Distributions: MvNormal
using Statistics: var, std
using DataFramesMeta: DataFrame, nrow, names
using Dates: Date, Day
using DBInterface: close, connect, execute
using DuckDB: DB
using PositiveFactorizations: Positive
using LowLevelParticleFilters: AdvancedParticleFilter, UnscentedKalmanFilter,
    particles, TrivialParams
import Random

@testset "Routine-compatible forecast sample output" begin
    target_dates = Date.(["2024-02-06", "2024-02-07"])
    samples = [11.0 12.0; 21.0 22.0]
    rows = forecast_sample_rows(
        samples, target_dates;
        geo_value = "US",
        disease = "covid",
        variable = "observed_ed_visits",
        resolution = "daily",
    )
    @test Symbol.(names(rows)) == collect(FORECAST_SAMPLE_COLUMNS)
    @test rows.date == repeat(target_dates, 2)
    @test rows[!, Symbol(".value")] == [11.0, 21.0, 12.0, 22.0]
    @test rows[!, Symbol(".draw")] == Int32[1, 1, 2, 2]
    @test eltype(rows.date) == Date
    @test eltype(rows[!, Symbol(".draw")]) == Int32

    multi_rows = forecast_sample_rows(
        cat(samples, samples .+ 100.0; dims = 3), target_dates;
        geo_values = ["CA", "TX"],
        diseases = ["covid", "covid"],
        variables = ["observed_ed_visits", "observed_ed_visits"],
        resolution = "daily",
    )
    @test nrow(multi_rows) == 8
    @test multi_rows.geo_value == [fill("CA", 4); fill("TX", 4)]
    @test multi_rows[!, Symbol(".value")] ==
        [11.0, 21.0, 12.0, 22.0, 111.0, 121.0, 112.0, 122.0]

    origin_1 = Date("2024-02-03")
    origin_2 = Date("2024-02-10")
    batch_1 = forecast_sample_rows(
        samples, target_dates;
        geo_value = "US", disease = "covid", variable = "observed_ed_visits",
        resolution = "daily", metadata = (; origin = origin_1),
    )
    batch_2 = forecast_sample_rows(
        samples .+ 100.0, target_dates .+ Day(7);
        geo_value = "US", disease = "covid", variable = "observed_ed_visits",
        resolution = "daily", metadata = (; origin = origin_2),
    )
    @test names(batch_1)[1:7] == collect(string.(FORECAST_SAMPLE_COLUMNS))
    @test batch_1.origin == fill(origin_1, 4)

    mktempdir() do dir
        path = joinpath(dir, "samples.parquet")
        @test write_forecast_samples(path, (batch_1, batch_2)) == path
        @test isfile(path)
        con = connect(DB, ":memory:")
        written = try
            DataFrame(
                execute(
                    con,
                    "SELECT * FROM read_parquet('$(replace(path, "'" => "''"))') " *
                        "ORDER BY origin, \".draw\", date",
                )
            )
        finally
            close(con)
        end
        @test nrow(written) == 8
        @test eltype(written.date) == Date
        @test eltype(written[!, Symbol(".draw")]) == Int32
        @test unique(written.origin) == [origin_1, origin_2]
    end

    @test_throws DimensionMismatch forecast_sample_rows(
        samples, target_dates[1:1];
        geo_value = "US", disease = "covid", variable = "observed_ed_visits",
        resolution = "daily",
    )
    @test_throws ArgumentError forecast_sample_rows(
        samples, target_dates;
        geo_value = "US", disease = "covid", variable = "observed_ed_visits",
        resolution = "daily", metadata = (; disease = "flu"),
    )
end

# Shared mock SI model whose transmission scales with R0_baseline (as in test_pf_builders.jl).
function mock_bt_vf!(du, u, p, t)
    hyper, latent = p
    beta = hyper.R0_baseline * latent.Rt * 0.3 / 1000.0
    du[:S] = -beta * u[:S] * u[:I]
    du[:I] = beta * u[:S] * u[:I] - 0.2 * u[:I]
    du[:O_I_1] = hyper.obs_scale * u[:I]
    return nothing
end

@testset "Backtest building blocks" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    Rt_spec = RWParamSpec(:Rt; init = positive_gaussian(:Rt, 1.0, 0.1), sigma_rate = FixedParam(:sigma_Rt, 0.05))
    ld = build_stochastic_update(layout, (Rt_spec,))
    hyper = (R0_baseline = 2.0, obs_scale = 0.25)
    obs_specs = (SignalObservationSpec(1, NegBinomialNoise(phi = 100.0); mean_modifier = 1.0),)

    @testset "asof_series — no-leakage (as_of <= r_k, latest per date)" begin
        df = DataFrame(
            date = Date.(["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08", "2024-01-15"]),
            as_of = Date.(["2024-01-07", "2024-01-14", "2024-01-14", "2024-01-21", "2024-01-21"]),
            value = [10.0, 12.0, 20.0, 22.0, 30.0],
        )
        a = asof_series(df; r_k = Date("2024-01-14"))
        @test a.date == Date.(["2024-01-01", "2024-01-08"])  # 01-15 excluded (as_of 01-21 > r_k)
        @test a.value == [12.0, 20.0]                          # latest as_of <= r_k per date
        @test isempty(asof_series(df; r_k = Date("2024-01-01")))  # before any as_of
    end

    @testset "asof_series — group_cols keeps every series" begin
        # Two locations reporting the same reference dates, with staggered revisions.
        df = DataFrame(
            date = Date.(
                [
                    "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08",
                ]
            ),
            location = ["ny", "ny", "ca", "ny", "ca"],
            as_of = Date.(
                [
                    "2024-01-07", "2024-01-14", "2024-01-07", "2024-01-14", "2024-01-14",
                ]
            ),
            value = [10.0, 12.0, 90.0, 20.0, 80.0],
        )
        grouped = asof_series(df; r_k = Date("2024-01-14"), group_cols = (:location,))
        # One row per (date, location) — four cells, not two.
        @test nrow(grouped) == 4
        @test Set(zip(grouped.location, grouped.date)) == Set(
            [
                ("ny", Date("2024-01-01")), ("ca", Date("2024-01-01")),
                ("ny", Date("2024-01-08")), ("ca", Date("2024-01-08")),
            ]
        )
        # The latest issue is taken WITHIN each location: ny's 01-01 revision is picked up.
        pick(loc, d) = only(
            grouped[(grouped.location .== loc) .& (grouped.date .== d), :value]
        )
        @test pick("ny", Date("2024-01-01")) == 12.0
        @test pick("ca", Date("2024-01-01")) == 90.0
        # Date-major ordering, so a caller can widen per-date blocks directly.
        @test issorted(grouped.date)

        # Without `group_cols` the grouping key is `date` alone, so all but one location per date is
        # silently dropped. This is the failure the keyword exists to prevent.
        @test nrow(asof_series(df; r_k = Date("2024-01-14"))) == 2

        # The no-leakage filter is unaffected by grouping.
        @test isempty(
            asof_series(df; r_k = Date("2024-01-01"), group_cols = (:location,))
        )
    end

    @testset "forecast_quantiles — 3-D multi-observation sibling" begin
        # samples[h, j, o]: 3 horizons, 200 draws, 2 observations on deliberately different scales.
        Random.seed!(11)
        draws = 200
        samples = Array{Float64, 3}(undef, 3, draws, 2)
        for h in 1:3, j in 1:draws
            samples[h, j, 1] = 100.0 * h + randn()
            samples[h, j, 2] = 1000.0 * h + randn()
        end
        qs = (0.1, 0.5, 0.9)
        q3 = forecast_quantiles(samples; qs = qs)
        @test size(q3) == (3, 2, length(qs))
        # Monotone in the quantile axis for every (horizon, observation).
        @test all(issorted(q3[h, o, :]) for h in 1:3, o in 1:2)
        # Each observation keeps its own scale — the axes are not transposed.
        @test all(isapprox(q3[h, 1, 2], 100.0 * h; atol = 1.0) for h in 1:3)
        @test all(isapprox(q3[h, 2, 2], 1000.0 * h; atol = 1.0) for h in 1:3)
        # Slicing one observation reproduces the matrix method exactly.
        @test q3[:, 1, :] ≈ forecast_quantiles(view(samples, :, :, 1); qs = qs)
    end

    @testset "backtest_forecast_rows — location-aware sibling" begin
        # qarr[h, o, k]: 2 horizons × 2 locations × 3 quantiles.
        qarr = reshape(Float64.(1:12), 2, 2, 3)
        tds = Date.(["2024-02-01", "2024-02-08"])
        truth = DataFrame(
            location = ["ny", "ny", "ca"],
            date = Date.(["2024-02-01", "2024-02-08", "2024-02-01"]),
            counts = [100.0, 200.0, 900.0],
        )
        rows = backtest_forecast_rows(
            qarr, Date("2024-01-25"), tds, truth;
            locations = ["ny", "ca"], qs = (0.25, 0.5, 0.75), model_id = "m",
        )
        @test names(rows) == [
            "location", "origin", "horizon", "target_date", "quantile", "value",
            "observed", "model_id",
        ]
        @test nrow(rows) == 2 * 2 * 3
        # One row per (location, origin, horizon, quantile).
        @test length(unique(zip(rows.location, rows.horizon, rows.quantile))) == nrow(rows)
        # Values are indexed [horizon, observation, quantile], with `locations[o]` naming o.
        cell(loc, h, q) = only(
            rows[
                (rows.location .== loc) .& (rows.horizon .== h) .& (rows.quantile .== q),
                :value,
            ]
        )
        @test cell("ny", 1, 0.25) == qarr[1, 1, 1]
        @test cell("ca", 2, 0.75) == qarr[2, 2, 3]
        # The truth join is keyed on (location, date), not date alone.
        obs(loc, h) = only(
            unique(rows[(rows.location .== loc) .& (rows.horizon .== h), :observed])
        )
        @test obs("ny", 1) == 100.0
        @test obs("ny", 2) == 200.0
        @test obs("ca", 1) == 900.0
        @test ismissing(obs("ca", 2))   # ca has no truth at the second target date
        # Location order is the caller's, not the truth frame's or alphabetical.
        @test unique(rows.location) == ["ny", "ca"]
        @test_throws DimensionMismatch backtest_forecast_rows(
            qarr, Date("2024-01-25"), tds, truth;
            locations = ["ny"], qs = (0.25, 0.5, 0.75), model_id = "m",
        )
    end

    @testset "backtest_forecast_rows — schema + observed join" begin
        qmat = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2 horizons × 3 quantiles
        truth = DataFrame(date = Date.(["2024-02-01", "2024-02-08"]), counts = [100.0, 200.0])
        tds = Date.(["2024-02-01", "2024-02-08"])
        rows = backtest_forecast_rows(qmat, Date("2024-01-25"), tds, truth; qs = (0.25, 0.5, 0.75), model_id = "m")
        @test names(rows) == ["origin", "horizon", "target_date", "quantile", "value", "observed", "model_id"]
        @test nrow(rows) == 6
        @test rows[(rows.horizon .== 1) .& (rows.quantile .== 0.5), :value][1] == 2.0
        @test rows[rows.horizon .== 1, :observed][1] == 100.0
        @test rows[rows.horizon .== 2, :observed][1] == 200.0
        rows2 = backtest_forecast_rows(
            qmat, Date("2024-01-25"), Date.(["2024-02-01", "2099-01-01"]), truth;
            qs = (0.25, 0.5, 0.75), model_id = "m",
        )
        @test ismissing(rows2[rows2.horizon .== 2, :observed][1])  # no truth at that date
    end

    @testset "optimize_hyperparams — recovers a known optimum" begin
        bundle = ParameterPriorBundle((R0_baseline = positive_gaussian(:R0_baseline, 2.0, 1.0),))
        target_unc = unconstrained_values(bundle, (R0_baseline = 2.5,))
        neg_ll(θ, _) = sum((θ .- target_unc) .^ 2)  # minimized at R0_baseline = 2.5
        res = optimize_hyperparams(neg_ll, (R0_baseline = 1.0,), bundle; options = (maxiters = 300,))
        @test res.θ.R0_baseline ≈ 2.5 atol = 1.0e-2
    end

    @testset "forecast_ensemble — PF path (no correct!, uncertainty grows)" begin
        Random.seed!(20260620)
        dynamics = build_full_dynamics(mock_bt_vf!, ld, layout; supersample = 2)
        measure, _ny, nv = build_measurement_model(layout, obs_specs, ld)
        pf_dyn = build_pf_dynamics(dynamics, layout)
        pf_meas = build_pf_measurement(layout, obs_specs, ld)
        g = build_measurement_logpdf(layout, obs_specs, ld)
        x0 = [900.0, 50.0, 0.0, ld.to_unconstrained((Rt = 1.0,))[1]]
        P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.04]))
        pf = AdvancedParticleFilter(500, pf_dyn, pf_meas, g, nothing, MvNormal(x0, P0); p = hyper, ny = 1, nu = 0)
        cloud = collect(particles(pf))

        s1, lat1 = forecast_ensemble(pf, cloud, hyper; n_ahead = 4, t0 = 0.0, dt = 1.0)
        @test size(s1) == (4, length(cloud))

        # `n_obs` is opt-in and the single-signal branch is unchanged: with the same seed the
        # `n_obs = 1` call reproduces the default one exactly, which is what keeps the existing
        # UKF/PF forecasts byte-identical (the two branches consume the RNG identically).
        Random.seed!(20260620)
        pf_a = deepcopy(pf)
        s_default, _ = forecast_ensemble(pf_a, cloud, hyper; n_ahead = 4, t0 = 0.0, dt = 1.0)
        Random.seed!(20260620)
        pf_b = deepcopy(pf)
        s_explicit, _ = forecast_ensemble(
            pf_b, cloud, hyper; n_ahead = 4, t0 = 0.0, dt = 1.0, n_obs = 1
        )
        @test s_default == s_explicit
        @test s_default isa Matrix
        @test_throws ArgumentError forecast_ensemble(
            pf, cloud, hyper; n_ahead = 2, t0 = 0.0, dt = 1.0, n_obs = 0
        )

        @test all(s1 .>= 0.0)
        @test var(s1[4, :]) > var(s1[1, :])  # predictive spread widens with horizon
        @test size(lat1) == (4, length(cloud), 0)  # latent capture is opt-in
    end

    @testset "forecast_ensemble — latent capture: a shorter memory forecasts tighter" begin
        # The forecast-variance lever. Under the STATIONARY parameterisation the horizon-h
        # conditional variance of the log-scale Rt correction is a convex combination,
        #
        #     Var(u_h) = rho^{2h} * v0  +  sigma_stat^2 * (1 - rho^{2h}),   rho = exp(-dt/tau)
        #
        # so it is bounded by `max(v0, sigma_stat^2)` for ANY memory. That bound is the property
        # the reparameterisation buys: the old `(rho, sigma_innovation)` form had the asymptote
        # `sigma^2/(1 - rho^2)`, which is 5.26x sigma^2 at rho = 0.9 and unbounded as rho -> 1.
        #
        # Here `v0 = 0.04` (sd 0.2) exceeds `sigma_stat^2 = 0.0025`, so the spread CONTRACTS toward
        # the stationary sd — and a longer memory contracts more slowly, hence stays wider at h=4.
        # `tau` is in the filter's own time unit, and this test runs at the default `dt = 1`.
        function latent_spread(tau)
            specs = (
                AR1ParamSpec(
                    :Rt; init = positive_gaussian(:Rt, 1.0, 0.1),
                    mu = 1.0, tau = tau, sigma = 0.05,
                ),
            )
            sto = build_stochastic_update(layout, specs)
            dynamics = build_full_dynamics(mock_bt_vf!, sto, layout; supersample = 2)
            measure, _ny, nv = build_measurement_model(layout, obs_specs, sto)
            pf = AdvancedParticleFilter(
                1000,
                build_pf_dynamics(dynamics, layout),
                build_pf_measurement(layout, obs_specs, sto),
                build_measurement_logpdf(layout, obs_specs, sto),
                nothing,
                MvNormal(
                    [900.0, 50.0, 0.0, sto.to_unconstrained((Rt = 1.0,))[1]],
                    Matrix(Diagonal([1.0, 1.0, 1.0, 0.04])),
                );
                p = hyper, ny = 1, nu = 0, rng = Random.MersenneTwister(4242),
            )
            _s, lat = forecast_ensemble(
                pf, collect(particles(pf)), hyper;
                n_ahead = 4, t0 = 0.0, dt = 1.0, latent_range = layout.latent_range,
            )
            return lat
        end

        # tau chosen so rho = exp(-1/tau) reproduces the persistences this test used to sweep.
        lat_sticky = latent_spread(-1 / log(0.9))    # rho = 0.9 at dt = 1
        lat_short = latent_spread(-1 / log(0.2))     # rho = 0.2 at dt = 1
        @test size(lat_sticky) == (4, 1000, 1)   # (horizon, draw, latent)
        @test std(view(lat_short, 4, :, 1)) < std(view(lat_sticky, 4, :, 1))

        # The convex-combination bound: neither arm may exceed the wider of the initial spread and
        # the stationary sd. Under the old parameterisation the sticky arm was NOT so bounded.
        bound = max(sqrt(0.04), 0.05)
        @test std(view(lat_sticky, 4, :, 1)) <= 1.05 * bound
        @test std(view(lat_short, 4, :, 1)) <= 1.05 * bound
        # ... and the short-memory arm has essentially relaxed to the stationary sd by h = 4.
        @test std(view(lat_short, 4, :, 1)) ≈ 0.05 rtol = 0.15

        # The summary rows the backtest writes to <stem>_hyperparameters.csv.
        summary = DataFrame(parameter = String[], statistic = String[], value = Float64[])
        log_sd = [std(view(lat_short, h, :, 1)) for h in 1:4, _l in 1:1]
        append_latent_spread!(summary, layout.latent_names, log_sd)
        @test nrow(summary) == 8                                   # 4 horizons × {log_sd, factor}
        @test Set(summary.parameter) == Set(["Rt"])
        @test "fc_log_sd_h4" in summary.statistic
        factor = summary[summary.statistic .== "fc_factor_h4", :value][1]
        sd = summary[summary.statistic .== "fc_log_sd_h4", :value][1]
        @test factor ≈ exp(sd)
    end

    @testset "append_latent_audit! — recovers a known AR(1)'s own statistics" begin
        # A generated path with known (rho, sigma) is the only way to check the audit reports the
        # process rather than an artefact of the estimator. Stationary sd = sigma/sqrt(1-rho^2),
        # lag-1 autocorrelation = rho, and the implied correlation time = -dt/log(rho).
        rho, sigma, dt = 0.7, 0.1, 7.0
        rng = Random.MersenneTwister(20260828)
        n = 20_000
        path = Vector{Float64}(undef, n)
        path[1] = sigma / sqrt(1 - rho^2) * randn(rng)     # start stationary, so there is no burn-in
        for t in 2:n
            path[t] = rho * path[t - 1] + sigma * randn(rng)
        end
        # The latent sits at slot 4 of a 4-element state, as it does in the real layouts.
        xt = [[0.0, 0.0, 0.0, u] for u in path]

        summary = DataFrame(parameter = String[], statistic = String[], value = Float64[])
        append_latent_audit!(summary, [:Rt], xt, 4:4; dt = dt)
        stat(name) = summary[summary.statistic .== name, :value][1]

        @test nrow(summary) == 4
        @test Set(summary.parameter) == Set(["Rt"])
        @test stat("is_acf1") ≈ rho rtol = 0.05
        @test stat("is_sd") ≈ sigma / sqrt(1 - rho^2) rtol = 0.05
        @test stat("is_tau_days") ≈ -dt / log(rho) rtol = 0.1
        @test abs(stat("is_drift")) < 0.02                # mean-reverting ⇒ no systematic drift
    end

    @testset "append_latent_audit! — degenerate inputs report NaN, not a number" begin
        summary = DataFrame(parameter = String[], statistic = String[], value = Float64[])

        # Too short to estimate anything.
        append_latent_audit!(summary, [:Rt], [[0.0, 1.0], [0.0, 2.0]], 2:2; dt = 7.0)
        @test nrow(summary) == 4
        @test all(isnan, summary.value)

        # Alternating path: acf1 is negative, so there is no correlation time to report. Reporting
        # NaN rather than 0 keeps "no memory measured" distinct from "memory measured as zero".
        empty!(summary)
        alternating = [[0.0, iseven(t) ? 1.0 : -1.0] for t in 1:200]
        append_latent_audit!(summary, [:Rt], alternating, 2:2; dt = 7.0)
        stat(name) = summary[summary.statistic .== name, :value][1]
        @test stat("is_acf1") < 0
        @test isnan(stat("is_tau_days"))
        @test stat("is_sd") ≈ 1.0 rtol = 1.0e-6

        # A persistent level offset is what the drift statistic exists to catch.
        empty!(summary)
        stepped = [[0.0, t <= 100 ? 0.0 : 0.5] for t in 1:200]
        append_latent_audit!(summary, [:Rt], stepped, 2:2; dt = 7.0)
        @test stat("is_drift") ≈ 0.5 rtol = 1.0e-6
    end

    @testset "forecast_ensemble — UKF path + forecast_quantiles" begin
        dynamics = build_full_dynamics(mock_bt_vf!, ld, layout; supersample = 2)
        measure, ny, nv = build_measurement_model(layout, obs_specs, ld)
        R1 = Matrix(build_R1(layout))
        R2 = Matrix{Float64}(I, nv, nv)
        x0 = [900.0, 50.0, 0.0, ld.to_unconstrained((Rt = 1.0,))[1]]
        P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.04]))
        ukf = UnscentedKalmanFilter{false, false, true, true}(
            dynamics, measure, R1, R2, MvNormal(x0, P0);
            p = hyper, ny = ny, nu = 0, weight_params = TrivialParams(),
            cholesky! = R -> cholesky!(Positive, Matrix(R)),
        )
        rng = Random.MersenneTwister(7)
        init = [rand(rng, MvNormal(x0, P0)) for _ in 1:300]

        s, _ = forecast_ensemble(ukf, init, hyper; n_ahead = 4, t0 = 0.0, dt = 1.0)
        @test size(s) == (4, 300)
        @test all(s .>= 0.0)

        qmat = forecast_quantiles(s)
        @test size(qmat) == (4, length(DEFAULT_QS))
        @test all(issorted(qmat[h, :]) for h in 1:4)  # quantiles monotone per horizon
    end

    @testset "forecast_states — UKF analytic predict! roll (deterministic, deepcopy)" begin
        dynamics = build_full_dynamics(mock_bt_vf!, ld, layout; supersample = 2)
        measure, ny, nv = build_measurement_model(layout, obs_specs, ld)
        R1 = Matrix(build_R1(layout))
        R2 = Matrix{Float64}(I, nv, nv)
        x0 = [900.0, 50.0, 0.0, ld.to_unconstrained((Rt = 1.0,))[1]]
        P0 = Matrix(Diagonal([1.0, 1.0, 1.0, 0.04]))
        ukf = UnscentedKalmanFilter{false, false, true, true}(
            dynamics, measure, R1, R2, MvNormal(x0, P0);
            p = hyper, ny = ny, nu = 0, weight_params = TrivialParams(),
            cholesky! = R -> cholesky!(Positive, Matrix(R)),
        )
        x_before = copy(ukf.x)
        means, covs = forecast_states(ukf, x0, P0; n_ahead = 4, t0 = 0.0, dt = 1.0, p = hyper)
        @test length(means) == 4 && length(covs) == 4
        @test all(length(m) == length(x0) for m in means)
        @test all(size(c) == (length(x0), length(x0)) for c in covs)
        # analytic ⇒ deterministic: no rng, so a second call is bit-identical
        means2, covs2 = forecast_states(ukf, x0, P0; n_ahead = 4, t0 = 0.0, dt = 1.0, p = hyper)
        @test means == means2 && covs == covs2
        # operates on a deepcopy — the caller's filter is left untouched
        @test ukf.x == x_before
        # predictive state covariance grows with horizon (process noise accumulates)
        tr(c) = sum(c[i, i] for i in 1:size(c, 1))
        @test tr(covs[4]) > tr(covs[1])
    end
end
