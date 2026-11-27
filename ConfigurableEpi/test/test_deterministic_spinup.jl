using Test
using ConfigurableEpi

# S and I with a per-capita infection hazard `hyper.beta * latent.Rt`, recovery at 0.2 a day, and
# one accumulator counting infections.
function _spinup_vf!(du, u, p, t)
    hyper, latent = p
    infection = hyper.beta * latent.Rt * u[:S] * u[:I] / 1000.0
    du[:S] = -infection
    du[:I] = infection - 0.2 * u[:I]
    du[:O_I_1] = infection
    return nothing
end

@testset "deterministic_spinup" begin
    layout = StateLayout((:S, :I), (:O_I_1,), (:Rt,))
    constant(_t) = (; Rt = 1.0)
    x0 = [990.0, 10.0, 0.0]
    hyper = (; beta = 0.5)
    spinup(x, from, to; latent = constant, p = hyper, kwargs...) =
        deterministic_spinup(_spinup_vf!, layout, x, p, latent, from, to; kwargs...)

    @testset "pure decay matches the closed form, and the accumulators are zeroed" begin
        x = spinup(x0, -30.0, 0.0; p = (; beta = 0.0), supersample = 4)
        @test x[1] == 990.0
        @test x[2] ≈ 10.0 * exp(-0.2 * 30) rtol = 1.0e-6
        @test x[3] == 0.0
    end

    @testset "the integrator steps the substeps" begin
        x = spinup(x0, -30.0, 0.0; p = (; beta = 0.0), supersample = 4, integrator = ConfigurableEpi.ForwardEuler)
        @test x[2] ≈ 10.0 * (1 - 0.2 * 0.25)^120 rtol = 1.0e-12
        @test !isapprox(x[2], spinup(x0, -30.0, 0.0; p = (; beta = 0.0), supersample = 4)[2]; rtol = 1.0e-6)
    end

    @testset "infections leave S, and keep_accumulators holds the last step's flow" begin
        to(t; kwargs...) = spinup(x0, -20.0, t; kwargs...)
        before, after = to(-1.0), to(0.0)
        @test after[1] < before[1] < 990.0
        @test after[3] == 0.0
        kept = to(0.0; keep_accumulators = true)
        @test kept[1:2] == after[1:2]
        @test kept[3] ≈ before[1] - after[1] rtol = 1.0e-10

        # A longer final step accumulates over the whole of it.
        week = spinup(to(-7.0), -7.0, 0.0; dt = 7.0, supersample = 14, keep_accumulators = true)
        @test week[3] ≈ to(-7.0)[1] - week[1] rtol = 1.0e-10
        @test week[1] ≈ after[1] rtol = 1.0e-6
    end

    @testset "the latents follow the given path" begin
        suppressed(t) = (; Rt = t < -10 ? 0.5 : 1.0)
        @test spinup(x0, -20.0, 0.0; latent = suppressed)[1] > spinup(x0, -20.0, 0.0)[1]
    end

    @testset "an empty compartment stays empty" begin
        @test spinup([1000.0, -5.0, 0.0], -30.0, 0.0) == [1000.0, 0.0, 0.0]
    end

    @testset "the span must be a whole number of forward steps" begin
        @test spinup(x0, -5.0, -5.0) == x0
        @test_throws ArgumentError spinup(x0, 0.0, -1.0)
        @test_throws ArgumentError spinup(x0, -2.5, 0.0)
        @test_throws ArgumentError spinup(x0, -2.0, 0.0; supersample = 0)
    end
end
