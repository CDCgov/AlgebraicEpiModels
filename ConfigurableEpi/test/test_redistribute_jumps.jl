using Test
using ConfigurableEpi
import Random

# The generic, mass-conserving redistribution primitives (`pool_redistribute!`, `pro_rata_move!`) —
# the building blocks a relabeling arrival is assembled from. The two-strain escape model that uses
# them lives in `submodels/two_strain_escape.jl` and is exercised end-to-end by
# `examples/variant_arrival_model_check.jl`; here we pin the primitives in isolation.

@testset "Redistribution jump primitives" begin

    @testset "pool_redistribute!: split a pooled mass by weights" begin
        # Four sources, redistribute f : (1-f) into the first two, empty the rest — the shape the
        # escape collapse uses for each (h,i) layer.
        x = Float64[10, 20, 30, 40]
        pool_redistribute!(x, (1, 2, 3, 4), (1, 2), (0.25, 0.75))
        @test x[1] ≈ 0.25 * 100
        @test x[2] ≈ 0.75 * 100
        @test x[3] == 0.0 && x[4] == 0.0

        # Disjoint sources and targets: sources empty, targets receive the split.
        y = Float64[10, 20, 0, 0]
        pool_redistribute!(y, (1, 2), (3, 4), (0.4, 0.6))
        @test y[1] == 0.0 && y[2] == 0.0
        @test y[3] ≈ 0.4 * 30 && y[4] ≈ 0.6 * 30

        # A three-way split (weights need only sum to 1).
        z = Float64[6, 0, 0, 0]
        pool_redistribute!(z, (1,), (1, 2, 3), (0.5, 0.25, 0.25))
        @test z[1] ≈ 3.0 && z[2] ≈ 1.5 && z[3] ≈ 1.5
    end

    @testset "pool_redistribute! overlap: a source that is also a target is SET, not accumulated" begin
        # Index 1 is both a source and a target: it must end at its weight share of the pool, not
        # its old value plus the share. (The pool is read and sources cleared before any write.)
        x = Float64[10, 20, 30]
        pool_redistribute!(x, (1, 2, 3), (1, 2), (0.5, 0.5))
        @test x[1] ≈ 30.0 && x[2] ≈ 30.0 && x[3] == 0.0   # 60 pooled, split 50/50, NOT 10+30
    end

    @testset "mismatched paired lengths throw cleanly, not UB under @inbounds" begin
        # Both primitives read a paired tuple (`weights[k]` / `targets[k]`) under `@inbounds`, so a
        # length mismatch must fail fast rather than read out of bounds. The check folds away for
        # equal-length tuples (see the zero-allocation test below).
        @test_throws DimensionMismatch pool_redistribute!(zeros(4), (1, 2, 3, 4), (1, 2), (0.5,))
        @test_throws DimensionMismatch pool_redistribute!(
            zeros(4), (1, 2), (1, 2), (0.3, 0.3, 0.4)
        )
        @test_throws DimensionMismatch pro_rata_move!(zeros(4), (1, 2), (3,), 10.0)
    end

    @testset "pool_redistribute! conserves mass for weights summing to 1" begin
        rng = Random.MersenneTwister(1)
        for _ in 1:2000
            x = abs.(randn(rng, 6)) .* 1000
            before = sum(x)
            f = rand(rng)
            # sources = a random subset; targets = two of them; weights (f, 1-f).
            pool_redistribute!(x, (1, 2, 3, 4), (2, 5), (f, 1 - f))
            @test sum(x) ≈ before rtol = 1.0e-12
            @test all(>=(0.0), x)
        end
    end

    @testset "pro_rata_move!: split a fixed amount by source occupancy, capped" begin
        # 100 moved out of pools of 1000 and 1375, split in that proportion into two targets.
        x = Float64[1000, 1375, 0, 0]
        pro_rata_move!(x, (1, 2), (3, 4), 100.0)
        pool = 1000 + 1375
        @test x[3] ≈ 100 * 1000 / pool
        @test x[4] ≈ 100 * 1375 / pool
        @test x[1] ≈ 1000 - 100 * 1000 / pool
        @test x[2] ≈ 1375 - 100 * 1375 / pool
        @test sum(x) ≈ pool                              # mass conserved

        # A request larger than the pool is capped: the whole pool moves, sources empty.
        y = Float64[10, 5, 0, 0]
        pro_rata_move!(y, (1, 2), (3, 4), 1.0e9)
        @test y[1] ≈ 0.0 atol = 1.0e-12
        @test y[2] ≈ 0.0 atol = 1.0e-12
        @test y[3] + y[4] ≈ 15.0

        # Non-positive request, or empty pools, moves nothing.
        z = Float64[10, 5, 0, 0]
        pro_rata_move!(z, (1, 2), (3, 4), -3.0)
        @test z == Float64[10, 5, 0, 0]
        e = zeros(4)
        pro_rata_move!(e, (1, 2), (3, 4), 50.0)
        @test e == zeros(4)
    end

    @testset "pro_rata_move! never goes negative when the cap binds (IEEE regression)" begin
        # When `amount ≥ pool`, each source should be drained to exactly 0 — but `moved·xᵢ/pool`
        # can round UP an ulp without the internal `min(·, available)` clamp, leaving a compartment
        # at -4e-16. Round power-of-ten pools never expose it; deliberately untidy ones do. This
        # test fails against a `pro_rata_move!` that drops the clamp.
        rng = Random.MersenneTwister(20260717)
        for _ in 1:5000
            x = zeros(4)
            x[1] = exp(6 * randn(rng))
            x[2] = exp(6 * randn(rng))
            before = x[1] + x[2]
            pro_rata_move!(x, (1, 2), (3, 4), 1.0e12)     # cap always binds
            @test x[1] >= 0.0
            @test x[2] >= 0.0
            @test x[1] + x[2] + x[3] + x[4] ≈ before rtol = 1.0e-12
        end
    end

    @testset "both primitives are allocation-free with concrete NTuple indices" begin
        # They run per-particle in the PF hot loop; a boxing regression (e.g. a Vector{Int} index
        # set, or a Union-typed weight) would show up as allocations here.
        x = collect(1.0:6.0)
        s4, t2, w2 = (1, 2, 3, 4), (1, 2), (0.3, 0.7)
        pool_redistribute!(x, s4, t2, w2)                # warmup
        @test @allocated(pool_redistribute!(x, s4, t2, w2)) == 0

        y = collect(1.0:4.0)
        s2, t2b = (1, 2), (3, 4)
        pro_rata_move!(y, s2, t2b, 1.5)                  # warmup
        @test @allocated(pro_rata_move!(y, s2, t2b, 1.5)) == 0
    end
end
