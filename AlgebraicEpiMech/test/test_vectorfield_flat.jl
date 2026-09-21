@testitem "flatten_symbols - basic tuple flattening" begin
    using AlgebraicEpiMech

    result = flatten_symbols((:x, :y, :z))
    @test result == :x_y_z
end

@testitem "flatten_symbols - nested tuple flattening" begin
    using AlgebraicEpiMech

    result = flatten_symbols(((:x, :y), :z))
    @test result == :x_y_z
end

@testitem "flatten_symbols - nested tuple flattening" begin
    using AlgebraicEpiMech

    result = flatten_symbols((((:a, :b), :c), :d))
    @test result == :a_b_c_d
end

@testitem "flatten_symbols - nested tuple flattening other direction" begin
    using AlgebraicEpiMech

    result = flatten_symbols((:a, (:b, (:c, :d))))
    @test result == :a_b_c_d
end

@testitem "flatten_symbols - single symbol" begin
    using AlgebraicEpiMech

    result = flatten_symbols(:single)
    @test result == :single
end

@testitem "vectorfield_flat - function signature and type" begin
    using AlgebraicPetri, LabelledArrays
    using AlgebraicEpiMech

    # Create a simple Petri net
    pn = LabelledPetriNet([:S, :I], :infection => ((:S, :I) => (:I, :I)))

    # Generate vectorfield function
    f = vectorfield_flat(pn)

    # Test that it returns a function
    @test isa(f, Function)

    # Test function can be called with proper arguments
    du = LVector(S = 0.0, I = 0.0)
    u = LVector(S = 100.0, I = 1.0)
    p = LVector(infection = 0.01)
    t = 0.0

    result = f(du, u, p, t)
    @test du[:S] == -1.0  # Should return -1.0 change in S for these inputs
    @test du[:I] == 1.0  # Should return 1.0 change in I for these inputs
end

@testitem "vectorfield_flat matches the dense reference bit for bit" begin
    # The sparse, name-resolved-once implementation must reproduce the dense reference
    # (`dense_vectorfield_flat`, defined below) exactly — same factor and summation order, so `==` rather
    # than `≈` — on both the production argument shapes (LVector state, NamedTuple rates whose
    # keys are NOT in net order) and the by-name dictionary path the test snippets use.
    using AlgebraicPetri, LabelledArrays, Catlab
    using AlgebraicPetri: snames, tnames
    using AlgebraicPetri.TypedPetri: typed_product
    using AlgebraicEpiMech

    # The dense, resolve-everything-per-call form `vectorfield_flat` replaced (kept here, not in
    # src, as the executable specification): the new implementation must match it bit for bit.
    function dense_vectorfield_flat(pn)
        tm = TransitionMatrices(pn)
        dt = tm.output - tm.input
        return (
            du,
            u,
            p,
            t,
        ) -> begin
            rates = zeros(valtype(du), nt(pn))
            u_m = [u[flatten_symbols(sname(pn, i))] for i in 1:ns(pn)]
            p_m = [p[flatten_symbols(tname(pn, i))] for i in 1:nt(pn)]
            for i in 1:nt(pn)
                rates[i] = AlgebraicPetri.valueat(p_m[i], u, t) *
                    prod(u_m[j]^tm.input[i, j] for j in 1:ns(pn))
            end
            for j in 1:ns(pn)
                du[flatten_symbols(sname(pn, j))] = sum(
                    rates[i] * dt[i, j] for i in 1:nt(pn); init = 0.0
                )
            end
            du
        end
    end

    # Deterministic, irrational-stride "random" values: no RNG dependency in the test env, and
    # every state and rate is distinct and non-round so an ordering slip cannot cancel out.
    strided(n, seed) = [mod(seed + k * 0.6180339887, 1.0) for k in 1:n]

    geo = attach_observation(
        dom(
            typed_product(
                create_model(OnePopulationTyping(), SEIRS()),
                create_model(OnePopulationTyping(), GeographicStratification([:a, :b, :c])),
            ),
        ),
        AtEvent(:transmission); n_stages = 2,
    )
    dimer = LabelledPetriNet([:A, :B], :dimerise => ((:A, :A) => :B), :decay => (:B => :A))
    # Multiplicity 4: `u^4` is a correctly-rounded `pow`, not a repeated product, so only
    # applying `^` per (species, multiplicity) as the reference does reproduces it exactly.
    quad = LabelledPetriNet(
        [:A, :B], :tetramerise => ((:A, :A, :A, :A) => :B), :split => (:B => (:A, :A, :A, :A))
    )
    simple = LabelledPetriNet([:S, :I], :infection => ((:S, :I) => (:I, :I)))

    for pn in (geo, dimer, quad, simple)
        f = vectorfield_flat(pn)
        g = dense_vectorfield_flat(pn)
        ssyms = [flatten_symbols(s) for s in snames(pn)]
        tsyms = [flatten_symbols(t) for t in tnames(pn)]
        for rep in 1:3
            uvals = 100 .* strided(length(ssyms), 0.1 * rep)
            pvals = strided(length(tsyms), 0.3 * rep)
            u = LVector(NamedTuple{Tuple(ssyms)}(Tuple(uvals)))
            # Rate keys deliberately out of net order: reversed, then rotated by `rep`.
            perm = circshift(reverse(1:length(tsyms)), rep)
            p = NamedTuple{Tuple(tsyms[perm])}(Tuple(pvals[perm]))
            du_fast = LVector(NamedTuple{Tuple(ssyms)}(Tuple(zeros(length(ssyms)))))
            du_ref = deepcopy(du_fast)
            f(du_fast, u, p, 0.0)
            g(du_ref, u, p, 0.0)
            @test du_fast == du_ref

            ud = Dict(zip(ssyms, uvals))
            pd = Dict(zip(tsyms, pvals))
            dd_fast = Dict(s => 0.0 for s in ssyms)
            dd_ref = copy(dd_fast)
            f(dd_fast, ud, pd, 0.0)
            g(dd_ref, ud, pd, 0.0)
            @test dd_fast == dd_ref
            @test all(dd_fast[ssyms[k]] == du_fast[k] for k in eachindex(ssyms))
        end
    end

    # The production path (LVector state, NamedTuple rates, Float64) allocates nothing per call.
    f_geo = vectorfield_flat(geo)
    gsyms = [flatten_symbols(s) for s in snames(geo)]
    gtsyms = [flatten_symbols(t) for t in tnames(geo)]
    u_geo = LVector(NamedTuple{Tuple(gsyms)}(Tuple(100 .* strided(length(gsyms), 0.7))))
    p_geo = NamedTuple{Tuple(gtsyms)}(Tuple(strided(length(gtsyms), 0.9)))
    du_geo = LVector(NamedTuple{Tuple(gsyms)}(Tuple(zeros(length(gsyms)))))
    f_geo(du_geo, u_geo, p_geo, 0.0)
    @test @allocated(f_geo(du_geo, u_geo, p_geo, 0.0)) == 0

    # A narrower output type keeps the dense form's numeric path: rates converted to Float32,
    # contributions accumulated from a Float64 zero, one final rounding. Six catalytic taps into
    # `X` with rates 1e8, 1, 1, 1, 1, 1 give 100000008 that way and 100000000 if accumulated in
    # Float32 — so this case discriminates.
    taps = LabelledPetriNet(
        [:X, :Y], [Symbol("tap", i) => (:Y => (:X, :Y)) for i in 1:6]...
    )
    f32 = vectorfield_flat(taps)
    g32 = dense_vectorfield_flat(taps)
    u32 = LVector(X = 0.0f0, Y = 1.0f0)
    p32 = (tap1 = 1.0f8, tap2 = 1.0f0, tap3 = 1.0f0, tap4 = 1.0f0, tap5 = 1.0f0, tap6 = 1.0f0)
    du32_fast = LVector(X = 0.0f0, Y = 0.0f0)
    du32_ref = LVector(X = 0.0f0, Y = 0.0f0)
    f32(du32_fast, u32, p32, 0.0)
    g32(du32_ref, u32, p32, 0.0)
    @test du32_fast == du32_ref
    @test du32_fast[:X] == 100000008.0f0

    # Function-valued rates still go through `AlgebraicPetri.valueat`.
    f = vectorfield_flat(simple)
    u = LVector(S = 10.0, I = 2.0)
    du = LVector(S = 0.0, I = 0.0)
    f(du, u, (infection = (u, t) -> 0.5 * u[:S],), 1.0)
    @test du[:S] == -0.5 * 10.0 * 10.0 * 2.0
    @test du[:I] == -du[:S]

    # A rate container missing a transition name is an error naming the transition, not a
    # silent zero.
    @test_throws ArgumentError f(du, u, (wrong = 1.0,), 0.0)
end
