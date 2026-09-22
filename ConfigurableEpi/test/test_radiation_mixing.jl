using Test
using ConfigurableEpi
using LinearAlgebra: I

@testset "Radiation mixing" begin
    mktempdir() do dir
        path = joinpath(dir, "radiation_flows.csv")
        rows = ["origin,destination,flow", "ny,ca,0.2", "ny,tx,0.1", "ny,ny,0.0", "ca,ny,0.3", "ca,tx,0.3", "tx,ny,0.5", "tx,ca,0.0"]
        write(path, join(rows, "\n") * "\n")

        # Rows are re-normalised over the SELECTED locations, so a subset keeps rows summing to 1.
        @test load_radiation_matrix(["ny", "ca"]; path) ≈ [0.0 1.0; 1.0 0.0]
        M = load_radiation_matrix(["NY", "ca", "tx"]; path)   # case-insensitive
        @test M[1, :] ≈ [0.0, 2 / 3, 1 / 3]
        @test M[2, :] ≈ [0.5, 0.0, 0.5]
        @test M[3, :] ≈ [1.0, 0.0, 0.0]
        @test all(iszero(M[i, i]) for i in 1:3)

        @test_throws ArgumentError load_radiation_matrix(["ny", "ny"]; path)      # duplicate
        @test_throws ArgumentError load_radiation_matrix(["ny", "fl"]; path)      # absent from the file
        @test_throws ArgumentError load_radiation_matrix(["tx", "ca"]; path)      # tx retains no flow to ca
        @test_throws ArgumentError load_radiation_matrix(String[]; path)
        @test_throws ErrorException load_radiation_matrix(["ny"]; path = joinpath(dir, "nope.csv"))

        bad = joinpath(dir, "bad.csv")
        write(bad, "origin,destination,flow\nny,ca,-0.2\nca,ny,0.3\n")
        @test_throws ErrorException load_radiation_matrix(["ny", "ca"]; path = bad)   # negative flow
        write(bad, "origin,destination,flow\nny,ca,Inf\nca,ny,0.3\n")
        @test_throws ErrorException load_radiation_matrix(["ny", "ca"]; path = bad)   # infinite flow
        write(bad, "origin,destination,flow\nny,ny,0.1\nny,ca,0.2\nca,ny,0.3\n")
        @test_throws ArgumentError load_radiation_matrix(["ny", "ca"]; path = bad)    # self-flow
        write(bad, "origin,destination\nny,ca\n")
        @test_throws ErrorException load_radiation_matrix(["ny", "ca"]; path = bad)   # wrong header
    end

    M = [0.0 1.0; 1.0 0.0]
    @test contact_matrix(M, 0.0) == Matrix(1.0I, 2, 2)
    @test contact_matrix(M, 1.0) == M
    @test contact_matrix(M, 0.25) ≈ [0.75 0.25; 0.25 0.75]
end
