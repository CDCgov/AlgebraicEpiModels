@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(AlgebraicEpiMech, ambiguities = false, persistent_tasks = false)
    Aqua.test_ambiguities(AlgebraicEpiMech)
end
