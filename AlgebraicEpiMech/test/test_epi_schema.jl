@testitem "create_one_population_schema - basic functionality" begin
    using AlgebraicPetri
    using AlgebraicPetri.TypedPetri
    using AlgebraicEpiMech

    schema_type = OnePopulationSchema()

    # Test basic schema creation with default parameters
    schema = create_schema(schema_type)

    # Verify it returns a LabelledPetriNet
    @test schema isa LabelledPetriNet

    # Verify it has the default population type (using proper API)
    @test :Population in snames(schema)

    # Verify it has the required transitions
    @test :transmission in tnames(schema)
    @test :disease in tnames(schema)
    @test :reversion in tnames(schema)
    @test :waning in tnames(schema)
end

# @testitem "create_one_population_schema - custom population type" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test with custom population type
#     custom_schema = create_one_population_schema(population_type = :CustomPop)

#     # Verify it has the custom population type (using proper API)
#     @test :CustomPop in snames(custom_schema)
#     @test :Population ∉ snames(custom_schema)  # Should not have default type

#     # Verify transitions are still present
#     @test :transmission in tnames(custom_schema)
#     @test :disease in tnames(custom_schema)
#     @test :reversion in tnames(custom_schema)
#     @test :waning in tnames(custom_schema)
# end

# @testitem "create_one_population_schema - with additional transitions" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test with additional population transitions
#     schema_with_extra = create_one_population_schema(
#         :birth => ((:Population) => (:Population, :Population))
#     )

#     # Verify all transitions are present
#     @test :transmission in tnames(schema_with_extra)
#     @test :disease in tnames(schema_with_extra)
#     @test :reversion in tnames(schema_with_extra)
#     @test :waning in tnames(schema_with_extra)
#     @test :birth in tnames(schema_with_extra)
# end

# @testitem "create_one_population_schema - schema structure validation" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test the internal structure of the schema
#     schema = create_one_population_schema()

#     # Verify the schema has exactly one population type
#     @test length(snames(schema)) == 1

#     # Verify all transition types exist
#     @test :transmission in tnames(schema)
#     @test :disease in tnames(schema)
#     @test :reversion in tnames(schema)
#     @test :waning in tnames(schema)
# end

# @testitem "create_uninfected_infected_schema - basic functionality" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test basic schema creation with default parameters
#     schema = create_uninfected_infected_schema()

#     # Verify it returns a LabelledPetriNet
#     @test schema isa LabelledPetriNet

#     # Verify it has the default population types
#     @test :Uninfected in snames(schema)
#     @test :Infected in snames(schema)
#     @test length(snames(schema)) == 2

#     # Verify it has the required transitions
#     @test :transmission in tnames(schema)
#     @test :disease in tnames(schema)
#     @test :reversion in tnames(schema)
#     @test :waning in tnames(schema)
# end

# @testitem "create_uninfected_infected_schema - custom population types" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test with custom population types
#     custom_schema = create_uninfected_infected_schema(
#         uninfected_type = :Susceptible,
#         infected_type = :Infectious
#     )

#     # Verify it has the custom population types
#     @test :Susceptible in snames(custom_schema)
#     @test :Infectious in snames(custom_schema)
#     @test :Uninfected ∉ snames(custom_schema)  # Should not have default types
#     @test :Infected ∉ snames(custom_schema)

#     # Verify transitions are still present
#     @test :transmission in tnames(custom_schema)
#     @test :disease in tnames(custom_schema)
#     @test :reversion in tnames(custom_schema)
#     @test :waning in tnames(custom_schema)
# end

# @testitem "create_uninfected_infected_schema - with additional transitions" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test with additional population transitions (recovery)
#     schema_with_recovery = create_uninfected_infected_schema(
#         :recovery => (:Infected => :Uninfected)
#     )

#     # Verify all transitions are present
#     @test :transmission in tnames(schema_with_recovery)
#     @test :disease in tnames(schema_with_recovery)
#     @test :reversion in tnames(schema_with_recovery)
#     @test :waning in tnames(schema_with_recovery)
#     @test :recovery in tnames(schema_with_recovery)

#     # Verify populations are still correct
#     @test :Uninfected in snames(schema_with_recovery)
#     @test :Infected in snames(schema_with_recovery)
#     @test length(snames(schema_with_recovery)) == 2
# end

# # Tests for the new schema type system and create_schema interface

# @testitem "OnePopulationSchema - constructor and fields" begin
#     using AlgebraicEpiMech

#     # Test default constructor
#     schema = OnePopulationSchema()
#     @test schema.population_type == :Population
#     @test schema isa OnePopulationSchema
#     @test schema isa EpidemiologicalTyping

#     # Test custom constructor
#     custom_schema = OnePopulationSchema(population_type = :Individual)
#     @test custom_schema.population_type == :Individual
# end

# @testitem "UninfectedInfectedSchema - constructor and fields" begin
#     using AlgebraicEpiMech

#     # Test default constructor
#     schema = UninfectedInfectedSchema()
#     @test schema.uninfected_type == :Uninfected
#     @test schema.infected_type == :Infected
#     @test schema isa UninfectedInfectedSchema
#     @test schema isa EpidemiologicalTyping

#     # Test custom constructor
#     custom_schema = UninfectedInfectedSchema(
#         uninfected_type = :Susceptible,
#         infected_type = :Infectious
#     )
#     @test custom_schema.uninfected_type == :Susceptible
#     @test custom_schema.infected_type == :Infectious
# end

# @testitem "create_schema - error for unimplemented schema types" begin
#     using AlgebraicEpiMech

#     # Define a mock schema type for testing error handling
#     struct MockSchema <: EpidemiologicalTyping end

#     # Test that unimplemented schema types throw an error
#     mock_schema = MockSchema()
#     @test_throws ErrorException create_schema(mock_schema)

#     # Verify the error message mentions the type
#     try
#         create_schema(mock_schema)
#         @test false  # Should never reach here
#     catch e
#         @test occursin("MockSchema", string(e))
#         @test occursin("not implemented", string(e))
#     end
# end

# @testitem "create_schema - OnePopulationSchema integration" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test schema-based creation
#     schema = OnePopulationSchema(population_type = :Individual)
#     net = create_schema(schema)

#     # Verify the resulting Petri net
#     @test net isa LabelledPetriNet
#     @test :Individual in snames(net)
#     @test :transmission in tnames(net)
#     @test :disease in tnames(net)
#     @test :reversion in tnames(net)
#     @test :waning in tnames(net)

#     # Test with additional transitions
#     net_with_extra = create_schema(
#         schema,
#         :birth => (:Individual => (:Individual, :Individual))
#     )
#     @test :birth in tnames(net_with_extra)
# end

# @testitem "create_schema - UninfectedInfectedSchema integration" begin
#     using AlgebraicPetri
#     using AlgebraicPetri.TypedPetri
#     using AlgebraicEpiMech

#     # Test schema-based creation
#     schema = UninfectedInfectedSchema(
#         uninfected_type = :Susceptible,
#         infected_type = :Infectious
#     )
#     net = create_schema(schema)

#     # Verify the resulting Petri net
#     @test net isa LabelledPetriNet
#     @test :Susceptible in snames(net)
#     @test :Infectious in snames(net)
#     @test :transmission in tnames(net)
#     @test :disease in tnames(net)
#     @test :reversion in tnames(net)
#     @test :waning in tnames(net)

#     # Test with additional transitions
#     net_with_recovery = create_schema(
#         schema,
#         :recovery => (:Infectious => :Susceptible)
#     )
#     @test :recovery in tnames(net_with_recovery)
# end
