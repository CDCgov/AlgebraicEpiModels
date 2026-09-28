# Julia used to regenerate committed docs output; override with `just julia=julia <recipe>`.
julia := "julia +1.13"

# List recipes
default:
    @just --list

# Run a package's tests
test pkg="ConfigurableEpi":
    {{ julia }} --project={{ pkg }} -e 'using Pkg; Pkg.test()'

# Instantiate the Julia docs environment
docs-setup:
    {{ julia }} --project=julia-docs -e 'using Pkg; Pkg.instantiate()'

# Regenerate the API reference in docs/api from docstrings
docs-api:
    {{ julia }} --project=julia-docs julia-docs/make.jl

# Execute the Literate examples into docs/examples (all, or the named ones)
docs-examples *names:
    {{ julia }} --project=julia-docs julia-docs/examples.jl {{ names }}

# Build the site strictly into site/
docs-build:
    uv run --group docs zensical build --strict --clean

# Serve the site locally with live reload
docs-serve:
    uv run --group docs zensical serve
