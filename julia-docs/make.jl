# Render the Julia API reference to Markdown for the Zensical site. The output in docs/api/ is
# committed, so the site builds without Julia; regenerate after changing docstrings:
#   julia +1.13 --project=julia-docs julia-docs/make.jl
using Documenter, DocumenterMarkdown
using AlgebraicEpiMech, ConfigurableEpi

const API_DIR = normpath(joinpath(@__DIR__, "..", "docs", "api"))

makedocs(;
    format = Markdown(),
    modules = [AlgebraicEpiMech, ConfigurableEpi],
    sitename = "AlgebraicEpiModels",
    remotes = nothing,  # source links embed the commit SHA, so committed output would always drift
    root = @__DIR__,
    source = "src",
    build = API_DIR,
    clean = true,
    checkdocs = :none,
    warnonly = true,
)

# Method anchors contain spaces, so DocumenterMarkdown escapes those links instead of emitting them.
const ESCAPED_LINK = r"\\\[`([^`]+)`\\\]\(([^)]+)\)"
unescape_link(s) = (m = match(ESCAPED_LINK, s); "[`$(m[1])`]($(replace(m[2], " " => "%20")))")

# Short-form `(@ref)` links to symbols outside the reference cannot be resolved by the site.
for file in readdir(API_DIR; join = true)
    endswith(file, ".md") || continue
    text = replace(read(file, String), r"\[`([^`]+)`\]\(@ref\)" => s"`\1`")
    text = replace(text, ESCAPED_LINK => unescape_link)
    write(file, rstrip(text) * "\n")
end
