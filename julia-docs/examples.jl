# Execute the Literate examples in julia-docs/examples/ and write Markdown and figures to
# docs/examples/. The output is committed; regenerate after changing an example or the API:
#   julia +1.13 --project=julia-docs julia-docs/examples.jl [example_name ...]
using Literate
import CairoMakie

CairoMakie.activate!(type = "png", px_per_unit = 1.5)

const SRC = joinpath(@__DIR__, "examples")
const OUT = normpath(joinpath(@__DIR__, "..", "docs", "examples"))

# PNG first keeps plots small; Graphviz diagrams have no PNG `show`, so they stay SVG.
const IMAGE_FORMATS = [(MIME("image/png"), ".png"), (MIME("image/svg+xml"), ".svg")]

mkpath(OUT)
names = isempty(ARGS) ? [first(splitext(f)) for f in readdir(SRC) if endswith(f, ".jl")] : ARGS
for name in names
    stale = Regex("^\\Q$name\\E(\\.md|-\\d+\\.(svg|png|jpeg))\$")
    foreach(f -> occursin(stale, f) && rm(joinpath(OUT, f)), readdir(OUT))
    Literate.markdown(
        joinpath(SRC, "$name.jl"), OUT; flavor = Literate.CommonMarkFlavor(), execute = true, credit = false,
        image_formats = IMAGE_FORMATS,
    )
end
