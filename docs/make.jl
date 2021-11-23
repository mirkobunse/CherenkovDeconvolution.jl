using Documenter, CherenkovDeconvolution

makedocs(;
    sitename = "CherenkovDeconvolution.jl",
    pages = [
        "Manual" => "index.md",
        "Developer manual" => "developer-manual.md",
        "Python wrapper" => "python-wrapper.md",
        "API reference" => "api-reference.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/mirkobunse/CherenkovDeconvolution.jl.git",
    devbranch = "main"
)
