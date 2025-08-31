using MaterialModelsTesting
using Documenter

DocMeta.setdocmeta!(MaterialModelsTesting, :DocTestSetup, :(using MaterialModelsTesting); recursive=true)

makedocs(;
    modules=[MaterialModelsTesting],
    authors="Knut Andreas Meyer and contributors",
    sitename="MaterialModelsTesting.jl",
    format=Documenter.HTML(;
        canonical="https://KnutAM.github.io/MaterialModelsTesting.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/KnutAM/MaterialModelsTesting.jl",
    devbranch="main",
)
