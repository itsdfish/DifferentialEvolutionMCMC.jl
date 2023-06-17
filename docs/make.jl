using Documenter
using DifferentialEvolutionMCMC

makedocs(
    sitename = "DifferentialEvolutionMCMC",
    format = Documenter.HTML(
        assets = [
            asset(
                "https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap",
                class = :css,
            ),
        ],
        collapselevel = 1,
    ),
    modules = [DifferentialEvolutionMCMC],
    pages = ["home" => "index.md",
    "examples" => ["Binomial Model" => "binomial.md",
                "Gaussian Model" => "gaussian.md"],
    "api" => "api.md"]
)

deploydocs(
    repo = "github.com/itsdfish/DifferentialEvolutionMCMC.jl.git",
)