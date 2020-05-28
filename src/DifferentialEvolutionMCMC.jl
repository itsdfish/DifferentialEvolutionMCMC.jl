module DifferentialEvolutionMCMC
    using Random, StatsBase, ProgressMeter, Parameters, Distributions
    import AbstractMCMC: step!, AbstractSampler, AbstractModel
    import AbstractMCMC: bundle_samples, sample, MCMCThreads
    import MCMCChains: Chains
    export DE, Particle, DEModel, sample, MCMCThreads
    include("structs.jl")
    include("main.jl")
    include("migration.jl")
    include("crossover.jl")
    include("mutation.jl")
    include("utilities.jl")
end
