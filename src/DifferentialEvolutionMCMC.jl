module DifferentialEvolutionMCMC
    using Random, StatsBase, ProgressMeter, Parameters, Distributions
    import AbstractMCMC: step!, AbstractSampler, AbstractModel
    import AbstractMCMC: bundle_samples, sample, MCMCThreads
    import MCMCChains: Chains
    export DE, Particle, DEModel, sample, MCMCThreads, fixed_gamma
    export variable_gamma, random_gamma, evaluate_fun!, compute_posterior!
    export greedy_update!, optimize, get_optimal
    include("structs.jl")
    include("main.jl")
    include("optimize.jl")
    include("migration.jl")
    include("crossover.jl")
    include("mutation.jl")
    include("utilities.jl")
end
