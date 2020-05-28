using DifferentialEvolutionMCMC, Test
cd(@__DIR__)
@testset "DE-MCMC Tests" begin
    include("DEMCMC_Tests.jl")
end
