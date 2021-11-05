###################################################################################
#                             load dependencies
###################################################################################
cd(@__DIR__)
using Revise, DifferentialEvolutionMCMC, Random, Distributions
Random.seed!(50514)
###################################################################################
#                             define functions
###################################################################################
# returns prior log likelihood
function prior_loglike(μ, σ)
    LL = 0.0
    LL += logpdf(Normal(0, 1), μ)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end

# function for initial values
function sample_prior()
    μ = rand(Normal(0, 1))
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    return [μ,σ]
end

# likelihood function 
function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end
###################################################################################
#                             generate data
###################################################################################
data = rand(Normal(0.0, 1.0), 50)
###################################################################################
#                             configure sampler
###################################################################################
# parameter names
names = (:μ,:σ)
# parameter bounds
bounds = ((-Inf,Inf),(0.0,Inf))

# model object
model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

# DEMCMC sampler object
de = DE(;bounds, burnin = 1000, Np = 6, 
    blocking_on = x -> true, blocks = [[true,false],[false,true]])
# number of interations per particle
n_iter = 2000
###################################################################################
#                             estimate parameters
###################################################################################
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
Random.seed!(1)
chains = sample(model, de, n_iter, progress=true)