###################################################################################
#                             load dependencies
###################################################################################
cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, Distributions
Random.seed!(9528)
###################################################################################
#                             define functions
###################################################################################
# returns prior log likelihood
function sample_prior()
    LL = 0.0
    β0 = rand(Normal(1, 1))
    β1 = rand(Normal(.5, 1))
    σβ0 = rand(truncated(Cauchy(0, 1), 0, Inf))
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    β0i = rand(Normal(0.0, σβ0), n_subj)
    return as_union([β0, β1, σβ0, β0i, σ])
end

# function for initial values
function prior_loglike(β0, β1, σβ0, β0i, σ)
    LL = 0.0
    LL += logpdf(Normal(0, 1), β0)
    LL += sum(logpdf.(Normal(0, σβ0), β0i))
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σβ0)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    LL += logpdf(Normal(.5, 1), β1)
    return LL
end

# likelihood function 
function loglike(data, x, β0, β1, σβ0, β0i, σ)
    LL = 0.0
    for s in 1:length(data)
        μ = β0 + β0i[s] + β1 * x[s]
        LL += sum(logpdf(Normal(μ, σ), data[s]))
    end
    return LL
end
###################################################################################
#                             generate data
###################################################################################
# number of responses 
n_data = 100
# number of subjects 
n_subj = 50
# mean intercept
β0 = 1.0
# standard deviation intercept
σβ0 = 1.0
# mean slope
β1 = .5
# subject intercept deviations
β0i = rand(Normal(0.0, σβ0), n_subj)
σ = 1.0

x = rand(Normal(0, 1), n_subj)
linear(x, β0, β0i, β1, σ, n_data) = β0 + β0i + β1 * x  .+ rand(Normal(0, σ), n_data)
data = [linear(x[i], β0, β0i[i], β1, σ, n_data) for i in 1:n_subj]
###################################################################################
#                             configure sampler
###################################################################################
# parameter bounds
bounds = (
    (-Inf,Inf),
    (-Inf,Inf),
    (0.0,Inf),
    (-Inf,Inf),
    (0.0,Inf),
)

# parameter names
names = (:β0, :β1, :σβ0, :β0i, :σ)

# model object 
model = DEModel(x; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

# sampler object
de = DE(;
    bounds, 
    sample = resample,
    burnin = 20_000, 
    n_initial = (n_subj + 1) * 10,
    Np = 3,
    n_groups = 2,
    θsnooker = 0.1
)
###################################################################################
#                             estimate parameters
###################################################################################
n_iter = 50_000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)