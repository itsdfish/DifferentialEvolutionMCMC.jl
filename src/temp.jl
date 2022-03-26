cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, Distributions
Random.seed!(9528)
###################################################################################
#                             define functions
###################################################################################
# returns prior log likelihood
function sample_prior()
    LL = 0.0
    μβ0 = rand(Normal(1, 1))
    σβ0 = rand(truncated(Cauchy(0, 1), 0, Inf))
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    β0 = rand(Normal(0.0, σβ0), n_subj)
    return as_union([μβ0, σβ0, β0, σ])
end

# function for initial values
function prior_loglike(μβ0, σβ0, β0, σ)
    LL = 0.0
    LL += logpdf(Normal(1, 1), μβ0)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σβ0)
    LL += sum(logpdf.(Normal(0, σβ0), β0))
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end

# likelihood function 
function loglike(data, μβ0, σβ0, β0, σ)
    LL = 0.0
    for s in 1:length(data)
        μ = μβ0 + β0[s] 
        y = data[s] .- μ
        LL += sum(logpdf.(Normal(0, σ), y))
    end
    return LL
end
###################################################################################
#                             generate data
###################################################################################
# number of responses 
n_data = 50
# number of subjects 
n_subj = 10
# mean intercept
μβ0 = 1.0
# standard deviation intercept
σβ0 = 1.0
# subject intercept deviations
β0 = rand(Normal(0.0, σβ0), n_subj)
σ = .5

sim(μβ0, β0, σ, n_data) = rand(Normal(μβ0 + β0, σ), n_data)
data = [sim(μβ0, β0[i], σ, n_data) for i in 1:n_subj]
###################################################################################
#                             configure sampler
###################################################################################
# parameter bounds
bounds = (
    (-Inf,Inf),
    (0.0,Inf),
    (-Inf,Inf),
    (0.0,Inf),
)

# parameter names
names = (:μβ0, :σβ0, :β0, :σ)

# model object 
model = DEModel(;
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

blocks = [
    [true,false,fill(false, n_subj),false],
    [false,true,fill(false, n_subj),false],
    [false,false,fill(false, n_subj),true],
    [false,false,fill(true, n_subj),false],
]
blocks = as_union.(blocks)

# blocks = [[true,true,fill(false, n_subj),true],]
# subj_blocks = [[false,false,fill(false, n_subj),false] for i in 1:n_subj]
# map(i -> subj_blocks[i][3][i] = true, 1:n_subj)
# push!(blocks, subj_blocks...)
# blocks = as_union.(blocks)


# use block updating on each iteration 
blocking_on = x -> true

# sampler object
de = DE(;
    sample_prior,
    bounds, 
    #sample = resample,
    burnin = 500, 
    #n_initial = (n_subj + 1) * 4,
    Np = n_subj * 2 + 4,
    #n_groups = 2,
    #θsnooker = 0.1,
    blocking_on,
    blocks,
    α=.5,
    generate_proposal = variable_gamma,
)
###################################################################################
#                             estimate parameters
###################################################################################
n_iter = 2500

using DifferentialEvolutionMCMC: sample_init, block_update!

groups = sample_init(model, de, 100)

group = groups[1]

particle = group[1]

block_update!(model, de, group)