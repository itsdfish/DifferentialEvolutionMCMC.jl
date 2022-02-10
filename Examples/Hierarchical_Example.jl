###################################################################################
#                             Warning
###################################################################################
# This is a work in progress. Convergence is currently sporadic. 
# Please submit PR if you would like to improve hierarchical models. 
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
n_subj = 50
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

# block update indicator
# update hyper parameters first
# update lower level parameters second
blocks = [
    [true,true,fill(false, n_subj),true],
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
    sample = resample,
    burnin = 20_000, 
    n_initial = (n_subj + 1) * 4,
    Np = 3,
    n_groups = 2,
    θsnooker = 0.1,
    blocking_on,
    blocks,
    #generate_proposal = variable_gamma
)
###################################################################################
#                             estimate parameters
###################################################################################
n_iter = 40_000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
###################################################################################
#                             Stan
###################################################################################
using StanSample
tmpdir = pwd() * "/tmp"

idx = repeat(1:n_subj, inner=n_data)
y = vcat(data...)
#xrep = repeat(x, inner=n_subj)
stan_data = Dict(
    "n_data" => n_data,
    "n_subj" => n_subj,
    "n" => n_data * n_subj,
    "idx" => idx,
    "y" => y,
    "x" => fill(0.0, n_data * n_subj),
)
###################################################################################################
#                                        Define Model
###################################################################################################
# load the Stan model file
stream = open("temp.stan","r")
stan_model = read(stream, String)
close(stream)
stan_model = SampleModel("temp", stan_model, tmpdir)
###################################################################################################
#                                     Estimate Parameters
###################################################################################################
stan_sample(
    stan_model;
    data = stan_data,
    #seed,
    num_chains = 4,
    num_samples = 1000,
    num_warmups = 1000,
    save_warmup = false
)
samples = read_samples(stan_model, :mcmcchains)
#rhats = filter(!isnan, MCMCChains.ess_rhat(samples).nt.rhat)
#mean(rhats .> 1.01)