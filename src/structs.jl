"""
    function DE(;
        n_groups = 4, 
        priors = nothing, 
        Np, 
        burnin = 1000, 
        discard_burnin = true, 
        α = .1,
        β = .1, 
        ϵ = .001,
        σ = .05, 
        κ = 1.0, 
        θsnooker = 0.0, 
        bounds, 
        n_initial = 0, 
        generate_proposal = random_gamma, 
        update_particle! = mh_update!,
        evaluate_fitness! = compute_posterior!, 
        sample = sample,
        blocking_on = x -> false,
        blocks = [false]
    )

Differential Evolution MCMC object.

# Keywords

- `n_groups=4`: number of groups of particles. 
- `Np`: number of particles per group.
- `burnin=1000`: number of burnin iterations
- `discard_burnin`: indicates whether burnin samples are discarded. Default is true.
- `α=.1`: migration probability.
- `β=.1`: mutation probability.
- `ϵ=.001`: noise in crossover step.
- `σ=.05`: standard deviation of noise added to parameters for mutation.
- `κ=1.0`: recombination with probability (1-κ) during crossover.
- `θsnooker=0`: sample along line x_i - z. 0.1 is recommended if > 0.
- `n_initial`: initial number of samples from the prior distribution when `sample=resample`. 10 times the number of parameters
is a typical value
- `bounds`: a vector of tuples for lower and upper bounds of each parameter
- `iter`: current iteration
- `generate_proposal`: a function that generates proposals. Default is the two mode proposal described in
Turner et al. 2012. You can also choose `fixed_gamma`, `variable_gamma` (see help) or pass a custom function
- `update_particle!`: a function for updating the particle with a proposal value. Default: `mh_update!`, which uses the 
Metropolis-Hastings rule.
- `evaluate_fitness!`: a function for evaluating the fitness of a posterior. The default is to compute the posterior loglikelihood with 
 `compute_posterior!`. Select `evaluate_fun!` for optimization rather than MCMC sampling.
- `sample`: a function for sampling particles during the crossover step. The default `sample` uses current particle
parameter values whereas `resample` samples from the history of accepted values for each particle. Np must 3 or greater 
when using `resample`.
- `blocking_on = x -> false`: a function that indicates whether block updating is used on each iteration. The function requires optimization_tests
arguement for the DE sampler object and must return a true or false value. 
- `blocks`: a vector of boolean vectors indicating which parameters to update. Each sub-vector represents a 
block and each element in the sub-vector indicates which parameters are updated within the block. For example, [[true,false],[false,true]]
indicates that the parameter in the first position is updated on the first block and the parameter in the second position is updated on the 
second block. If a parameter is a vector or matrix, they are nested within the block sub-vector. 

# References

* Ter Braak, C. J. A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces.

* Ter Braak, Cajo JF, and Jasper A. Vrugt. "Differential evolution Markov chain with snooker updater and fewer chains." Statistics and Computing 18.4 (2008): 435-446

* Turner, B. M., Sederberg, P. B., Brown, S. D., & Steyvers, M. (2013). A method for efficiently sampling from distributions with correlated dimensions. Psychological methods, 18(3), 368.

* Turner, B. M., & Sederberg, P. B. (2012). Approximate Bayesian computation with differential evolution. Journal of Mathematical Psychology, 56(5), 375-385.
"""
@concrete mutable struct DE <: AbstractSampler
    n_groups::Int64
    Np::Int64
    burnin::Int64
    discard_burnin::Bool
    α::Float64
    β::Float64
    ϵ::Float64
    σ::Float64
    κ::Float64
    θsnooker::Float64
    bounds
    n_initial::Int64
    iter::Int64
    generate_proposal
    update_particle!
    evaluate_fitness!
    sample
    blocking_on
    blocks
    samples
end

function DE(;
        n_groups = 4, 
        priors = nothing, 
        Np, 
        burnin = 1000, 
        discard_burnin = true, 
        α = .1,
        β = .1, 
        ϵ = .001,
        σ = .05, 
        κ = 1.0, 
        θsnooker = 0.0, 
        bounds, 
        n_initial = 0, 
        generate_proposal = random_gamma, 
        update_particle! = mh_update!,
        evaluate_fitness! = compute_posterior!, 
        sample = sample,
        blocking_on = x -> false,
        blocks = [false],
        sample_prior,
    )

    if  (n_groups == 1) && (α > 0)
        α = 0.0
        @warn "migration probability α > 0 but n_groups == 1. Changing α = 0.0"
    end

    samples = initialize_samples(sample_prior)

    return DE(
        n_groups, 
        Np, 
        burnin, 
        discard_burnin, 
        α, 
        β, 
        ϵ, 
        σ, 
        κ, 
        θsnooker, 
        bounds, 
        n_initial, 
        1, 
        generate_proposal, 
        update_particle!, 
        evaluate_fitness!, 
        sample,
        blocking_on,
        blocks,
        samples
    )
end

"""
    function DEModel(
        args...; 
        prior_loglike = nothing, 
        loglike, 
        names, 
        sample_prior, 
        data, 
        kwargs...
    )

A model object containing the log likelihood function and prior distributions.

# Keywords

- `args...`: optional positional arguments for `loglike`
- `prior_loglike=nothing`: log likelihood of posterior sample. A function must be 
define for `sample`, but not for `optimize`.
- `loglike`: a log likelihood function for Bayesian parameter estimation or an objective function for 
optimization. 
- `sample_prior`: a function for initial values. Typically, a prior distribution is ideal.
- `names`: parameter names
- `kwargs...`: optional keyword arguments for `loglike`
"""
struct DEModel{F,L,T,S} <: AbstractModel where {F <: Function,L,T,S}
    prior_loglike::L
    loglike::F
    sample_prior::S
    names::T
end

function DEModel(
        args...; 
        prior_loglike = nothing, 
        loglike, 
        names, 
        sample_prior, 
        data, 
        kwargs...
    )
    return DEModel(
        x -> prior_loglike(x...), 
        x -> loglike(data, args..., x...; kwargs...),
        sample_prior, 
        names
    )
end

"""
    Particle{T}

# Fields 

- `θ`: a vector of parameters
- `samples`: a 2-dimensional array containing all acccepted proposals
- `accept`: proposal acceptance. 1: accept, 0: reject
- `weight`: particle weight based on model fit (currently posterior log likelihood)
- `lp`: a vector of log posterior probabilities associated with each accepted proposal
- `id`: particle id
"""
mutable struct Particle{T}
    Θ::Vector{T}
    accept::Vector{Bool}
    weight::Float64
    lp::Vector{Float64}
    id::Int
end

Base.broadcastable(x::Particle) = Ref(x)

function Particle(;
        Θ=[.0],
        accept = Bool[], 
        weight = 0.0,
        id = 0
    )
    return Particle(Θ, accept, weight, Float64[], id)
end

function Particle(Θ::Number, accept, weight, lp, id) 
    Particle([Θ], accept, weight, lp, id)
end