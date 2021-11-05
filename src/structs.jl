"""
    DE(;
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
        sample = sample
    )

Differential Evolution MCMC object.

# Keywords

- `n_groups`: number of groups of particles. Default = 4
- `Np`: number of particles per group. Default = number of parameters * 3 if
    priors are passed
- `burnin`: number of burnin iterations. Default = 1000
- `discard_burnin`: indicates whether burnin samples are discarded. Default is true.
- `α`: migration probability. Default = .1
- `β`: mutation probability. Default = .1
- `ϵ`: noise in crossover step. Default = .001
- `σ`: standard deviation of noise added to parameters for mutation. Default = .05
- `κ`: recombination with probability (1-κ) during crossover. Default = 1.0
- `θsnooker`: sample along line x_i - z. Default = 0.0.  0.1 is recommended otherwise.
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
Constructor signature:

# References

* Ter Braak, C. J. A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces.

* Ter Braak, Cajo JF, and Jasper A. Vrugt. "Differential evolution Markov chain with snooker updater and fewer chains." Statistics and Computing 18.4 (2008): 435-446

* Turner, B. M., Sederberg, P. B., Brown, S. D., & Steyvers, M. (2013). A method for efficiently sampling from distributions with correlated dimensions. Psychological methods, 18(3), 368.

* Turner, B. M., & Sederberg, P. B. (2012). Approximate Bayesian computation with differential evolution. Journal of Mathematical Psychology, 56(5), 375-385.
"""
mutable struct DE{T1,F1,F2,F3,F4} <: AbstractSampler
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
    bounds::T1
    n_initial::Int64
    iter::Int64
    generate_proposal::F1
    update_particle!::F2
    evaluate_fitness!::F3
    sample::F4
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
        sample = sample
    )

    if  (n_groups == 1) && (α > 0)
        α = 0.0
        @warn "migration probability α > 0 but n_groups == 1. Changing α = 0.0"
    end

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
        sample
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
"""
mutable struct Particle{T}
    Θ::Vector{T}
    samples::Array{T,2}
    accept::Vector{Bool}
    weight::Float64
    lp::Vector{Float64}
end

Base.broadcastable(x::Particle) = Ref(x)

function Particle(;
        Θ=[.0],
        samples = Array{eltype(Θ),2}(undef, 1, 1),
        accept = Bool[], 
        weight = 0.0
    )
    return Particle(Θ, samples, accept, weight, Float64[])
end

function Particle(Θ::Number, samples, accept, weight, lp) 
    Particle([Θ], samples, accept, weight, lp)
end