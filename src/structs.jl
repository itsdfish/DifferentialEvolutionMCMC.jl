"""
Differential Evolution MCMC object
* `n_groups`: number of groups of particles. Default = 4
* `Np`: number of particles per chain. Default = number of parameters * 3 if
    priors are passed
* `burnin`: number of burnin iterations. Default = 1000
* `α`: migration probability. Default = .1
* `β`: mutation probability. Default = .1
* `ϵ`: noise in crossover step. Default = .001
* `σ`: standard deviation of noise added to parameters for mutation. Default = .05
* `κ`: recombination with probability (1-κ) during crossover. Default = 1.0
* `visualize`: generate scatter plot of particle locations on each iteration. Default=false
* `iter`: current iteration
* `progress`: show progress meter if true. Default = true

Constructor signature:

```@example
DE(;n_groups=4, priors=nothing, Np=num_parms(priors)*3,burnin=1000,
    α=.1, β=.1, ϵ=.001, σ=.05, bounds, visualize=false, iter=1, progress=false)
```
References:

* Ter Braak, C. J. A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces.

* Turner, B. M., Sederberg, P. B., Brown, S. D., & Steyvers, M. (2013). A method for efficiently sampling from distributions with correlated dimensions. Psychological methods, 18(3), 368.

* Turner, B. M., & Sederberg, P. B. (2012). Approximate Bayesian computation with differential evolution. Journal of Mathematical Psychology, 56(5), 375-385.
"""
mutable struct DE{T1} <: AbstractSampler
    n_groups::Int64
    Np::Int64
    burnin::Int64
    α::Float64
    β::Float64
    ϵ::Float64
    σ::Float64
    κ::Float64
    bounds::T1
    iter::Int64
    progress::Bool
end

function DE(;n_groups=4, priors=nothing, Np=num_parms(priors)*3, burnin=1000, α=.1, β=.1, ϵ=.001, σ=.05,
    κ=1.0, bounds, progress=true)
    return DE(n_groups, Np, burnin, α, β, ϵ, σ, κ, bounds, 1, progress)
end

"""
A model object containing the log likelihood function and prior distributions
* `priors`: prior distributions
* `model`: log likelihood function
* `names`: parameter names
"""
struct DEModel{F,L,T} <: AbstractModel where {F<:Function,L,T}
    priors::L
    model::F
    names::T
end

function DEModel(;priors, model, names=String.(keys(priors)))
    priors′ = values(priors)
    return DEModel(priors′, model, names)
 end

"""
Computes the number of parameters based on scalars or vectors
"""
function num_parms(priors)
    if isnothing(priors)
        error("Np undefined. Define Np=x in constructor or pass priors for default Np")
    end
    n = 0
    for p in priors
        length(p) == 1 ? n += 1 : (n += p[2])
    end
    return n
end

"""
* `θ`: a vector of parameters
* `samples`: a 2-dimensional array containing all acccepted proposals
* `accept`: proposal acceptance. 1: accept, 0: reject
* `weight`: particle weight based on model fit (currently posterior log likelihood)
* `lp`: a vector of log posterior probabilities associated with each accepted proposal
"""
mutable struct Particle{T}
    Θ::Vector{T}
    samples::Array{T,2}
    accept::Vector{Bool}
    weight::Float64
    lp::Vector{Float64}
end

Base.broadcastable(x::Particle) = Ref(x)

function Particle(;Θ=[.0], samples=Array{eltype(Θ),2}(undef,1,1),
    accept=Bool[],weight=0.0)
    Particle(Θ, samples, accept, weight, Float64[])
end
