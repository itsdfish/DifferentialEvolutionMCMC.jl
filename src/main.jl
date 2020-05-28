"""
Samples from the posterior distribution
* `model`: a model containing likelihood function with data and priors
* `de`: differential evolution object
* `n_iter`: number of iterations or samples

Function signature
```@example
    sample(model::DEModel, de::DE, n_iter::Int; kwargs...)
```
"""
sample(model::DEModel, de::DE, n_iter::Int; kwargs...) = _sample(model::DEModel, de::DE, n_iter::Int; stepfun=step!, kwargs...)

function _sample(model::DEModel, de::DE, n_iter::Int; progress=false, stepfun=step!, kwargs...)
    meter = Progress(n_iter)
    # initialize particles based on prior distribution
    groups = sample_init(model, de, n_iter)
    for iter in 1:n_iter
        de.iter = iter
        # explicitly pass groups so parallel works
        groups = stepfun(model, de, groups)
        progress ? next!(meter) : nothing
    end
    # convert to chain object
    chain = bundle_samples(model, de, groups, n_iter)
    return chain
end

"""
Samples from the posterior distribution with each group of particles on a seperarate thread for
the mutation and crossover steps.
* `model`: a model containing likelihood function with data and priors
* `de`: differential evolution object
* `n_iter`: number of iterations or samples

Function signature
```@example
    psample(model::DEModel, de::DE, n_iter::Int; kwargs...)
```
"""
function sample(model::DEModel, de::DE, ::MCMCThreads, n_iter::Int; progress=false, kwargs...)
    _sample(model::DEModel, de::DE, n_iter::Int; progress=progress, stepfun=pstep!, kwargs...)
end

"""
Perform a single step for DE-MCMC.
* `model`: model containing a likelihood function with data and priors
* `de`: DE-MCMC sampler object
* `groups`: Array of vectors of particles
"""
function step!(model::DEModel, de::DE, groups)
    rand() <= de.α ? migration!(de, groups) : nothing
    groups = mutate_crossover!(model, de, groups)
    store_samples!(de, groups)
    return groups
end

function pstep!(model::DEModel, de::DE, groups)
    rand() <= de.α ? migration!(de, groups) : nothing
    groups = pmutate_crossover!(model, de, groups)
    store_samples!(de, groups)
    return groups
end

_f(x...; y...) = nothing

"""
On a single core, selects between mutation and crossover step with probability β
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `group`: a vector of interacting particles (e.g. chains)
"""
function mutate_crossover!(model, de, groups)
    seeds = rand(UInt, length(groups))
    groups = map((group,s) -> mutate_or_crossover!(model, de, group, s), groups, seeds)
    return groups
end

"""
On multiple cores, selects between mutation and crossover step with probability β
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `groups`: a vector of interacting particles (e.g. chains)
"""
function pmutate_crossover!(model, de, groups)
    seeds = rand(UInt, de.n_groups)
    Threads.@threads for g in 1:de.n_groups
        group = mutate_or_crossover!(model, de, groups[g], seeds[g])
    end
    return groups
end

"""
Selects between mutation and crossover step with probability β
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `group`: a vector of interacting particles (e.g. chains)
"""
function mutate_or_crossover!(model, de, group, seed)
    Random.seed!(seed)
    rand() <= de.β ? mutation!(model, de, group) : crossover!(model, de, group)
    return group
end

"""
Converts group particles to a chain object capable of generating convergence diagnostics
and posterior summaries
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `groups`: a vector of groups of particles
* `n_iter`: number of iterations
"""
function bundle_samples(model::DEModel, de::DE, groups, n_iter)
    particles = vcat(groups...)
    Np = length(particles)
    Ns = n_iter- de.burnin
    all_names = get_names(model, particles[1])
    n_parms = length(all_names)
    Nnames = length(model.names)
    v = fill(0.0, Ns, n_parms, Np)
    for (c,p) in enumerate(particles)
        for s in 1:Ns
            temp = Float64[]
            for ni in 1:Nnames
                push!(temp, p.samples[s,ni]...)
            end
            push!(temp, p.accept[s], p.lp[s])
            v[s,:,c] = temp'
        end
    end
    return Chains(v, all_names, (internals=["acceptance", "lp"],); sorted=true)
end

"""
Creates vectors of particles and samples initial parameter values from priors
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `n_iter`: number of iterations
"""
function sample_init(model::DEModel, de::DE, n_iter)
    groups = [[Particle(Θ=sample_prior(model.priors)) for p in 1:de.Np]
        for c in 1:de.n_groups]
    for group in groups
        for p in group
            init_particle!(model, de, p, n_iter)
        end
    end
    return groups
end
