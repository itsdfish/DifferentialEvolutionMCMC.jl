const DEMCMC = DifferentialEvolutionMCMC

"""
    sample(model::DEModel, de::DE, n_iter::Int; progress=false, kwargs...)

Samples from the posterior distribution.

# Arguments

- `model`: a model containing likelihood function with data and priors
- `de`: differential evolution object
- `n_iter`: number of iterations or samples

# Keywords

- `progress=false`: show progress of sampler
- `kwargs...`: optional keyword arguments
"""
sample(model::DEModel, de::DE, n_iter::Int; progress=false, kwargs...) = _sample(model::DEModel, de::DE, n_iter::Int; progress, stepfun=step!, kwargs...)

function _sample(model::DEModel, de::DE, n_iter::Int; progress=false, stepfun=step!, kwargs...)
    meter = Progress(n_iter)
    # initialize particles based on prior distribution
    groups = sample_init(model, de, n_iter)
    for iter in 1:n_iter
        de.iter = iter + de.n_initial
        # explicitly pass groups so parallel works
        groups = stepfun(model, de, groups)
        progress ? next!(meter) : nothing
    end
    # convert to chain object
    chain = bundle_samples(model, de, groups, n_iter)
    return chain
end

"""
    sample(model::DEModel, de::DE, ::MCMCThreads, n_iter::Int; progress=false, kwargs...)

Samples from the posterior distribution with each group of particles on a seperarate thread for
the mutation and crossover steps.

# Arguments

- `model`: a model containing likelihood function with data and priors
- `de`: differential evolution object
- `MCMCThreads`: pass MCMCThreads() object to run on multiple threads
- `n_iter`: number of iterations or samples

# Keywords

- `progress=false`: show progress of sampler
- `kwargs...`: optional keyword arguments
"""
function sample(model::DEModel, de::DE, ::MCMCThreads, n_iter::Int; progress=false, kwargs...)
    _sample(model::DEModel, de::DE, n_iter::Int; progress, stepfun=pstep!, kwargs...)
end

"""
    step!(model::DEModel, de::DE, groups)

Perform a single step for DE-MCMC.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: DE-MCMC sampler object
- `groups`: Array of vectors of particles
"""
function step!(model::DEModel, de::DE, groups)
    rand() <= de.α ? migration!(de, groups) : nothing
    groups = update!(model, de, groups)
    store_samples!(de, groups)
    return groups
end

"""
    pstep!(model::DEModel, de::DE, groups)

Perform a single step for DE-MCMC with each particle group on a separate thread.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: DE-MCMC sampler object
- `groups`: Array of vectors of particles
"""
function pstep!(model::DEModel, de::DE, groups)
    rand() <= de.α ? migration!(de, groups) : nothing
    groups = p_update!(model, de, groups)
    store_samples!(de, groups)
    return groups
end

"""
    p_update!(model, de, groups)

Multithreaded update of particles. Particles are updated by block or simultaneously.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `group`: a vector of interacting particles (e.g. chains)
"""
function p_update!(model, de, groups)
    seeds = rand(UInt, length(groups))
    if de.blocking_on(de)
        Threads.@threads for g in 1:de.n_groups
            group = block_update!(model, de, groups[g], seeds[g])
        end
        return groups
    else
        Threads.@threads for g in 1:de.n_groups
            group = mutate_or_crossover!(model, de, groups[g], seeds[g])
        end
        return groups
    end
end

"""
    update!(model, de, groups)(model, de, groups)

Particles are updated by block or simultaneously on a single thread.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `group`: a vector of interacting particles (e.g. chains)
"""
function update!(model, de, groups)
    if de.blocking_on(de)
        return map(g -> block_update!(model, de, g), groups)
    else
        return map(g -> mutate_or_crossover!(model, de, g), groups)
    end
end

function block_update!(model, de, group, seed)
    Random.seed!(seed)
    return block_update!(model, de, group)
end

function block_update!(model, de, group)
    for block_idx in de.blocks
        mutate_or_crossover!(model, de, group, block_idx)
    end
    return group
end

"""
    mutate_or_crossover!(model, de, group, seed::Number)

Selects between mutation and crossover step with probability β.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `group`: a vector of interacting particles (e.g. chains)
- `seed::Number`: RNG seed
"""
function mutate_or_crossover!(model, de, group, seed::Number)
    Random.seed!(seed)
    mutate_or_crossover!(model, de, group)
    return group
end

function mutate_or_crossover!(model, de, group)
    rand() <= de.β ? mutation!(model, de, group) : crossover!(model, de, group)
    return group
end

function mutate_or_crossover!(model, de, group, block_idx)
    rand() <= de.β ? mutation!(model, de, group) : crossover!(model, de, group, block_idx)
    return group
end

"""
    bundle_samples(model::DEModel, de::DE, groups, n_iter)

Converts group particles to a chain object capable of generating convergence diagnostics
and posterior summaries.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `groups`: a vector of groups of particles
- `n_iter`: number of iterations
"""
function bundle_samples(model::DEModel, de::DE, groups, n_iter)
    @unpack burnin, discard_burnin = de
    particles = vcat(groups...)
    Np = length(particles)
    Ns = discard_burnin ? n_iter - burnin : n_iter
    offset = discard_burnin ? burnin : 0
    all_names = get_names(model, particles[1])
    n_parms = length(all_names)
    n_names = length(model.names)
    v = fill(0.0, Ns, n_parms, Np)
    for (c,p) in enumerate(particles)
        for s in 1:Ns
            sΔ = s + offset
            temp = Float64[]
            for ni in 1:n_names
                push!(temp, p.samples[sΔ,ni]...)
            end
            push!(temp, p.accept[sΔ], p.lp[sΔ])
            v[s,:,c] = temp'
        end
    end
    chains = Chains(v, all_names, (parameters = [model.names...],
        internals = ["acceptance", "lp"]))
    return chains
end

"""
    sample_init(model::DEModel, de::DE, n_iter)

Creates vectors of particles and samples initial parameter values from priors.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `n_iter`: number of iterations
"""
function sample_init(model::DEModel, de::DE, n_iter)
    groups = [[Particle(Θ=model.sample_prior()) for p in 1:de.Np]
        for c in 1:de.n_groups]
    for group in groups
        for p in group
            init_particle!(model, de, p, n_iter)
        end
    end
    return groups
end
