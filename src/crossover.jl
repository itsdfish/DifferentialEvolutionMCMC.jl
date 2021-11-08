"""
    crossover!(model, de, group)

Performs crossover step for each particle pt in the chain

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `group`: a group of particles
"""
function crossover!(model, de, group)
    for pt in group
        crossover!(model, de, group, pt)
    end
    return nothing
end

"""
    crossover!(model, de, group, pt::Particle)

Performs crossover step for for a given particle pt

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `pt`: a target particle to be updated
"""
function crossover!(model, de, group, pt::Particle)
    if rand() ≤ de.θsnooker
        # generate the proposal
        proposal,pz = snooker_update!(de, pt, group)
        log_adj = adjust_loglike(pt, proposal, pz)
        # compute the weight of the proposal: prior loglikelihood + data loglikelihood
        de.evaluate_fitness!(de, model, proposal)
        # accept proposal according to Metropolis-Hastings rule
        de.update_particle!(de, pt, proposal, log_adj)
    else
        # generate the proposal
        proposal = de.generate_proposal(de, pt, group)
        # compute the weight of the proposal: prior loglikelihood + data loglikelihood
        de.evaluate_fitness!(de, model, proposal)
        # accept proposal according to Metropolis-Hastings rule
        de.update_particle!(de, pt, proposal)
    end
end

"""
    crossover!(model, de, group)

Performs crossover step for each particle pt in the chain

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `group`: a group of particles
- `block`: a vector of boolean values indicating which parameters to update
"""
function crossover!(model, de, group, block)
    for pt in group
        crossover!(model, de, group, pt, block)
    end
    return nothing
end

"""
    crossover!(model, de, group, pt::Particle, block)

Performs crossover step for a given particle pt.

# Arguments

- `model`: model containing a likelihood function with data and priors
- `de`: differential evolution object
- `group`: a group of particles
- `block`: a vector of boolean values indicating which parameters to update
"""
function crossover!(model, de, group, pt::Particle, block)
    if rand() ≤ de.θsnooker
        # generate the proposal
        proposal,pz = snooker_update!(de, pt, group)
        reset!(proposal, pt, block)
        log_adj = adjust_loglike(pt, proposal, pz)
        # compute the weight of the proposal: prior loglikelihood + data loglikelihood
        de.evaluate_fitness!(de, model, proposal)
        # accept proposal according to Metropolis-Hastings rule
        de.update_particle!(de, pt, proposal, log_adj)
    else
        # generate the proposal
        proposal = de.generate_proposal(de, pt, group)
        reset!(proposal, pt, block)
        # compute the weight of the proposal: prior loglikelihood + data loglikelihood
        de.evaluate_fitness!(de, model, proposal)
        # accept proposal according to Metropolis-Hastings rule
        de.update_particle!(de, pt, proposal)
    end
end

"""
    resample(de, group, n, replace)

Sample a random particle from previously accepted values for snooker update.

# Arguments

- `de`: differential evolution object
- `group`: a group of particles
- `n`: number of particles to sample
- `replace`: sample with replacement if true
"""
function resample(de, group, n, replace)
    P′ = Vector{eltype(group)}(undef,n)
    mx_idx = de.iter - 1
    idx = sample_indices(de.samples, mx_idx, n; replace)
    for i in 1:n 
        P′[i] = Particle(;Θ=de.samples[idx[i][1],:,idx[i][2]])
    end
    return P′
end

sample_indices(x, ub, n; replace) = @views sample(CartesianIndices(x[1:ub,1,:]), n; replace)

"""
    sample(de::DE, group_diff, n, replace) 

Sample a random particle.

# Arguments

- `de`: differential evolution object
- `group`: a group of particles
- `n`: number of particles to sample
- `replace`: sample with replacement if true
"""
function sample(de::DE, group_diff, n, replace) 
    return sample(group_diff, n; replace)
end

"""
    random_gamma(de, Pt, group)

Generate proposal according to θ' = θt + γ1(θm − θn) + γ2(θb − θt) + b.
γ2=0 after burnin

# Arguments

- `de`: differential evolution object
- `Pt`: current particle
- `group`: a group of particles
"""
function random_gamma(de, Pt, group)
    Np = length(Pt.Θ)
    # select the base particle θb
    Pb = select_base(group)
    # group without Pb
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pm,Pn = de.sample(de, group_diff, 2, false) 
    # sample gamma weights
    γ₁ = rand(Uniform(.5, 1))
    # set γ₂ = 0 after burnin
    γ₂ = de.iter > de.burnin ? 0.0 : rand(Uniform(.5, 1))
    # sample noise for each parameter
    b = Uniform(-de.ϵ, de.ϵ)
    # compute proposal value
    Θp = Pt + γ₁ * (Pm - Pn) + γ₂ * (Pb - Pt) + b
    # reset each parameter to previous value with probability (1-κ)
    recombination!(de, Pt, Θp)
    return Θp
end

"""
    fixed_gamma(de, Pt, group)

Generate proposal according to θ' = θt + γ(θm − θn) + b
where γ = 2.38.

# Arguments

- `de`: differential evolution object
- `Pt`: current particle
- `group`: a group of particles
"""
function fixed_gamma(de, Pt, group)
    Np = length(Pt.Θ)
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pm,Pn = de.sample(de, group_diff, 2, false) 
    # sample gamma weights
    γ = 2.38
    # sample noise for each parameter
    b = Uniform(-de.ϵ, de.ϵ)
    # compute proposal value
    Θp = Pt + γ * (Pm - Pn) + b
    # reset each parameter to previous value with probability (1-κ)
    recombination!(de, Pt, Θp)
    return Θp
end

"""
    variable_gamma(de, Pt, group)

Generate proposal according to θ' = θt + γ(θm − θn) + b
where γ = 2.38/√(2d) where d is the number of parameters

# Arguments 

- `de`: differential evolution object
- `Pt`: current particle
- `group`: a group of particles
"""
function variable_gamma(de, Pt, group)
    Np = length(Pt.Θ)
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pm,Pn = de.sample(de, group_diff, 2, false) 
    γ = 2.38 / sqrt(2 * Np)
    # sample noise for each parameter
    b = Uniform(-de.ϵ, de.ϵ)
    # compute proposal value
    Θp = Pt + γ * (Pm - Pn) + b
    # reset each parameter to previous value with probability (1-κ)
    recombination!(de, Pt, Θp)
    return Θp
end

"""
    snooker_update!(de, Pt, group)

Performs snooker update during crossover

# Arguments 

- `de`: differential evolution object
- `Pt`: current particle
- `group`: a group of particles
"""
function snooker_update!(de, Pt, group)
    Pz,Pm,Pn = de.sample(de, group, 3, false) 
    Pd = Pt - Pz
    Pr1 = project(Pm, Pd)
    Pr2 = project(Pn, Pd)
    γ = rand(Uniform(1.2, 2.2))
    # sample noise for each parameter
    b = Uniform(-de.ϵ, de.ϵ)
    # compute proposal value
    Θp = Pt + γ * (Pr1 - Pr2) + b
    # reset each parameter to previous value with probability (1-κ)
    recombination!(de, Pt, Θp)
    return Θp,Pz
end

"""
    adjust_loglike(Pt, proposal, Pz)
    
The adjusted log likelihood component for a snooker update. 

- `Pt`: the target particle from iteration n - 1
- `proposal`: the proposal particle on iteration n 
- `Pz`: the particle formed by the projection of particles m and n 
"""
function adjust_loglike(Pt, proposal, Pz)
    Np = length(Pt.Θ)
    adj1 = norm(proposal - Pz)^(Np - 1)
    adj2 = norm(Pt - Pz)^(Np - 1)
    return log(adj1 / adj2)
end

"""
    select_base(group)

Selects base particle θb with probability proportional to weight.

- `group`: a group of particles
"""
function select_base(group)
    w = map(x -> x.weight, group)
    θ = exp.(w) / sum(exp.(w))
    p = sample(group, Weights(θ))
    return p
end

"""
    recombination!(de, pt::Particle, pp::Particle)

Resets parameters of proposal to previous value with probability
(1-κ).

- `de`: differential evolution object
- `pt`: current partical
- `pp`: proposal particle
"""
function recombination!(de, pt::Particle, pp::Particle)
    de.κ == 1.0 ? (return nothing) : nothing
    N = length(pt.Θ)
    for i in 1:N
        if isa(pp.Θ[i], Array)
            recombination!(de, pt.Θ[i], pp.Θ[i])
        else
            pp.Θ[i] = rand() <= (1 - de.κ) ?  pt.Θ[i] : pp.Θ[i]
        end
    end
    return nothing
end

# Handles elements within an array
function recombination!(de, Θt, Θp)
    N = length(Θt)
    for i in 1:N
        Θp[i] = rand() <= (1 - de.κ) ? Θt[i] : Θp[i]
    end
    return nothing
end

"""
    reset!(p1::Particle, p2::Particle, idx)

During block updates, all parameters are updated. This function resets 
parameters in the proposal `p1` with values from the previous particle. 

# Arguments

`p1`: proposal particle which will have values reset 
`p2`: previous particle
`idx`: boolean vector indicating which values are updated in block. False 
values are reset. 
"""
reset!(p1::Particle, p2::Particle, idx) = reset!(p1.Θ, p2.Θ, idx)

function reset!(Θ1, Θ2, idx)
    for i in 1:length(Θ1)
        reset!(Θ1, Θ2, idx[i], i)
    end
    return nothing
end

function reset!(Θ1, Θ2, idx::Bool, i)
    !idx ? Θ1[i] = Θ2[i] : nothing 
    return nothing
end

function reset!(Θ1, Θ2, idx::Array{Bool,N}, i) where {N}
    return reset!(Θ1[i], Θ2[i], idx)
end