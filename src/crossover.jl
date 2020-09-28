"""
Performs crossover step for each particle pt in the chain
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `group`: a group of particles
"""
function crossover!(model, de, group)
    for pt in group
        if rand() ≤ de.θsnooker
            # generate the proposal
            proposal,log_adj = snooker_update!(de, pt, group)
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
    return nothing
end

"""
Generate proposal according to θ' = θt + γ1(θm − θn) + γ2(θb − θt) + b
γ2=0 after burnin
* `de`: differential evolution object
* `Pt`: current particle
* `group`: a group of particles
"""
function random_gamma(de, Pt, group)
    Np = length(Pt.Θ)
    # select the base particle θb
    Pb = select_base(group)
    # group without Pb
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pm,Pn = sample(group_diff, 2, replace=false)
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
Generate proposal according to θ' = θt + γ(θm − θn) + b
where γ = 2.38
* `de`: differential evolution object
* `Pt`: current particle
* `group`: a group of particles
"""
function fixed_gamma(de, Pt, group)
    Np = length(Pt.Θ)
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pm,Pn = sample(group_diff, 2, replace=false)
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
Generate proposal according to θ' = θt + γ(θm − θn) + b
where γ = 2.38/√(2d) where d is the number of parameters
* `de`: differential evolution object
* `Pt`: current particle
* `group`: a group of particles
"""
function variable_gamma(de, Pt, group)
    Np = length(Pt.Θ)
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pm,Pn = sample(group_diff, 2, replace=false)
    γ = 2.38 / sqrt(2 * Np)
    # sample noise for each parameter
    b = Uniform(-de.ϵ, de.ϵ)
    # compute proposal value
    Θp = Pt + γ * (Pm - Pn) + b
    # reset each parameter to previous value with probability (1-κ)
    recombination!(de, Pt, Θp)
    return Θp
end

function snooker_update!(de, Pt, group)
    Np = length(Pt.Θ)
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pz,Pm,Pn = sample(group_diff, 3, replace=false)
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
    adj1 = norm(Θp - Pz)^(Np - 1)
    adj2 = norm(Pt - Pz)^(Np - 1)
    return Θp,log(adj1/adj2)
end

"""
Selects base particle θb with probability proportional to weight
* `group`: a group of particles
"""
function select_base(group)
    w = map(x -> x.weight, group)
    θ = exp.(w) / sum(exp.(w))
    p = sample(group, Weights(θ))
    return p
end

"""
Resets parameters of proposal to previous value with probability
(1-κ).
* `de`: differential evolution object
* `pt`: current partical
* `pp`: proposal particle
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