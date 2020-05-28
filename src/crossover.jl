"""
Performs crossover step for each particle pt in the chain
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `group`: a group of particles
"""
function crossover!(model, de, group)
    for pt in group
        # generate the proposal
        proposal = generate_proposal(de, pt, group)
        # compute the weight of the proposal: prior loglikelihood + data loglikelihood
        proposal.weight = priorlike(model, proposal) + model.model(proposal.Θ)
        # accept proposal according to Metropolis-Hastings rule
        update_particle!(de, pt, proposal)
    end
end

"""
Generate proposal according to θ' = θt + γ1(θm − θn) + γ2(θb − θt) + b
γ2=0 after burnin
* `de`: differential evolution object
* `Pt`: current particle
* `group`: a group of particles
"""
# 2.38/sqrt(2d)
function generate_proposal(de, Pt, group)
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
    de.iter > de.burnin ? γ₂=0.0 : γ₂=rand(Uniform(.5, 1))
    # sample noise for each parameter
    b = Uniform(-de.ϵ, de.ϵ)
    # compute proposal value
    Θp = Pt + γ₁*(Pm - Pn) + γ₂*(Pb - Pt) + b
    # enforce the parameter boundaries
    enforce_bounds!(de.bounds, Θp)
    # reset each parameter to previous value with probability (1-κ)
    recombination!(de, Pt, Θp)
    return Θp
end

"""
Selects base particle θb with probability proportional to weight
* `group`: a group of particles
"""
function select_base(group)
    w = map(x->x.weight, group)
    θ = exp.(w)/sum(exp.(w))
    p = sample(group, Weights(θ))
    return p
end

"""
Resets parameters of proposal to previous value with probability
(1-κ).
* `pp`: proposal particle
* `pt`: current partical
"""
function recombination!(de, pt, pp)
    de.κ == 1.0 ? (return nothing) : nothing
    N = length(pt.Θ)
    for i in 1:N
        if isa(pp.Θ[i], Array)
            recombination!(de, pt, pp, i)
        else
            rand() <= (1-de.κ) ? pp.Θ[i] = pt.Θ[i] : nothing
        end
    end
    return nothing
end

# Handles elements within an array
function recombination!(de, pt, pp, idx)
    N = length(pt.Θ[idx])
    for i in 1:N
        rand() <= (1-de.κ) ? pp.Θ[idx][i] = pt.Θ[idx][i] : nothing
    end
    return nothing
end
