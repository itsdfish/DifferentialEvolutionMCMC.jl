"""
Mutates each particle
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `group`: a group of particles
"""
function mutation!(model, de, group)
    @unpack σ,bounds = de
    noise = Normal(0.0, σ)
    for pt in group
        # add noise to each parameter
        proposal = pt + noise
        # ensure that bounds are respected
        enforce_bounds!(bounds, proposal)
        # add Loglikelihood of prior to particle weight
        proposal.weight = priorlike(model, proposal) + model.model(proposal.Θ)
        # update whether proposal was accepted and add proposal if accepted
        update_particle!(de, pt, proposal)
    end
    return nothing
end
