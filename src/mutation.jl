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
        # add Loglikelihood of prior to particle weight
        compute_posterior!(de, model, proposal)
        # update whether proposal was accepted and add proposal if accepted
        update_particle!(de, pt, proposal)
    end
    return nothing
end