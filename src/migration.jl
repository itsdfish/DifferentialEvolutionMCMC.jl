"""
select a subset of groups and swap one poorly particle from each selected groups
* `de`: differential evolution object
* `groups`: groups of particles
"""
function migration!(de, groups)
    # select groups for migration
    sub_group = select_groups(de, groups)
    # select particles within groups for migration
    p_idx,particles = select_particles(sub_group)
    # swap the particles so that p1->pn, p2 -> p1,..., pn -> pn-1
    shift_particles!(sub_group, p_idx, particles)
    return nothing
end

"""
select a subset of groups for migration and return their indices
* `de`: differential evolution object
* `groups`: groups of particles
"""
function select_groups(de, groups)
    N = rand(2:de.n_groups)
    sub_group = sample(groups, N, replace=false)
    return sub_group
end

"""
Select particles from groups for migration. Returns particle index and particles.
* `group_idx`: indices of groups for migration
* `groups`: groups of particles
"""
function select_particles(sub_group)
    Ng = length(sub_group)
    p_idx = fill(0, Ng)
    particles = Vector{eltype(sub_group[1])}(undef,Ng)
    for (i,g) in enumerate(sub_group)
        p_idx[i],particles[i] = select_particle(g)
    end
    return p_idx,particles
end

"""
Select particle from a single chain inversely proportional to its weight
* `group`: a group of particles
"""
function select_particle(group)
    w = map(x -> x.weight, group)
    θ = exp.(-w)/sum(exp.(-w))
    idx = sample(1:length(group), Weights(θ))
    # if numberical error occurs, select the worst particle index (lower is worse)
    any(isnan, θ) ? idx = findmin(w)[2] : nothing
    return idx,group[idx]
end

"""
Swap the particles so that p1->pn, p2 -> p1,..., pn -> pn-1 where
pi is the particle belonging to the ith group
* `group_idx`: indices of groups for migration
* `p_idx`: particle index
* `particles`: particle objects representing position in parameter space
* `groups`: groups of particles
"""
function shift_particles!(sub_group, p_idx, particles)
    # perform a circular shift
    particles = circshift(particles, 1)
    # assign shifted particles to the new chain
    for (g,j,p) in zip(sub_group, p_idx, particles)
        g[j] = p
    end
end
