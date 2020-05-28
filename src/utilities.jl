"""
Initializes values for a particle
* `model`: model containing a likelihood function with data and priors
* `de`: differential evolution object
* `p`: a particle
* `n_iter`: the number of iterations
"""
function init_particle!(model, de, p, n_iter)
    N = n_iter - de.burnin
    p.samples = typeof(p.samples)(undef,N,length(p.Θ))
    p.accept = fill(false, N)
    p.weight = priorlike(model, p) + model.model(p.Θ)
    p.lp = fill(0.0, N)
    return nothing
end

function priorlike(model, p::Particle{Array{T,1}}) where {T<:Real}
    return sum(logpdf.(model.priors, p.Θ))
end

function priorlike(model, p)
    LL = 0.0
    for (pr,θ) in zip(model.priors, p.Θ)
        LL += get_LL(pr[1], θ)
    end
    return LL
end

# handles arbitrary distribution types. Not flexible enough otherwise
get_LL(d::MultivariateDistribution, x::Vector) = logpdf(d, x)
get_LL(d::UnivariateDistribution, x::Real) = logpdf(d, x)
get_LL(d, x) = loglikelihood(d, x)

"""
Metropolis-Hastings proposal selection
Note: assumes weights are posterior log likelihoods
* `proposal`: weight of proposal e.g. posterior log likelihood
* `current`: weight of current value e.g. posterior log likelihood
"""
function accept(proposal, current)
    p = min(1.0, exp(proposal-current))
    rand() <= p ? (return true) : (return false)
end

"""
replaces values outside of bounds with the boundary
* `b`: boundary (lowerbound,upperbound)
* `θ`: a parameter value
"""
function enforce_bounds(b, θ::Real)
    θ < b[1] ? (return b[1]) : nothing
    θ > b[2] ? (return b[2]) : nothing
    return θ
end

function enforce_bounds(b, Θ)
    map(θ->enforce_bounds(b, θ), Θ)
end

function enforce_bounds!(bounds, p)
    i = 0
    for (b,θ) in zip(bounds, p.Θ)
        i += 1
        p.Θ[i] = enforce_bounds(b, θ)
    end
    return nothing
end

"""
Returns parameters names.
* `model`: model containing a likelihood function with data and priors
* `p`: a particle
"""
function get_names(model, p)
    N = length.(p.Θ)
    parm_names = fill("", sum(N))
    cnt = 0
    for (k,n) in zip(model.names, N)
        if n > 1
            for i in 1:n
                cnt += 1
                parm_names[cnt] = string(k, "[",i,"]")
            end
        else
            cnt += 1
            parm_names[cnt] = string(k)
        end
    end
    push!(parm_names, "acceptance", "lp")
    return parm_names
end

"""
Store samples after burnin period
Selects between mutation and crossover step with probability β
* `de`: differential evolution object
* `groups`: groups of particles
"""
function store_samples!(de, groups)
    de.iter <= de.burnin ? (return) : nothing
    i = de.iter - de.burnin
    for group in groups
        for p in group
            add_sample!(p, i)
        end
    end
    return nothing
end

function add_sample!(p::Particle{T}, i) where {T<:Real}
    p.samples[i,:] = p.Θ'
end

function add_sample!(p, i)
    p.samples[i,:] = p.Θ
end

function sample_prior(priors)
    p = [rand(p...) for p in priors]
    t = findtype(p)
    return returntype(t, p)
end

findtype(p) = Union{unique(typeof.(p))...}
returntype(t, p) = t[p...]

"""
Update particle based on Metropolis-Hastings rule.
* `de`: differential evolution object
* `groups`: groups of particles
"""
function update_particle!(de, current, proposal)
    @unpack iter,burnin = de
    i = iter - burnin
    accepted = accept(proposal.weight, current.weight)
    if accepted
        current.Θ = proposal.Θ
        current.weight = proposal.weight
    end
    if iter > burnin
         current.accept[i] = accepted
         current.lp[i] = current.weight
     end
    return nothing
end


# Type-stable arithmatic operations for Union{Array{Float64,1},Float64} types (which return Any otherwise)
import Base: +, - ,*

function +(x::Particle, y::Particle)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] + y.Θ[i]
    end
    return Particle(Θ=z)
end

+(x::Real, y::Particle) = +(y, x)

function +(x::Particle, y::Real)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .+ y
    end
    return Particle(Θ=z)
end

function +(x::Particle, d::Distribution)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .+′ draw(d, x.Θ[i])
    end
    return Particle(Θ=z)
end

function draw(d, v::Float64)
    return rand(d)
end

function draw(d, v)
    return rand(d, length(v))
end

function *(x::Particle, y::Particle)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .*′ y.Θ[i]
    end
    return Particle(Θ=z)
end

*(x::Real, y::Particle) = *(y, x)

function *(x::Particle, y::Real)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .*′ y
    end
    return Particle(Θ=z)
end

*(x::Array{<:Real,1}, y::Particle) = *(y, x)

function *(x::Particle, y::Array{<:Real,1})
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .* y[i]
    end
    return Particle(Θ=z)
end

function -(x::Particle, y::Particle)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] - y.Θ[i]
    end
    return Particle(Θ=z)
end

-(x::Real, y::Particle) = -(y, x)

function -(x::Particle, y::Real)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .- y
    end
    return Particle(Θ=z)
end

# arithmatic methods for hanlding discrete parameters
*′(x, y) = x*y
*′(x::Int64, y::Float64) = Int(round(x*y))
*′(x::Float64, y::Int64) = Int(round(x*y))
*′(x::Array{Int64,N}, y::Float64) where {N} = @. Int(round(x*y))
*′(x::Float64, y::Array{Int64,N}) where {N} = @. Int(round(x*y))
+′(x, y) = x + y
+′(x::Int64, y::Float64) = Int(round(x+y))
+′(x::Float64, y::Int64) = Int(round(x+y))
+′(x::Array{Int64,N}, y::Float64) where {N} = @. Int(round(x+y))
+′(x::Float64, y::Array{Int64,N}) where {N} = @. Int(round(x+y))
