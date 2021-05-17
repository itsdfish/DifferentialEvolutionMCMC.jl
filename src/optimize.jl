"""
Samples from the posterior distribution
* `model`: a model containing likelihood function with data and priors
* `de`: differential evolution object
* `n_iter`: number of iterations or samples
* `progress`: show progress (default false)

Function signature
```@example
    sample(model::DEModel, de::DE, n_iter::Int; progress=false, kwargs...)
```
"""
optimize(model::DEModel, de::DE, n_iter::Int; progress=false, kwargs...) = _optimize(model::DEModel, de::DE, n_iter::Int; progress, stepfun=step!, kwargs...)

function _optimize(model::DEModel, de::DE, n_iter::Int; progress=false, stepfun=step!, kwargs...)
    meter = Progress(n_iter)
    # initialize particles based on prior distribution
    groups = sample_init(model, de, n_iter)
    for iter in 1:n_iter
        de.iter = iter
        # explicitly pass groups so parallel works
        groups = stepfun(model, de, groups)
        progress ? next!(meter) : nothing
    end
    return vcat(groups...)
end

"""
Samples from the posterior distribution with each group of particles on a seperarate thread for
the mutation and crossover steps.
* `model`: a model containing likelihood function with data and priors
* `de`: differential evolution object
* `MCMCThreads`: pass MCMCThreads() object to run on multiple threads
* `n_iter`: number of iterations or samples
* `progress`: show progress (default false)

Function signature
```@example
    sample(model::DEModel, de::DE, ::MCMCThreads, n_iter::Int; progress=false, kwargs...)
```
"""
function optimize(model::DEModel, de::DE, ::MCMCThreads, n_iter::Int; progress=false, kwargs...)
    _optimize(model::DEModel, de::DE, n_iter::Int; progress, stepfun=pstep!, kwargs...)
end