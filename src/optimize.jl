"""
    optimize(model::DEModel, de::DE, n_iter::Int; progress=false, kwargs...)

Finds optimal set of parameters.

# Arguments

- `model`: a model containing likelihood function with data and priors
- `de`: differential evolution object
- `n_iter`: number of iterations or samples

# Keywords

- `progress=false`: show progress of algorithm
- `kwargs...`: optional keyword arguments
"""
optimize(model::DEModel, de::DE, n_iter::Int; progress = false, kwargs...) =
    _optimize(model::DEModel, de::DE, n_iter::Int; progress, stepfun = step!, kwargs...)

function _optimize(
    model::DEModel,
    de::DE,
    n_iter::Int;
    progress = false,
    stepfun = step!,
    kwargs...
)
    meter = Progress(n_iter)
    # initialize particles based on prior distribution
    groups = sample_init(model, de, n_iter)
    for iter = 1:n_iter
        de.iter = iter
        # explicitly pass groups so parallel works
        groups = stepfun(model, de, groups)
        progress ? next!(meter) : nothing
    end
    return vcat(groups...)
end

"""
    optimize(model::DEModel, de::DE, ::MCMCThreads, n_iter::Int; progress=false, kwargs...)

Finds optimal set of parameters.

# Arguments

- `model`: a model containing likelihood function with data and priors
- `de`: differential evolution object
- `MCMCThreads`: pass MCMCThreads() object to run on multiple threads
- `n_iter`: number of iterations or samples

# Keywords

- `progress=false`: show progress of algorithm
- `kwargs...`: optional keyword arguments
"""
function optimize(
    model::DEModel,
    de::DE,
    ::MCMCThreads,
    n_iter::Int;
    progress = false,
    kwargs...
)
    _optimize(model::DEModel, de::DE, n_iter::Int; progress, stepfun = pstep!, kwargs...)
end
