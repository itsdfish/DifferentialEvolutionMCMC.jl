@safetestset "rastrigin" begin 
    using Test, DifferentialEvolutionMCMC, Random, Distributions
    import DifferentialEvolutionMCMC: minimize!

    Random.seed!(514)

    priors = (
        x = (Uniform(-5, 5), 2),
    )

    bounds = ((-5.0,5.0),)

    function rastrigin(data, x)
        A = 10.0
        n = length(x)
        y = A * n
        for  i in 1:n
            y +=  + x[i]^2 - A * cos(2 * π * x[i])
        end
        return y 
    end

    model = DEModel(; priors, model=rastrigin, data=nothing)

    de = DE(bounds=bounds, Np=6, n_groups=1, update_particle! = minimize!,
        evaluate_fitness! = evaluate_fun!)
    n_iter = 10000
    particles = optimize(model, de, MCMCThreads(), n_iter, progress=true);
    results = get_optimal(de, model, particles)
    @test results[2] ≈ 0.0 atol = 1e-8
end

@safetestset "maximum likelihood estimation" begin
    cd(@__DIR__)
    using Test, DifferentialEvolutionMCMC, Random, Distributions
    import DifferentialEvolutionMCMC: maximize!

    Random.seed!(50514)

    # selects a random starting point for each parameter
    priors = (
        μ = (Uniform(-50, 50),),
        σ = (Uniform(0, 50),),
    )

    # bounds of parameters, same order as above
    bounds = (
        (-Inf, Inf), 
        (0.1, Inf), 
    )
    # a function to optimize, data argument is required but ignored if not needed
    function loglike(data, μ, σ)
        return logpdf.(Normal(μ, σ), data) |> sum
    end

    data = rand(Normal(0, 1), 100)
    # a model object containing prior, model function and data
    model = DEModel(; priors, model=loglike, data)
    # optimization settings, which minimizes the function
    de = DE(;bounds, Np=6, n_groups=1, update_particle! = maximize!,
        evaluate_fitness! = evaluate_fun!)
    # iterations of the optimizer
    n_iter = 10000
    # run the optimization algorithm
    particles = optimize(model, de, MCMCThreads(), n_iter, progress=true);
    # extract the optimal parameters
    parms,LL = get_optimal(de, model, particles)
    @test mean(data) ≈ parms.μ atol = .0001
    @test std(data; corrected=false) ≈ parms.σ atol = .0001
end