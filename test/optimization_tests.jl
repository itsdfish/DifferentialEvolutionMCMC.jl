@safetestset "rastrigin" begin 
    using Test, DifferentialEvolutionMCMC, Random, Distributions
    import DifferentialEvolutionMCMC: minimize!

    Random.seed!(78454111)

    function sample_prior()
       return [rand(Uniform(-5, 5), 2)]
    end

    bounds = ((-5.0,5.0),)
    names = (:x,)

    function rastrigin(data, x)
        A = 10.0
        n = length(x)
        y = A * n
        for  i in 1:n
            y +=  + x[i]^2 - A * cos(2 * π * x[i])
        end
        return y 
    end

    model = DEModel(; 
        sample_prior, 
        loglike = rastrigin, 
        data = nothing,
        names
    )

    de = DE(;
        sample_prior,
        bounds,
        Np = 6, 
        n_groups = 1, 
        update_particle! = minimize!,
        evaluate_fitness! = evaluate_fun!
    )

    n_iter = 10_000
    particles = optimize(model, de, n_iter, progress=true);
    results = get_optimal(de, model, particles)
    @test results[2] ≈ 0.0 atol = 1e-8
end

@safetestset "maximum likelihood estimation" begin
    using Test, DifferentialEvolutionMCMC, Random, Distributions
    import DifferentialEvolutionMCMC: maximize!

    Random.seed!(50514)

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

    function prior_loglike(μ, σ)
        LL = 0.0
        LL += logpdf(Normal(0, 1), μ)
        LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
        return LL
    end
    
    function sample_prior()
        μ = rand(Normal(0, 1))
        σ = rand(truncated(Cauchy(0, 1), 0, Inf))
        return [μ,σ]
    end
    
    names = (:μ,:σ)
    
    model = DEModel(; 
        sample_prior, 
        loglike, 
        data,
        names
    )
    
    de = DE(;
        sample_prior,
        bounds, 
        burnin = 1000, 
        Np = 6, 
        n_groups = 1, 
        update_particle! = maximize!,
        evaluate_fitness! = evaluate_fun!
    )

    n_iter = 10000
    # # run the optimization algorithm
    particles = optimize(model, de, MCMCThreads(), n_iter, progress=true);
    # extract the optimal parameters
    parms,LL = get_optimal(de, model, particles)
    @test mean(data) ≈ parms.μ atol = .0001
    @test std(data; corrected=false) ≈ parms.σ atol = .0001
end