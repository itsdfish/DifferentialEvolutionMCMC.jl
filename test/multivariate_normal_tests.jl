@safetestset "multivariate normal" begin
    using DifferentialEvolutionMCMC, Test, Random, Distributions
    using MCMCChains, LinearAlgebra
    Random.seed!(50514)

    # number of variables 
    n_μ = 30
    # number of observations per variable 
    n_d = 100
    # μ parameters
    μs = fill(0.0, n_μ)
    # data
    data = rand(MvNormal(μs, 1.0 * I), n_d)

    # function for initial values
    function sample_prior()
        μ = rand(Normal(0, 1), n_μ)
        σ = rand(truncated(Cauchy(0, 1), 0, Inf))
        return as_union([μ,σ])
    end

    # returns prior log likelihood
    function prior_loglike(μ, σ)
        LL = 0.0
        LL += sum(logpdf.(Normal(0, 1), μ))
        LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
        return LL
    end

    # likelihood function 
    function loglike(data, μs, σ)
        return sum(logpdf(MvNormal(μs, σ^2 * I), data))
    end

    # upper and lower bounds of parameters
    bounds = ((-Inf,Inf),(0.0,Inf))
    # parameter names 
    names = (:μ,:σ)

    # model object
    model = DEModel(; 
        sample_prior, 
        prior_loglike, 
        loglike, 
        data,
        names
    )

    # DEMCMC sampler 
    de = DE(;
        sample_prior,
        bounds, 
        sample = resample,
        burnin = 5000, 
        n_initial = (n_μ + 1) * 4,
        Np = 3,
        n_groups = 1,
        θsnooker = 0.1,
    )
    # sample from the posterior distribution 
    n_iter = 50_000
    chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
    sds = describe(chains)[1][1:n_μ,:std]
    means = describe(chains)[1][1:n_μ,:mean]
    @test all(x -> isapprox(x, 0.1; atol = .01), sds)
    @test all(x -> isapprox(x, 0.0; atol = .2), means)
    @test std(means) ≈ 0.1 atol = .01
end