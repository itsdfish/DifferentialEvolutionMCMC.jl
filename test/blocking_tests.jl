@safetestset "blocking" begin
    using DifferentialEvolutionMCMC, Test, Random, Distributions
    using MCMCChains
    Random.seed!(58122)

    data = rand(Normal(0.0, 1.0), 100)

    function prior_loglike(μ, σ)
        LL = 0.0
        LL += logpdf(Normal(0, 10), μ)
        LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
        return LL
    end

    function sample_prior()
        μ = rand(Normal(0, 10))
        σ = rand(truncated(Cauchy(0, 1), 0, Inf))
        return [μ, σ]
    end

    bounds = ((-Inf, Inf), (0.0, Inf))

    data = rand(Normal(0.0, 1.0), 1000)

    function loglike(data, μ, σ)
        return sum(logpdf.(Normal(μ, σ), data))
    end

    bounds = ((-Inf, Inf), (0.0, Inf))

    names = (:μ, :σ)

    blocks = [[true, false], [false, true]]

    blocks = as_union(blocks)

    blocking_on = x -> true

    model = DEModel(;
        sample_prior,
        prior_loglike,
        loglike,
        data,
        names
    )

    de = DE(;
        sample_prior,
        bounds,
        burnin = 1000,
        Np = 6,
        blocking_on,
        blocks
    )

    n_iter = 2000
    chains = sample(model, de, n_iter, progress = true)

    means = describe(chains)[1][:, :mean]
    rhat = describe(chains)[1][:, :rhat]

    @test means[1] ≈ 0.0 atol = 0.1
    @test means[2] ≈ 1.0 atol = 0.1
    @test rhat[1] ≈ 1.0 atol = 0.01
    @test rhat[2] ≈ 1.0 atol = 0.01

    chains = sample(model, de, MCMCThreads(), n_iter, progress = true)

    means = describe(chains)[1][:, :mean]
    rhat = describe(chains)[1][:, :rhat]

    @test means[1] ≈ 0.0 atol = 0.1
    @test means[2] ≈ 1.0 atol = 0.1
    @test rhat[1] ≈ 1.0 atol = 0.01
    @test rhat[2] ≈ 1.0 atol = 0.01
end
