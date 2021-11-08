@safetestset "Binomial Model" begin
    using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
    Random.seed!(29542)
    N = 10
    k = rand(Binomial(N, .5))
    data = (N = N,k = k)
    
    prior_loglike(θ) = logpdf(Beta(1, 1), θ)

    sample_prior() = rand(Beta(1, 1))

    bounds = ((0,1),)
    names = (:θ,)

    function loglike(data, θ)
        return logpdf(Binomial(data.N, θ), data.k)
    end

    model = DEModel(; 
        sample_prior, 
        prior_loglike, 
        loglike, 
        data,
        names
    )

    de = DE(;sample_prior, bounds, burnin=1500, Np=3)

    n_iter = 3000
    chains = sample(model, de, n_iter)
    μθ = describe(chains)[1][:,:mean][1]
    σθ = describe(chains)[1][:,:std][1]
    rhat = describe(chains)[1][:,:rhat][1]
    solution = Beta(k + 1, N - k + 1)
    @test μθ ≈ mean(solution) rtol = .02
    @test σθ ≈ std(solution) rtol = .02
    @test rhat ≈ 1.0 atol = .01
end