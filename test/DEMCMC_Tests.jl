cd(@__DIR__)

@testset "Binomial Model" begin
    using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
    import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
    Random.seed!(29542)
    N = 10
    k = rand(Binomial(N, .5))
    data = (N = N,k = k)
    priors = (
        θ = (Beta(1, 1),),
    )

    bounds = ((0,1),)

    function loglike(θ, data)
        return logpdf(Binomial(data.N, θ), data.k)
    end

    loglike(θ) = loglike(θ..., data)

    model = DEModel(priors=priors, model=loglike)

    de = DE(;priors=priors, bounds=bounds, burnin=1500)
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

@testset "Gaussian" begin
    using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
    import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
    Random.seed!(973536)
    priors = (
        μ = (Normal(0, 10),),
        σ = (truncated(Cauchy(0, 1), 0.0, Inf),)
    )

    bounds = ((-Inf,Inf),(0.0,Inf))

    data = rand(Normal(0.0, 1.0), 100)

    function loglike(μ, σ, data)
        return sum(logpdf.(Normal(μ, σ), data))
    end

    loglike(θ) = loglike(θ..., data)
    model = DEModel(priors=priors, model=loglike)
    de = DE(;priors=priors, bounds=bounds, burnin=1500)
    n_iter = 3000
    chains = sample(model, de, n_iter)
    μ_de = describe(chains)[1][:,:mean]
    σ_de = describe(chains)[1][:,:std]
    rhat = describe(chains)[1][:,:rhat]

    @model model(data) = begin
        μ ~ Normal(0, 10)
        σ ~ truncated(Cauchy(0, 1), 0.0, Inf)
        for i in 1:length(data)
            data[i] ~ Normal(μ, σ)
        end
    end
    chn = sample(model(data), NUTS(1000, .85), 2000)
    μ_nuts = describe(chn)[1][:,:mean]
    σ_nuts = describe(chn)[1][:,:std]

    @test all(isapprox.(rhat, fill(1.0, 2), atol=.05))
    @test all(isapprox.(μ_nuts, μ_de, atol=.01))
    @test all(isapprox.(σ_nuts, σ_de, atol=.01))
end

@testset "LNR" begin
    using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
    import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
    Random.seed!(59391)
    include("LogNormalRace.jl")

    dist = LNR(μ=[-2.,-2.,-3.,-3], σ=1.0, ϕ=.5)
    data = rand(dist, 100)

    function loglike(μ, σ, ϕ, data)
        dist = LNR(μ=μ, σ=σ, ϕ=ϕ)
        return sum(logpdf.(dist, data))
    end

    loglike(θ) = loglike(θ..., data)

    minRT = minimum(x -> x[2], data)
    priors = (μ = (Normal(0, 3), 4),σ = (truncated(Cauchy(0, 1), 0.0, Inf),),
        ϕ = (Uniform(0., minRT),))
    bounds = ((-Inf,0.),(1e-10,Inf),(0.,minRT))
    model = DEModel(priors=priors, model=loglike)
    de = DE(;priors=priors, bounds=bounds, burnin=2000)
    n_iter = 4000
    chains = sample(model, de, n_iter)
    μ_de = describe(chains)[1][:,:mean]
    σ_de = describe(chains)[1][:,:std]
    rhat = describe(chains)[1][:,:r_hat]

    @model model(data) = begin
        minRT = minimum(x -> x[2], data)
        μ ~ MvNormal(zeros(4), 3)
        σ ~ truncated(Cauchy(0, 1), 0.0, Inf)
        ϕ ~ Uniform(0.0, minRT)
        data ~ LNR(μ=μ, σ=σ, ϕ=ϕ)
    end

    chn = sample(model(data), NUTS(1000, .85), 2000)
    μ_nuts = describe(chn)[1][:,:mean]
    σ_nuts = describe(chn)[1][:,:std]

    @test all(isapprox.(rhat, fill(1.0, 6), atol=.05))
    @test all(isapprox.(μ_nuts, μ_de, rtol=.05))
    @test all(isapprox.(σ_nuts, σ_de, rtol=.05))
end

function equal(p1::Particle, p2::Particle)
    fields = fieldnames(Particle)
    for field in fields
        if getfield(p1, field) != getfield(p2, field)
            println(field)
            return false
        end
    end
    return true
end

@testset "Migration" begin
    using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
    import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init

    Random.seed!(0451) # Random.seed!(459)
    priors = (
        θ = (Beta(1, 1),),
    )
    bounds = ((0,1),)

    data = (N = 10,k = 5)

    function loglike(θ, data)
        return logpdf(Binomial(data.N, θ), data.k)
    end

    loglike(θ) = loglike(θ..., data)

    model = DEModel(priors=priors, model=loglike)
    de = DE(;priors=priors, bounds=bounds, burnin=1500)
    n_iter = 3000
    groups = sample_init(model, de, n_iter)
    sub_group = select_groups(de, groups)
    c_groups = deepcopy(groups)
    c_sub_group = deepcopy(sub_group)
    p_idx,particles = select_particles(sub_group)
    c_particles = deepcopy(particles)
    shift_particles!(sub_group, p_idx, particles)
    gidx = 1:length(sub_group)
    cidx = circshift(gidx, 1)
    cp_idx = circshift(p_idx, 1)
    for (i,c,p,cp) in zip(gidx, p_idx, cidx, cp_idx)
        @test sub_group[i][c].Θ == c_sub_group[p][cp].Θ
    end
    ridx = [4,3] # ridx = [4,1]
    for (i,c,r) in zip(1:2, p_idx[1:2], ridx)
        @test sub_group[i][c].Θ == groups[r][c].Θ
    end
end
