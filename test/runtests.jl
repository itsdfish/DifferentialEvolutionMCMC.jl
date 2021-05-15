using SafeTestsets

@safetestset "Binomial Model" begin
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

    function loglike(data, θ)
        return logpdf(Binomial(data.N, θ), data.k)
    end

    model = DEModel(;priors, model=loglike, data)

    de = DE(;priors, bounds, burnin=1500)
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

@safetestset "Discard Burnin" begin
    using DifferentialEvolutionMCMC, Test, Random, Parameters, Distributions
    import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
    Random.seed!(29542)
    N = 10
    k = rand(Binomial(N, .5))
    data = (N = N,k = k)
    priors = (
        θ = (Beta(1, 1),),
    )

    bounds = ((0,1),)

    function loglike(data, θ)
        return logpdf(Binomial(data.N, θ), data.k)
    end

    model = DEModel(;priors, model=loglike, data)
    
    burnin = 1500
    n_iter = 3000

    de = DE(;priors, bounds, burnin, discard_burnin=false)
    chains = sample(model, de, n_iter)
    @test length(chains) == n_iter

    de = DE(;priors, bounds, burnin)
    chains = sample(model, de, n_iter)
    @test length(chains) == burnin
end

@safetestset "Gaussian" begin
    include("Guassian_Test.jl")
end

@safetestset "LNR" begin
   include("Log_Normal_Race_Test.jl")
end

@safetestset "Migration" begin
    using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
    import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init

    Random.seed!(459) #Random.seed!(0451) # 

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

    priors = (
        θ = (Beta(1, 1),),
    )
    bounds = ((0,1),)

    data = (N = 10,k = 5)

    function loglike(data, θ)
        return logpdf(Binomial(data.N, θ), data.k)
    end

    model = DEModel(priors=priors, model=loglike, data=data)
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
    ridx = [4,3] 
    for (i,c,r) in zip(1:2, p_idx[1:2], ridx)
        @test sub_group[i][c].Θ == groups[r][c].Θ
    end
end
