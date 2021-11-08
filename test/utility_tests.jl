@testset verbose = true "utility tests" begin 
    @safetestset "Discard Burnin" begin
        using DifferentialEvolutionMCMC, Test, Random, Parameters, Distributions
        import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
        Random.seed!(29542)
        N = 10
        k = rand(Binomial(N, .5))
        data = (N = N,k = k)

        prior_loglike(θ) = logpdf(Beta(1, 1), θ)

        sample_prior() = rand(Beta(1, 1))

        bounds = ((0,1),)

        function loglike(data, θ)
            return logpdf(Binomial(data.N, θ), data.k)
        end

        names = (:θ,)

        model = DEModel(; 
            sample_prior, 
            prior_loglike, 
            loglike, 
            data,
            names
        )

        burnin = 1500
        n_iter = 3000

        de = DE(; Np=4, bounds, burnin, discard_burnin=false)

        chains = sample(model, de, n_iter)
        @test length(chains) == n_iter

        de = DE(; Np=4, bounds, burnin)
        chains = sample(model, de, n_iter)
        @test length(chains) == burnin
    end


    @safetestset "reset!" begin
        using DifferentialEvolutionMCMC, Test
        import DifferentialEvolutionMCMC: reset!

        p1 = Particle(Θ = [[.7,.5,.1],.4,.6])
        p2 = Particle(Θ = [[.9,.8,.5],.7,.8])
        idx = [[true,false,false],false,true]
        reset!(p1, p2, idx)

        @test p1.Θ[1][1] ≠ p2.Θ[1][1]
        @test p1.Θ[1][2] == p2.Θ[1][2]
        @test p1.Θ[1][3] == p2.Θ[1][3]
        @test p1.Θ[2] == p2.Θ[2]
        @test p1.Θ[3] ≠ p2.Θ[3]

        p1 = Particle(Θ = [[.7 .5;.1 .3],.4,.6])
        p2 = Particle(Θ = [[.9 .8;.5 .2],.7,.8])
        idx = [[true false; false true],false,true]
        reset!(p1, p2, idx)

        @test p1.Θ[1][1,1] ≠ p2.Θ[1][1,1]
        @test p1.Θ[1][1,2] == p2.Θ[1][1,2]
        @test p1.Θ[1][2,1] == p2.Θ[1][2,1]
        @test p1.Θ[1][2,2] ≠ p2.Θ[1][2,2]
        @test p1.Θ[1][3] == p2.Θ[1][3]
        @test p1.Θ[2] == p2.Θ[2]
        @test p1.Θ[3] ≠ p2.Θ[3]
    end

    @testset "projection" begin 
        using Test, DifferentialEvolutionMCMC
        import DifferentialEvolutionMCMC: project

        # for example, see: https://www.youtube.com/watch?v=xSu-0xcRBo8&ab_channel=FireflyLectures
        proj(x1, x2) = (x1' * x2) / (x2' * x2) * x2
        x1 = [-1.0,4.0]
        x2 = [2.0,7.0]
        p1 = Particle(Θ = x1)
        p2 = Particle(Θ = x2)
        p3 = project(p1, p2)
        correct = proj(x1, x2)

        @test correct ≈ [52/53,182/53]
        @test p3.Θ ≈ correct

        x1 = [[-1.0,],4.0]
        x2 = [[2.0,],7.0]
        p1 = Particle(Θ = x1)
        p2 = Particle(Θ = x2)
        p3 = project(p1, p2)
        @test vcat(p3.Θ...) ≈ correct
    end

    @safetestset "Migration" begin
        using DifferentialEvolutionMCMC, Test, Random, Parameters, Distributions
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

        N = 10
        k = rand(Binomial(N, .5))
        data = (N = N,k = k)
        
        prior_loglike(θ) = logpdf(Beta(1, 1), θ)

        sample_prior() = rand(Beta(1, 1))

        bounds = ((0,1),)

        function loglike(data, θ)
            return logpdf(Binomial(data.N, θ), data.k)
        end

        names = (:θ,)

        model = DEModel(; 
            sample_prior, 
            prior_loglike, 
            loglike, 
            data,
            names
        )

        burnin = 1500
        n_iter = 3000

        de = DE(; Np=4, bounds, burnin, discard_burnin=false)

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
end
