# DifferentialEvolutionMCMC.jl

```@setup de_animation 
using DifferentialEvolutionMCMC
using DifferentialEvolutionMCMC: sample_init
using DifferentialEvolutionMCMC: crossover!
using Distributions
using StatsPlots
#using PyPlot
using Random

#pyplot()

Random.seed!(81872)

data = rand(Normal(0.0, 1.0), 5)

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
bounds = ((-Inf,Inf),(0.0,Inf))

function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end

model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names)

de = DE(;sample_prior, bounds, burnin = 10, n_groups = 4, Np = 8)
n_iter = 60

colors = [RGB(.251, .388, .847),
            RGB(.220, .596, .149),
            RGB(.584, .345, .698),
            RGB(.796, .235, .200)]

groups = sample_init(model, de, n_iter)
animation = @animate for i in 1:n_iter
    de_plot = scatter()
    for (j,g) in enumerate(groups)
        crossover!(model, de, g)
        Θ = mapreduce(p -> p.Θ, hcat, g)'
        scatter!(de_plot, Θ[:,1], Θ[:,2], xlabel="μ", ylabel = "σ", grid=false, leg=false,
            xlims=(-2,2), ylims=(0,2), framestyle=:box, title = "iteration $i",
            markersize=5, xaxis=font(12), yaxis=font(12), markerstrokewidth=3, color = colors[j])
        vline!([mean(data)], color = :black, linestyle=:dash)
        hline!([std(data)], color = :black, linestyle=:dash)
     end
end
gif(animation, "de_animation.gif", fps = 4)
```
Welcome to DifferentialEvolutionMCMC.jl. With this package, you can perform Bayesian parameter estimation using Differential Evolution MCMC (DEMCMC), and perform optimization using the basic DE algorithm  Please see the navigation panel on the left for information pertaining to the API and runnable examples. 

## How Does it Work?
### Intuition 

The basic idea behind DEMCMC is that a group of interacting particles traverse the parameter space and share information about the joint posterior distribution of model parameters. Across many iterations, the samples obtained from the particles will approximate the posterior distribution. The image below illustrates how the particles sample from the posterior distribution of $\mu$ and $\sigma$ of a simple Gaussian model. In this example, five observations were sampled from a Gaussian distribution with $\mu=0$ and $\sigma=1.0$. The DEMCMC sampler consists of four color coded groups of particles which operate semi-independently from each other. Note that the dashed lines represent the maximum likelihood estimates. The particles cluster near the maximum likelihood estimates because the true parameters are close the center of the prior distributions. 

![](de_animation.gif)

### Technical Description 

This section provides a more technical explanation of the basic algorithm. Please see the references below for more details. More formally, a particle $p \in [1,2,\dots, P]$ is a vector of $n$ parameters in a $\mathbb{R}^n$ parameter space defined as:

$\Theta_p = [\theta_{p,1},\theta_{p,2},\dots \theta_{p,n}].$ 

On each iteration $i$, a new position for each particle $p$ is proposed by adding the weighted difference of two randomly selected particles $j,k$ to particle $p$ along with a small amount of noise. Formally, the proposal is given by:

$\Theta_p^\prime = \Theta_p + \gamma (\Theta_j - \Theta_k) + b,$

where $b \sim \mathrm{uniform}(-\epsilon, \epsilon)$. DEMCMC uses the difference between randomly selected particles to leverage approximate derivatives in the proposal process. The proposal is accepted according to the Metropolis-Hastings rule whereby the proposal is always accepted if its log likelihood is greater than that of the current position, but is accepted proportionally to the ratio of log likelihoods otherwise.
# References

Ter Braak, C. J. (2006). A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces. Statistics and Computing, 16, 239-249.

Ter Braak, C. J., & Vrugt, J. A. (2008). Differential evolution Markov chain with snooker updater and fewer chains. Statistics and Computing, 18, 435-446.