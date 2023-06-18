```@setup optimization_setup
using DifferentialEvolutionMCMC
using DifferentialEvolutionMCMC: minimize!
using Random 
Random.seed!(6845)

function sample_prior()
    return [rand(Uniform(-5, 5), 2)]
 end

 function rastrigin(_, x)
     A = 10.0
     n = length(x)
     y = A * n
     for  i in 1:n
         y +=  + x[i]^2 - A * cos(2 * π * x[i])
     end
     return y 
 end

bounds = ((-5.0,5.0),)
names = (:x,)

model = DEModel(; 
    sample_prior, 
    loglike = rastrigin, 
    data = nothing,
    names)

de = DE(;
    sample_prior,
    bounds,
    Np = 10, 
    n_groups = 1, 
    update_particle! = minimize!,
    evaluate_fitness! = evaluate_fun!)
```
# Optimization Example

The purpose of this example is to demonstrate how to optimize a function as opposed to perform Bayesian parameter estimation.
## Load Packages
Our first step is to load the required packages.

```@example optimization_setup
using DifferentialEvolutionMCMC
using DifferentialEvolutionMCMC: minimize!
using Distributions
using Random 
Random.seed!(6845)
```

## Initialize the DE Algorithm
Each particle in DifferentialEvolution must be seeded with an initial value. To do so, we define a function that returns the initial value. Even though we are not performing Bayesian parameter estimation, we want to use prior information to intelligently seed the DE algorithm. In our case, we will return a vector contained in a vector. 
```@example optimization_setup
function sample_prior()
    return [rand(Uniform(-5, 5), 2)]
 end
```

## Objective Function
In this example, we will find the minimum of the [rastrigin](https://en.wikipedia.org/wiki/Rastrigin_function), which challenging due to its "egg carton-like" surface containing several local minima. 

![](https://upload.wikimedia.org/wikipedia/commons/8/8b/Rastrigin_function.png)

The minimum of the rastrigin function is the zero vector for input `x`. The code block below defines the rastrigin function for an input vector of an arbitrary length. Since we will not be passing data to the function, we will set the first argument to `_` and assign it a value of `nothing` below.

```@example optimization_setup
function rastrigin(_, x)
    A = 10.0
    n = length(x)
    y = A * n
    for  i in 1:n
        y +=  + x[i]^2 - A * cos(2 * π * x[i])
    end
    return y 
end
```
## Define Bounds and Parameter Names
We must define the lower and upper bounds of each parameter. The bounds are a `Tuple` of tuples, where the $i^{\mathrm{th}}$ inner tuple contains the lower and upper bounds of the $i^{\mathrm{th}}$ parameter. In the `rastrigin` function, `x` is a vector with an unspecified length. The bounds below applies to each element in `x`.
```example optimization_setup
bounds = ((-5.0,5.0),)
```
We can also pass a `Tuple` of names for the argument `x`.

```@example optimization_setup
names = (:x,)
```
## Define Model Object
Now that we have defined the necessary components of our model, we will organize them into a `DEModel` object as shown below. Notice that `data = nothing` because we do not need data for this optimization problem.

```@example optimization_setup
model = DEModel(; 
    sample_prior, 
    loglike = rastrigin, 
    data = nothing,
    names)
```
## Define Sampler
Next, we will create a sampler with the constructor `DE`. Here, we will pass the `sample_prior` function and the variable `bounds`, which constains the lower and upper bounds of each parameter. In addition, we will specify $1,000$ burnin samples, `Np=10` particles for `n_groups=1` groups. In the last two keyword arguments, we use `minimize!` for the `update_particle!` function because we want to minimize `rastrigin`, and `evaluate_fitness! = evaluate_fun!` because we do not want to incorporate the prior log likelihood into our evaluation of `rastrigin`.
```@example optimization_setup
de = DE(;
    sample_prior,
    bounds,
    Np = 10, 
    n_groups = 1, 
    update_particle! = minimize!,
    evaluate_fitness! = evaluate_fun!)
```

## Optimize the Function
The code block below runs the optimizer for $2000$ iterations with the progress bar. 
```@example optimization_setup
n_iter = 2000
particles = optimize(model, de, n_iter, progress=true);
results = get_optimal(de, model, particles)
```