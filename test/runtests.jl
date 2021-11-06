using SafeTestsets, Test

files = [
    "blocking_tests.jl",
    "optimization_tests.jl",
    "utility_tests.jl"
]

for f in files
    include(f)
end

@safetestset "gaussian tests" begin
    include("gaussian_tests.jl")
end

@safetestset "lognormal race tests" begin
    include("lognormal_race_tests.jl")
end