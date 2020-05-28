using KernelDensity, Distributions, Interpolations
import KernelDensity: kernel_dist
kernel_dist(::Type{Epanechnikov}, w::Float64) = Epanechnikov(0.0, w)
kernel(data) = kde(data; kernel=Epanechnikov)
