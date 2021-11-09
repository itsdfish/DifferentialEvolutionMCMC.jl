cd(@__DIR__)
using Pkg 
# you may need to use Pkg.instantiate() the first time you run the file
Pkg.activate("")
using Plots
include("functions.jl")

# current chain 
x = [.8,.5]
# random chain
z = [1.5,2.5]
# difference of current and random z 
# forms dotted line 
d = x - z

# random chains r1 and r2
r1 = [0,-1.5] 
r2 = [-.3,-2.1]

# project r1 onto line spanned by d = x - z
proj1 = project(r1, d)
# project r2 onto line spanned by d = x - z
proj2 = project(r2, d)

# difference scalar set to 1 for simplicity
γ = 1.0

# difference in projections 
dproj1_2 = proj1 - proj2

# proposal x + γ * (proj1 - proj2)
proposal = x + γ * dproj1_2

scatter(
    [x[1]],
    [x[2]],
    xlims = (-3.0, 3.0), 
    ylims = (-3.0, 3.0), 
    color = :black, 
    linewidth = 2, 
    label = "",
    framestyle = :origin,
)
annotate!(x[1]-.1,x[2]+.2, text("X", :black, 12))

scatter!(
    [z[1]],
    [z[2]],
    xlims = (-3.0, 3.0), 
    ylims = (-3.0, 3.0), 
    color = :black, 
    linewidth = 2, 
    label = "",
    framestyle = :origin,
)
annotate!(z[1]-.1,z[2] +.1, text("Z", :black, 12))

xvals = -30:.1:30
β = get_coefs(x, z)
yvals = make_line(β..., xvals)
plot!(
    xvals,
    yvals,
    color = :black,
    linestyle = :dash,
    linewidth = 2, 
    label = "",
    framestyle = :origin,
)

scatter!(
    [r1[1]],
    [r1[2]],
    color = :black,
    linewidth = 2, 
    label = "",
    framestyle = :origin,
)
annotate!(r1[1]-.1,r1[2] +.2, text("R1", :black, 12))


scatter!(
    [r2[1]],
    [r2[2]],
    color = :black,
    linewidth = 2, 
    label = "",
    framestyle = :origin,
)
annotate!(r2[1]-.1,r2[2] +.2, text("R2", :black, 12))

scatter!(
    [proposal[1]],
    [proposal[2]],
    color = :red,
    linewidth = 2, 
    label = "",
    framestyle = :origin,
)
annotate!(proposal[1] - .1,proposal[2] +.1, text("X*", :black, 12))
