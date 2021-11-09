# project x1 on to x2
project(x1, x2) = (x1' * x2) / (x2' * x2) * x2

# compute intercept and slope from two points
function get_coefs(x1, x2)
     β1 = (x2[2] - x1[2]) / (x2[1] - x1[1])
     return x2[2] - β1 * x2[1], β1
end
 
# make a line 
make_line(β0, β1, x) = β0 .+ β1 .* x 