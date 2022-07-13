module StencilInterpolations

using CUDA
using VectorizationBase: Vec, vsum, vadd

include("utils.jl")
include("gather.jl")
include("scatter.jl")
include("kernels.jl")
# include("bilinear.jl")
# include("trilinear.jl")

export scattering, scattering!, gathering!
export grid2particle, grid2particle!, grid2particle_xcell!, gathering_xcell!
export lerp, ndlinear, random_particles
export parent_cell

end # module
