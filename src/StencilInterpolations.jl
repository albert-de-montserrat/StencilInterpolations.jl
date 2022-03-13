module StencilInterpolations

using CUDA
using VectorizationBase: Vec, vsum, vadd

include("utils.jl")
include("gather.jl")
include("kernels.jl")
include("bilinear.jl")
include("trilinear.jl")

export scattering, scattering!, gathering!
export lerp, bilinear, trilinear

end # module
