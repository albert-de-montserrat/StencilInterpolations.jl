module StencilInterpolation

using CUDA
using MuladdMacro
using VectorizationBase
using VectorizationBase: Vec, vsum, vadd

include("Utils.jl")
include("Scatter.jl")
include("Gather.jl")

export scattering, scattering!
export gathering!

end # module
