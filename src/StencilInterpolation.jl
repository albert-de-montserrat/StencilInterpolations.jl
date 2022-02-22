module StencilInterpolation

using CUDA

include("utils/Utils.jl")
include("Scatter.jl")
include("Gather.jl")

export scattering, scattering!
export gathering!

end # module
