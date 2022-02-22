
distance_weigth(a::NTuple{N, T}, b::NTuple{N, T}; order = 4) where {T, N} = 1/distance(a, b)^order

function foo!(upper, lower, Fpd, xi, dxi, p; order = 4)
    
    # unpack tuples
    px, py = p
    x, y = xi
    
    idx  = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx ≤ length(px)
        p_idx = (px[idx], py[idx])

        # indices of lowermost-left corner of
        # the cell containing the particle
        idx_x, idx_y = parent_cell(p_idx, dxi)

        ω1::Float64 = distance_weigth( (x[idx_x],   y[idx_y]),   p_idx, order=order)
        ω2::Float64 = distance_weigth( (x[idx_x+1], y[idx_y]),   p_idx, order=order)
        ω3::Float64 = distance_weigth( (x[idx_x],   y[idx_y+1]), p_idx, order=order)
        ω4::Float64 = distance_weigth( (x[idx_x+1], y[idx_y+1]), p_idx, order=order)
        
        Fpi::Float64 = Fpd[idx]

        CUDA.@atomic upper[idx_x,     idx_y] += ω1*Fpi
        CUDA.@atomic upper[idx_x+1,   idx_y] += ω2*Fpi
        CUDA.@atomic upper[idx_x,   idx_y+1] += ω3*Fpi
        CUDA.@atomic upper[idx_x+1, idx_y+1] += ω4*Fpi
        CUDA.@atomic lower[idx_x,     idx_y] += ω1
        CUDA.@atomic lower[idx_x+1,   idx_y] += ω2
        CUDA.@atomic lower[idx_x,   idx_y+1] += ω3
        CUDA.@atomic lower[idx_x+1, idx_y+1] += ω4
    end

    return
end

@time CUDA.@sync begin
    @cuda threads=256 blocks=numblocks foo!(upper, lower, Fpd, xi, dxi, p)
end

function potato!(Fd, upper, lower)
    
    idx  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy  = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (idx < size(Fd, 1)) && (idy < size(Fd, 2))
        @inbounds Fd[idx, idy] = upper[idx, idy] / lower[idx, idy]
    end

    return
end

nt=2^7
numblocks = ceil(Int, N/nt)
@btime CUDA.@sync begin
    @cuda threads=$101 blocks=$1 potato!($Fd, $upper, $lower)
end

@parallel function potato!(Fd, upper, lower)
    @all(Fd) = @all(upper) / @all(lower)
    return
end

@parallel (nblocks=numblocks, nthreads=256) potato!(Fd, upper, lower)

# upper = zeros(size(F)) # we can recycle F here
p = CuArray.(particle_coords)
upper = CUDA.zeros(size(F)) 
lower = CUDA.zeros(size(F)) 

N = length(particle_coords[1])
numblocks = ceil(Int, N/256)
CUDA.@sync begin
    @cuda threads=256 blocks=numblocks foo!(upper, lower, Fpd, xi, dxi, CuArray.(particle_coords))
end



# synchronize()

# Threads.@threads for i in eachindex(F)
#     @inbounds F[i] = 
#         sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
#         sum(lower[nt][i] for nt in 1:Threads.nthreads())
# end
