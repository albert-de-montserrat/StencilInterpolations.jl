
distance_weigth(a::NTuple{N, T}, b::NTuple{N, T}; order = 4) where {T, N} = 1/distance(a, b)^order

## CPU

@inbounds function _gathering!(upper, lower, Fpi, p, x, y, dxi, order)
    # indices of lowermost-left corner of   
    # the cell containing the particle
    idx_x, idx_y = parent_cell(p, dxi)

    ω = (
        distance_weigth( (x[idx_x],   y[idx_y]),   p, order=order),
        distance_weigth( (x[idx_x+1], y[idx_y]),   p, order=order),
        distance_weigth( (x[idx_x],   y[idx_y+1]), p, order=order),
        distance_weigth( (x[idx_x+1], y[idx_y+1]), p, order=order)
    )

    nt = Threads.threadid()

    upper[nt][idx_x,     idx_y] += ω[1]*Fpi
    upper[nt][idx_x+1,   idx_y] += ω[2]*Fpi
    upper[nt][idx_x,   idx_y+1] += ω[3]*Fpi
    upper[nt][idx_x+1, idx_y+1] += ω[4]*Fpi
    lower[nt][idx_x,     idx_y] += ω[1]
    lower[nt][idx_x+1,   idx_y] += ω[2]
    lower[nt][idx_x,   idx_y+1] += ω[3]
    lower[nt][idx_x+1, idx_y+1] += ω[4]
end


@inbounds function _gathering!(upper, lower, Fpi, p, x, y, z, dxi, order)
    # indices of lowermost-left corner of   
    # the cell containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    ω = (
        distance_weigth( (x[idx_x],     y[idx_y],   z[idx_z]),   p, order=order),
        distance_weigth( (x[idx_x+1],   y[idx_y],   z[idx_z]),   p, order=order),
        distance_weigth( (x[idx_x],   y[idx_y+1],   z[idx_z]),   p, order=order),
        distance_weigth( (x[idx_x+1], y[idx_y+1],   z[idx_z]),   p, order=order),
        distance_weigth( (x[idx_x],     y[idx_y], z[idx_z+1]),   p, order=order),
        distance_weigth( (x[idx_x+1],   y[idx_y], z[idx_z+1]),   p, order=order),
        distance_weigth( (x[idx_x],   y[idx_y+1], z[idx_z+1]),   p, order=order),
        distance_weigth( (x[idx_x+1], y[idx_y+1], z[idx_z+1]),   p, order=order)
    )

    nt = Threads.threadid()

    upper[nt][idx_x,     idx_y,   idx_z] += ω[1]*Fpi
    upper[nt][idx_x+1,   idx_y,   idx_z] += ω[2]*Fpi
    upper[nt][idx_x,   idx_y+1,   idx_z] += ω[3]*Fpi
    upper[nt][idx_x+1, idx_y+1,   idx_z] += ω[4]*Fpi
    upper[nt][idx_x,     idx_y, idx_z+1] += ω[5]*Fpi
    upper[nt][idx_x+1,   idx_y, idx_z+1] += ω[6]*Fpi
    upper[nt][idx_x,   idx_y+1, idx_z+1] += ω[7]*Fpi
    upper[nt][idx_x+1, idx_y+1, idx_z+1] += ω[8]*Fpi
  
    lower[nt][idx_x,     idx_y,   idx_z] += ω[1]
    lower[nt][idx_x+1,   idx_y,   idx_z] += ω[2]
    lower[nt][idx_x,   idx_y+1,   idx_z] += ω[3]
    lower[nt][idx_x+1, idx_y+1,   idx_z] += ω[4]
    lower[nt][idx_x,     idx_y, idx_z+1] += ω[5]
    lower[nt][idx_x+1,   idx_y, idx_z+1] += ω[6]
    lower[nt][idx_x,   idx_y+1, idx_z+1] += ω[7]
    lower[nt][idx_x+1, idx_y+1, idx_z+1] += ω[8]
end

function gathering!(F::Array{T, 2}, Fp::Vector{T}, xi, dxi, particle_coords; order = 4) where {T}
    
    # unpack tuples
    px, py = particle_coords
    x, y = xi

    # number of particles
    np = length(Fp)

    # TODO think about pre-allocating these 2 buffers
    upper = [zeros(size(F)) for _ in 1:Threads.nthreads()]
    lower = [zeros(size(F)) for _ in 1:Threads.nthreads()]

    # compute ∑ωᵢFᵢ and ∑ωᵢ
    Threads.@threads for i in 1:np
        _gathering!(upper, lower, Fp[i], (px[i], py[i]), x, y, dxi, order)
    end
    
    # compute Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] = 
            sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
            sum(lower[nt][i] for nt in 1:Threads.nthreads())
    end

end

function gathering!(F::Array{T, 3}, Fp::Vector{T}, xi, dxi, particle_coords; order = 4) where {T}
    
    # unpack tuples
    px, py, pz = particle_coords
    x, y, z = xi

    # number of particles
    np = length(Fp)

    # TODO think about pre-allocating these 2 buffers
    upper = [zeros(size(F)) for _ in 1:Threads.nthreads()]
    lower = [zeros(size(F)) for _ in 1:Threads.nthreads()]

    # compute ∑ωᵢFᵢ and ∑ωᵢ
    Threads.@threads for i in 1:np
        _gathering!(upper, lower, Fp[i], (px[i], py[i], pz[i]), x, y, z, dxi, order)
    end
    
    # compute Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] = 
            sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
            sum(lower[nt][i] for nt in 1:Threads.nthreads())
    end

end

## CUDA

function _gather1!(upper::CuDeviceMatrix{T, 1}, lower::CuDeviceMatrix{T, 1}, Fpd::CuDeviceVector{T, 1}, xi, dxi, p; order = 4) where T
    idx  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # unpack tuples
    px, py = p
    x, y = xi

    @inbounds if idx ≤ length(px)
        p_idx = (px[idx], py[idx])

        # indices of lowermost-left corner of
        # the cell containing the particle
        idx_x, idx_y = parent_cell(p_idx, dxi)

        ω1::Float64 = distance_weigth((x[idx_x],   y[idx_y]),   p_idx, order=order)
        ω2::Float64 = distance_weigth((x[idx_x+1], y[idx_y]),   p_idx, order=order)
        ω3::Float64 = distance_weigth((x[idx_x],   y[idx_y+1]), p_idx, order=order)
        ω4::Float64 = distance_weigth((x[idx_x+1], y[idx_y+1]), p_idx, order=order)
        
        Fpi::Float64 = Fpd[idx] # use @shared here?

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

function _gather2!(Fd, upper, lower)
    idx  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy  = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (idx < size(Fd, 1)) && (idy < size(Fd, 2))
        @inbounds Fd[idx, idy] = upper[idx, idy] / lower[idx, idy]
    end

    return
end
    
function gathering!(Fd::CuArray{T, 2}, Fpd::CuArray{T, 1}, xi, dxi, particle_coords; nt = 512) where T
    # TODO: pre-allocate the following buffers
    upper = CUDA.zeros(T, size(Fd)) # we can recycle F here as buffer
    # fill!(Fd, zero(T))
    lower = CUDA.zeros(T, size(Fd)) 

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    N = length(Fpd)
    numblocks = ceil(Int, N/nt)
    CUDA.@sync begin
        @cuda threads=nt blocks=numblocks _gather1!(upper, lower, Fpd, xi, dxi, particle_coords)
    end

    # seond and final kernel that computes Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    nx, ny = size(Fd)
    nblocksx = ceil(Int, nx/32)
    nblocksy = ceil(Int, ny/32)
    CUDA.@sync begin
        @cuda threads=(32,32) blocks=(nblocksx,nblocksy) _gather2!(Fd, upper, lower)
    end
end


