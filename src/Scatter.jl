## CPU

function _scattering(p::NTuple{N, A}, dxi::NTuple{N, B}, xi::NTuple{N, C}, F::Array{D, N}) where {A,B,C,D,N}
    # unpack tuples
    dx, dy = dxi
    px, py = p
    x, y = xi

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y = parent_cell(p, dxi)

    dx_particle = px - x[idx_x]
    dy_particle = py - y[idx_y]
   
    # Interpolate field F onto particle
    Fp =
        F[idx_x,   idx_y  ]*(1-dx_particle/dx)*(1-dy_particle/dy) + 
        F[idx_x,   idx_y+1]*(dx_particle/dx)*(1-dy_particle/dy) + 
        F[idx_x+1, idx_y  ]*(1-dx_particle/dx)*(dy_particle/dy) + 
        F[idx_x+1, idx_y+1]*(dx_particle/dx)*(dy_particle/dy)

    return Fp
end

function scattering(xi, dxi, F::Array{T, N}, particle_coords) where {T, N}
    
    np = length(particle_coords[1])
    Fp = zeros(T, np)

    Threads.@threads for i in 1:np
        @inbounds Fp[i] = _scattering((particle_coords[1][i], particle_coords[2][i]), dxi, xi, F)
    end

    return Fp
end

## CUDA

function _scattering!(Fp::CuDeviceVector{T, 1}, p::NTuple{2, A}, dxi::NTuple{2, B}, xi::NTuple{2, C}, F::CuDeviceMatrix{T, 1}) where {A, B, C, T}

    ix  = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    dx, dy = dxi
    px, py = p
    x, y = xi

    if ix < length(px)
        # indices of lowermost-left corner of the cell 
        # containing the particle
        idx_x, idx_y = parent_cell((px[ix], py[ix]), dxi)

        dx_particle = px[ix] - x[idx_x]
        dy_particle = py[ix] - y[idx_y]
    
        # Interpolate field F onto particle
        Fp[ix] =
            F[idx_y,   idx_x  ]*(1-dx_particle/dx)*(1-dy_particle/dy) + 
            F[idx_y,   idx_x+1]*(dx_particle/dx)*(1-dy_particle/dy) + 
            F[idx_y+1, idx_x  ]*(1-dx_particle/dx)*(dy_particle/dy) + 
            F[idx_y+1, idx_x+1]*(dx_particle/dx)*(dy_particle/dy)
    end

    return 
end

function scattering!(Fpd, xi, dxi, Fd::CuArray{T, 2}, particle_coords::NTuple{2, CuArray}) where {T}
    N = length(particle_coords[1])
    numblocks = ceil(Int, N/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=20 _scattering!(Fpd, particle_coords, dxi, xi, Fd)
    end
end