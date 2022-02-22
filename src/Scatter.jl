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
end

function scattering(xi, dxi, F::Array{T, N}, particle_coords) where {T, N}
    
    np = length(particle_coords[1])
    Fp = zeros(T, np)

    Threads.@threads for i in 1:np
        @inbounds Fp[i] = _scattering((particle_coords[1][i], particle_coords[2][i]), dxi, xi, F)
    end

    return Fp
end

function scattering!(Fp::CuDeviceVector{T, 1}, p::NTuple{N, A}, dxi::NTuple{N, B}, xi::NTuple{N, C}, F::CuDeviceMatrix{T, 1}) where {A, B, C, N, T}

    ix  = threadIdx().x

    # unpack tuples
    dx, dy = dxi
    px, py = p
    x, y = xi

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
    return 
end


# px, py = CuArray.(particle_coords)

# numblocks = ceil(Int, N/256)
# @btime CUDA.@sync begin
#     @cuda threads=256 blocks=numblocks foo!(Fp, (px, py) , dxi, xi, Fd)
# end