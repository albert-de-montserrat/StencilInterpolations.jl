## CPU BI-LINEAR

function _scattering(p::NTuple{2, A}, dxi::NTuple{2, B}, xi::NTuple{2, C}, F::Array{D, 2}) where {A,B,C,D}
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

function scattering(xi, dxi, F::Array{T, 2}, particle_coords) where {T}
    
    np = length(particle_coords[1])
    Fp = zeros(T, np)

    Threads.@threads for i in 1:np
        @inbounds Fp[i] = _scattering((particle_coords[1][i], particle_coords[2][i]), dxi, xi, F)
    end

    return Fp
end


## CPU TRI-LINEAR

function _scattering(p::NTuple{3, A}, dxi::NTuple{3, B}, xi::NTuple{3, C}, F::Array{D, 3}) where {A, B, C, D}
    # unpack tuples
    dx, dy, dz = @. 1/dxi
    px, py, pz = p
    x, y, z = xi

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    # distance from particle to lowermost-left corner of the cell 
    dx_particle = px - x[idx_x]
    dy_particle = py - y[idx_y]
    dz_particle = pz - z[idx_z]
   
    # Interpolate field F onto particle
    Fp =
        F[idx_x,   idx_y  , idx_z  ]*(1-dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz) + 
        F[idx_x,   idx_y+1, idx_z  ]*(dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz) + 
        F[idx_x+1, idx_y  , idx_z  ]*(1-dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz) + 
        F[idx_x+1, idx_y+1, idx_z  ]*(dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz) +
        F[idx_x,   idx_y  , idx_z+1]*(1-dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz) +  
        F[idx_x,   idx_y+1, idx_z+1]*(dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz) +  
        F[idx_x+1, idx_y  , idx_z+1]*(1-dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz) +  
        F[idx_x+1, idx_y+1, idx_z+1]*(dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz)

    return Fp
end

# manually vectorized version of trilinear kernel
function _vscattering(p::NTuple{3, A}, dxi::NTuple{3, B}, xi::NTuple{3, C}, F::Array{Float64, 3}) where {A, B, C}
    # unpack tuples
    dx, dy, dz = @. 1/dxi
    px, py, pz = p
    x, y, z = xi

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    # distance from particle to lowermost-left corner of the cell 
    dx_particle = px - x[idx_x]
    dy_particle = py - y[idx_y]
    dz_particle = pz - z[idx_z]
   
    a1 = VectorizationBase.Vec(
        (
            F[idx_x+1, idx_y  , idx_z], # 3
            F[idx_x+1, idx_y+1, idx_z], # 4
            F[idx_x,   idx_y+1, idx_z], # 2
            F[idx_x,   idx_y  , idx_z], # 1
        )...
    )
    a2 = VectorizationBase.Vec(
        (
            F[idx_x+1, idx_y  , idx_z+1], # 3
            F[idx_x+1, idx_y+1, idx_z+1], # 4
            F[idx_x,   idx_y+1, idx_z+1], # 2
            F[idx_x,   idx_y  , idx_z+1], # 1
        )...
    )

    b1 = VectorizationBase.Vec(
        (
            (1-dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz),     # 3
            (dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz),       # 4
            (dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz),     # 2
            (1-dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz),   # 1
        )...
    )
    b2 = VectorizationBase.Vec(
        (
            (1-dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz),   # 3
            (dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz),     # 4
            (dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz),   # 2
            (1-dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz), # 1
        )...
    )

    # Interpolate field F onto particle
    Fp =  VectorizationBase.vsum(VectorizationBase.vmul(a1, b1)) + VectorizationBase.vsum(VectorizationBase.vmul(a2, b2))

    return Fp
end

function _vscattering(p::NTuple{3, A}, dxi::NTuple{3, B}, xi::NTuple{3, C}, F::Array{Float32, 3}) where {A, B, C}
    # unpack tuples
    dx, dy, dz = @. 1/dxi
    px, py, pz = p
    x, y, z = xi

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    # distance from particle to lowermost-left corner of the cell 
    dx_particle = px - x[idx_x]
    dy_particle = py - y[idx_y]
    dz_particle = pz - z[idx_z]
   
    a = VectorizationBase.Vec(
        (
            F[idx_x+1, idx_y  , idx_z], # 3
            F[idx_x+1, idx_y+1, idx_z], # 4
            F[idx_x,   idx_y+1, idx_z], # 2
            F[idx_x,   idx_y  , idx_z], # 1
            F[idx_x+1, idx_y  , idx_z+1], # 3
            F[idx_x+1, idx_y+1, idx_z+1], # 4
            F[idx_x,   idx_y+1, idx_z+1], # 2
            F[idx_x,   idx_y  , idx_z+1], # 1
        )...
    )

    b = VectorizationBase.Vec(
        (
            (1-dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz),     # 3
            (dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz),       # 4
            (dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz),     # 2
            (1-dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz),   # 1
            (1-dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz),       # 3
            (dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz),         # 4
            (dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz),       # 2
            (1-dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz),     # 1
        )...
    )

    # Interpolate field F onto particle
    Fp =  VectorizationBase.vsum(VectorizationBase.vmul(a, b))

    return Fp
end

function scattering(xi, dxi, F::Array{T, 3}, particle_coords) where {T}
    
    np = length(particle_coords[1])
    Fp = zeros(T, np)

    Threads.@threads for i in 1:np
        @inbounds Fp[i] = _vscattering((particle_coords[1][i], particle_coords[2][i],  particle_coords[3][i]), dxi, xi, F)
    end

    return Fp
end


## CUDA BI-LINEAR
function _scattering!(Fp::CuDeviceVector{T, 1}, p::NTuple{2, A}, dxi::NTuple{2, B}, xi::NTuple{2, C}, F::CuDeviceMatrix{T, 1}) where {A, B, C, T}

    ix  = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    dx, dy = dxi
    px, py = p
    x, y = xi

    @inbounds if ix ≤ length(px)
        # indices of lowermost-left corner of the cell 
        # containing the particle
        idx_x, idx_y = parent_cell((px[ix], py[ix]), dxi)

        dx_particle = px[ix] - x[idx_x]
        dy_particle = py[ix] - y[idx_y]
    
        # Interpolate field F onto particle
        Fp[ix] = 
            muladd(F[idx_x+1, idx_y+1]*(dx_particle/dx), dy_particle/dy, 
            muladd(F[idx_x+1, idx_y  ]*(1 - dx_particle/dx), dy_particle/dy,
            muladd(F[idx_x,   idx_y+1]*(dx_particle/dx), 1 - dy_particle/dy, 
                   F[idx_x,   idx_y  ]*(1 - dx_particle/dx) * (1 - dy_particle/dy))))
        
    end

    return 
end

function scattering!(Fpd, xi, dxi, Fd::CuArray{T, 2}, particle_coords::NTuple{2, CuArray}; nt = 512) where {T}
    N = length(particle_coords[1])
    numblocks = ceil(Int, N/nt)
    CUDA.@sync begin
        @cuda threads=nt blocks=numblocks _scattering!(Fpd, particle_coords, dxi, xi, Fd)
    end
end

## CUDA TRI-LINEAR

function _scattering!(
    Fp::CuDeviceVector{T, 1}, 
    p::NTuple{3, CuDeviceVector{T, 1}}, 
    dxi::NTuple{3, T}, 
    xi::NTuple{3, A}, 
    F::CuDeviceArray{T, 3}
) where {A, T}

    ix  = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    dx, dy, dz = dxi
    px, py, pz = p
    x, y, z = xi

    @inbounds if ix ≤ length(px)
        # indices of lowermost-left corner of the cell 
        # containing the particle
        idx_x, idx_y, idx_z = parent_cell((px[ix], py[ix], pz[ix]), dxi)

        dx_particle = px[ix] - x[idx_x]
        dy_particle = py[ix] - y[idx_y]
        dz_particle = pz[ix] - z[idx_z]
    
        # Interpolate field F onto particle
        Fp[ix] = @muladd(
            F[idx_x,   idx_y  , idx_z  ]*(1-dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz) + 
            F[idx_x,   idx_y+1, idx_z  ]*(dx_particle*dx)*(1-dy_particle*dy)*(1-dz_particle*dz) + 
            F[idx_x+1, idx_y  , idx_z  ]*(1-dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz) + 
            F[idx_x+1, idx_y+1, idx_z  ]*(dx_particle*dx)*(dy_particle*dy)*(1-dz_particle*dz) +
            F[idx_x,   idx_y  , idx_z+1]*(1-dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz) +  
            F[idx_x,   idx_y+1, idx_z+1]*(dx_particle*dx)*(1-dy_particle*dy)*(dz_particle*dz) +  
            F[idx_x+1, idx_y  , idx_z+1]*(1-dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz) +  
            F[idx_x+1, idx_y+1, idx_z+1]*(dx_particle*dx)*(dy_particle*dy)*(dz_particle*dz)
        )
        
    end

    return 
end

function scattering!(Fpd, xi, dxi, Fd::CuArray{T, 3}, particle_coords::NTuple{3, CuArray}; nt = 512) where {T}
    N = length(particle_coords[1])
    numblocks = ceil(Int, N/nt)
    CUDA.@sync begin
        @cuda threads=nt blocks=numblocks _scattering!(Fpd, particle_coords, dxi, xi, Fd)
    end
end