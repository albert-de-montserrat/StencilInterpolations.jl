## CPU

function _grid2particle(p::NTuple, xci::Tuple, xi::NTuple, dxi::NTuple, F::AbstractArray)
    # check that the particle is inside the grid
    # isinside(p, xi)

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx = parent_cell(p, dxi, xci)

    # normalize particle coordinates
    ti = normalize_coordinates(p, xi, dxi, idx)

    # F at the cell corners
    Fi = field_corners(F, idx)

    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end

function _grid2particle_xcell_centered(p_i::NTuple, xi::NTuple, dxi::NTuple, F::AbstractArray, icell, jcell)
   
    # cell indices
    idx = (icell, jcell)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xi, dxi, idx)
    # F at the cell corners
    Fi = field_centers(F, p_i, xi, idx)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end


function grid2particle(xi, F::Array{T,N}, particle_coords) where {T,N}
    Fp = zeros(T, np)

    # cell dimensions
    dxi = grid_size(xi)

    # origin of the domain 
    xci = minimum.(xi)
    
    Threads.@threads for i in eachindex(particle_coords[1])
        @inbounds Fp[i] = _grid2particle(
            ntuple(j -> particle_coords[j][i], Val(N)), xci, xi, dxi, F
        )
    end

    return Fp
end

function grid2particle!(Fp, xi, F::Array{T,N}, particle_coords) where {T,N}
    # cell dimensions
    dxi = grid_size(xi)
    # origin of the domain 
    xci = minimum.(xi)
    Threads.@threads for i in eachindex(particle_coords[1])
        if !any(isnan, ntuple(j -> particle_coords[j][i], Val(N)))
            @inbounds Fp[i] = _grid2particle(
                ntuple(j -> particle_coords[j][i], Val(N)), xci, xi, dxi, F
            )
        end
    end
end


function grid2particle_xcell!(Fp, xi, F::Array{T,N}, particle_coords, max_xcell) where {T,N}
    # cell dimensions
    dxi = grid_size(xi)
    # origin of the domain 
    # xci = minimum.(xi)
    nx, ny = length.(xi)
    # Threads.@threads 
    for jcell in 1:ny
        for icell in 1:nx
            _grid2particle_xcell!(
                Fp, particle_coords, xi, dxi, F, max_xcell, icell, jcell
            )
        end
    end
end

function _grid2particle_xcell!(Fp, p::NTuple, xi::NTuple, dxi::NTuple, F::AbstractArray, max_xcell, icell, jcell)
    idx = (icell, jcell)
    # dx_2, dy_2 = 0.5.*dxi
    # xlo, xhi = extrema(xi[1])
    # ylo, yhi = extrema(xi[1])
    # xi .-= dxi
    # function correct_pxy(px_i, py_i)
    #     px_i = px_i > xlo ? px_i : px_i + dx_2 
    #     px_i = px_i < xhi ? px_i : px_i - dx_2 
    #     py_i = py_i > ylo ? py_i : py_i + dy_2 
    #     py_i = py_i < yhi ? py_i : py_i - dy_2 
    #     return px_i, py_i
    # end

    for i in 1:max_xcell
        # check that the particle is inside the grid
        # isinside(p, xi)

        # indices of lowermost-left corner of the cell 
        # containing the particle
        px_i = p[1][i, icell, jcell]
        py_i = p[2][i, icell, jcell]

        p_i = (px_i, py_i)

        any(isnan, p_i) && continue

        # normalize particle coordinates
        ti = normalize_coordinates(p_i, xi, dxi, idx)

        # F at the cell corners
        Fi = field_centers(F, p_i, xi, idx)

        # Interpolate field F onto particle
        Fp[i, icell, jcell] = ndlinear(ti, Fi)

        # isnan(Fp[i, icell, jcell]) && @show ti, Fi, p_i, xi, dxi, idx
    end
end

## CUDA

function _grid2particle!(
    Fp::CuDeviceVector{T,1},
    p::NTuple{N,CuDeviceVector{T,1}},
    dxi::NTuple{N,T},
    xci::NTuple{N,B},
    xi::NTuple{N,A},
    F::CuDeviceArray{T,N},
    n::Integer,
) where {T,A,B,N}
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if ix ≤ n
        pix = particle2tuple(p, ix)

        if !any(isnan, pix)
            # check that the particle is inside the grid
            # isinside(pix, xi)

            # indices of lowermost-left corner of the cell 
            # containing the particle
            idx = parent_cell(pix, dxi, xci)

            # normalize particle coordinates
            ti = normalize_coordinates(pix, xi, dxi, idx)

            # F at the cell corners
            Fi = field_corners(F, idx)

            # Interpolate field F onto particle
            Fp[ix] = ndlinear(ti, Fi)
        end
    end

    return nothing
end

function grid2particle(
    xi, Fd::CuArray{T,N}, particle_coords::NTuple{N,CuArray}; nt=512
) where {T,N}
    dxi = grid_size(xi)
    n = length(particle_coords[1])
    Fpd = CuArray{T,1}(undef, n)
    # origin of the domain 
    xci = minimum.(xi)
    numblocks = ceil(Int, n / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _grid2particle!(
            Fpd, particle_coords, dxi, xci, xi, Fd, n
        )
    end

    return Fpd
end

function grid2particle!(
    Fpd::CuArray{T,1}, xi, Fd::CuArray{T,N}, particle_coords::NTuple{N,CuArray}; nt=512
) where {T,N}
    dxi = grid_size(xi)
    n = length(particle_coords[1])
    # origin of the domain 
    xci = minimum.(xi)
    numblocks = ceil(Int, n / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _grid2particle!(
            Fpd, particle_coords, dxi, xci, xi, Fd, n
        )
    end
end
