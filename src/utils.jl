@inline function parent_cell(p::NTuple{N,A}, dxi::NTuple{N,B}, xci::NTuple{N,B}) where {N,A,B}
    return ntuple(i -> Int((p[i] - xci[i]) ÷ dxi[i] + 1), Val(N))
end

# dimension-agnostic fully unrolled euclidean distance
@generated function distance(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    ex = zero(T)
    @inbounds for i in 1:N
        ex = :((a[$i] - b[$i])^2 + $ex)
    end
    return :(√($ex))
end

# check whether particle is inside the grid (includes boundary)
@inline function isinside(px::Real, py::Real, x, y)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    @assert (px === NaN) || (py === NaN) (xmin ≤ px ≤ xmax) && (ymin ≤ py ≤ ymax)
end

@inline function isinside(px::Real, py::Real, pz::Real, x, y, z)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    zmin, zmax = extrema(z)
    @assert (px === NaN) ||
        (py === NaN) ||
        (pz === NaN) ||
        (xmin ≤ px ≤ xmax) && (ymin ≤ py ≤ ymax) && (zmin ≤ pz ≤ zmax)
end

@inline function isinside(p::NTuple{2,T1}, x::NTuple{2,T2}) where {T1,T2}
    return isinside(p[1], p[2], x[1], x[2])
end

@inline function isinside(p::NTuple{3,T1}, x::NTuple{3,T2}) where {T1,T2}
    return isinside(p[1], p[2], p[3], x[1], x[2], x[3])
end

# normalize coordinates
@inline function normalize_coordinates(
    p::NTuple{N,A}, xi::NTuple{N,B}, dxi::NTuple{N,C}, idx::NTuple{N,D}
) where {N,A,B,C,D}
    return ntuple(i -> (p[i] - xi[i][idx[i]]) * (1 / dxi[i]), Val(N))
end

# compute grid size
function grid_size(x::NTuple{N,T}) where {T,N}
    return ntuple(i -> x[i][2] - x[i][1], Val(N))
end

# Get field F at the corners of a given cell
@inline function field_corners(F::AbstractArray{T,2}, idx::NTuple{2,Integer}) where {T}
    idx_x, idx_y = idx
    return (
        F[idx_x, idx_y], F[idx_x + 1, idx_y], F[idx_x, idx_y + 1], F[idx_x + 1, idx_y + 1]
    )
end

@inline function field_corners(F::AbstractArray{T,3}, idx::NTuple{3,Integer}) where {T}
    idx_x, idx_y, idx_z = idx
    return (
        F[idx_x, idx_y, idx_z],   # v000
        F[idx_x + 1, idx_y, idx_z],   # v100
        F[idx_x, idx_y, idx_z + 1], # v001
        F[idx_x + 1, idx_y, idx_z + 1], # v101
        F[idx_x, idx_y + 1, idx_z],   # v010
        F[idx_x + 1, idx_y + 1, idx_z],   # v110
        F[idx_x, idx_y + 1, idx_z + 1], # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )
end

@inline function particle2tuple(p::NTuple{N,AbstractArray}, ix) where {N}
    return ntuple(i -> p[i][ix], Val(N))
end

# 2D random particle generator for regular grids
function random_particles(nxcell, x, y, dx, dy, nx, ny)
    # number of cells
    ncells = (nx - 1) * (ny - 1)
    # allocate particle coordinate arrays
    px, py = zeros(nxcell * ncells), zeros(nxcell * ncells)
    Threads.@threads for i in 1:(nx - 1)
        @inbounds for j in 1:(ny - 1)
            # lowermost-left corner of the cell
            x0, y0 = x[i], y[j]
            # cell index
            cell = i + (nx - 1) * (j - 1)
            for l in 1:nxcell
                px[(cell - 1) * nxcell + l] = rand() * dx + x0
                py[(cell - 1) * nxcell + l] = rand() * dy + y0
            end
        end
    end

    return px, py
end

# 3D random particle generator for regular grids
function random_particles(nxcell, x, y, z, dx, dy, dz, nx, ny, nz)
    # number of cells
    ncells = (nx - 1) * (ny - 1) * (nz - 1)
    # allocate particle coordinate arrays
    px, py, pz = zeros(nxcell * ncells), zeros(nxcell * ncells), zeros(nxcell * ncells)
    Threads.@threads for i in 1:(nx - 1)
        @inbounds for j in 1:(ny - 1), k in 1:(nz - 1)
            # lowermost-left corner of the cell
            x0, y0, z0 = x[i], y[j], z[k]
            # cell index
            cell = i + (nx - 1) * (j - 1) + (nx - 1) * (ny - 1) * (k - 1)
            for l in 1:nxcell
                px[(cell - 1) * nxcell + l] = rand() * dx + x0
                py[(cell - 1) * nxcell + l] = rand() * dy + y0
                pz[(cell - 1) * nxcell + l] = rand() * dz + z0
            end
        end
    end

    return px, py, pz
end
