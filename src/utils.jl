function parent_cell(particle_coords::NTuple{N,A}, dxi::NTuple{N,B}) where {N,A,B}
    return ntuple(i -> Int(particle_coords[i] ÷ dxi[i]) + 1, Val(N))
end

# dimension-agnostic fully unrolled euclidean distance
@generated function distance(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    ex = zero(T)
    @inbounds for i in 1:N
        ex = :((a[$i] - b[$i]) * (a[$i] - b[$i]) + $ex)
    end
    return :(√($ex))
end

# check whether particle is inside the grid (includes boundary)
function isinside(px::Real, py::Real, x, y)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    @assert (xmin ≤ px ≤ xmax) && (ymin ≤ py ≤ ymax)
end

function isinside(px::Real, py::Real, pz::Real, x, y, z)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    zmin, zmax = extrema(z)
    @assert (xmin ≤ px ≤ xmax) && (ymin ≤ py ≤ ymax) && (zmin ≤ pz ≤ zmax)
end

isinside(p::NTuple{2,T}, x, y) where {T} = isinside(p[1], p[2], x, y)
isinside(p::NTuple{3,T}, x, y, z) where {T} = isinside(p[1], p[2], p[3], x, y, z)

# 2D random particle generator for regular grids
function random_particles(nxcell, x, y, dx, dy, nx, ny)
    # number of cells
    ncells = (nx - 1) * (ny - 1)
    # allocate particle coordinate arrays
    px, py = zeros(nxcell * ncells), zeros(nxcell * ncells)
    Threads.@threads for i in 1:nx-1
        @inbounds for j in 1:ny-1
            # lowermost-left corner of the cell
            x0, y0 = x[i], y[j]
            # cell index
            cell = i + (nx - 1) * (j - 1)
            for l in 1:nxcell
                px[(cell-1)*nxcell+l] = rand() * dx + x0
                py[(cell-1)*nxcell+l] = rand() * dy + y0
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
    Threads.@threads for i in 1:nx-1
        @inbounds for j in 1:ny-1, k in 1:nz-1
            # lowermost-left corner of the cell
            x0, y0, z0 = x[i], y[j], z[k]
            # cell index
            cell = i + (nx - 1) * (j - 1) + (nx - 1) * (ny - 1) * (k - 1)
            for l in 1:nxcell
                px[(cell-1)*nxcell+l] = rand() * dx + x0
                py[(cell-1)*nxcell+l] = rand() * dy + y0
                pz[(cell-1)*nxcell+l] = rand() * dz + z0
            end
        end
    end

    return px, py, pz
end
