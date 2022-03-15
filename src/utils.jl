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
