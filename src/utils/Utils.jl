function parent_cell(particle_coords::NTuple{N, A}, dxi::NTuple{N, B}) where {N, A, B}
    return ntuple(i -> Int(particle_coords[i] ÷ dxi[i]) + 1, Val(N))
end
