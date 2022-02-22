function parent_cell(particle_coords::NTuple{N, A}, dxi::NTuple{N, B}) where {N, A, B}
    return ntuple(i -> Int(particle_coords[i] ÷ dxi[i]) + 1, Val(N))
end

@generated function distance(a::NTuple{N, T}, b::NTuple{N, T}) where {T, N}
    ex = zero(T)
    @inbounds for i in 1:N
        ex = :( (a[$i]-b[$i])*(a[$i]-b[$i]) + $ex)
    end
    return :(√($ex))
end

