
distance_weigth(a::NTuple{N, T}, b::NTuple{N, T}; order = 4) where {T, N} = 1/distance(a, b)^order

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

function gathering!(F::Array{T, N}, Fp::Vector{T}, xi, dxi, particle_coords; order = 4) where {T, N}
    
    # unpack tuples
    px, py = particle_coords
    x, y = xi

    np = length(Fp)

    # upper = zeros(size(F)) # we can recycle F here
    upper = [zeros(size(F)) for _ in 1:Threads.nthreads()]
    lower = [zeros(size(F)) for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:np
        _gathering!(upper, lower, Fp[i], (px[i], py[i]), x, y, dxi, order)
    end

    Threads.@threads for i in eachindex(F)
        @inbounds F[i] = 
            sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
            sum(lower[nt][i] for nt in 1:Threads.nthreads())
    end

end

