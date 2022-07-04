function distance_weigth(a::NTuple{N,T}, b::NTuple{N,T}; order=2) where {N,T}
    return one(T) / distance(a, b)^order
end

## CPU 2D

@inbounds function _gathering!(upper, lower, Fpi, p, x, y, dxi, order)

    # check that the particle is inside the grid
    # isinside(p, (x, y))

    # indices of lowermost-left corner of   
    # the cell containing the particlex
    idx_x, idx_y = parent_cell(p, dxi)

    ω = (
        distance_weigth((x[idx_x], y[idx_y]), p; order=order),
        distance_weigth((x[idx_x + 1], y[idx_y]), p; order=order),
        distance_weigth((x[idx_x], y[idx_y + 1]), p; order=order),
        distance_weigth((x[idx_x + 1], y[idx_y + 1]), p; order=order),
    )

    nt = Threads.threadid()

    upper[nt][idx_x, idx_y] += ω[1] * Fpi
    upper[nt][idx_x + 1, idx_y] += ω[2] * Fpi
    upper[nt][idx_x, idx_y + 1] += ω[3] * Fpi
    upper[nt][idx_x + 1, idx_y + 1] += ω[4] * Fpi
    lower[nt][idx_x, idx_y] += ω[1]
    lower[nt][idx_x + 1, idx_y] += ω[2]
    lower[nt][idx_x, idx_y + 1] += ω[3]

    return lower[nt][idx_x + 1, idx_y + 1] += ω[4]
end

function gathering!(F::Array{T,2}, Fp::Vector{T}, xi, particle_coords; order=2) where {T}

    # unpack tuples
    px, py = particle_coords
    x, y = xi
    dxi = (x[2] - x[1], y[2] - y[1])

    # number of particles
    np = length(Fp)

    # TODO think about pre-allocating these 2 buffers
    upper = [zeros(size(F)) for _ in 1:Threads.nthreads()]
    lower = [zeros(size(F)) for _ in 1:Threads.nthreads()]

    # compute ∑ωᵢFᵢ and ∑ωᵢ
    Threads.@threads for i in 1:np
        if !isnan(px[i]) && !isnan(py[i])
            _gathering!(upper, lower, Fp[i], (px[i], py[i]), x, y, dxi, order)
        end
    end

    # compute Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] =
            sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
            sum(lower[nt][i] for nt in 1:Threads.nthreads())
    end
end

## CPU 3D

@inbounds function _gathering!(upper, lower, Fpi, p, x, y, z, dxi, order)
    # check that the particle is inside the grid
    # isinside(p, x, y, z)

    # indices of lowermost-left corner of   
    # the cell containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    ω = (
        distance_weigth((x[idx_x], y[idx_y], z[idx_z]), p; order=order),
        distance_weigth((x[idx_x + 1], y[idx_y], z[idx_z]), p; order=order),
        distance_weigth((x[idx_x], y[idx_y + 1], z[idx_z]), p; order=order),
        distance_weigth((x[idx_x + 1], y[idx_y + 1], z[idx_z]), p; order=order),
        distance_weigth((x[idx_x], y[idx_y], z[idx_z + 1]), p; order=order),
        distance_weigth((x[idx_x + 1], y[idx_y], z[idx_z + 1]), p; order=order),
        distance_weigth((x[idx_x], y[idx_y + 1], z[idx_z + 1]), p; order=order),
        distance_weigth((x[idx_x + 1], y[idx_y + 1], z[idx_z + 1]), p; order=order),
    )

    nt = Threads.threadid()

    upper[nt][idx_x, idx_y, idx_z] += ω[1] * Fpi
    upper[nt][idx_x + 1, idx_y, idx_z] += ω[2] * Fpi
    upper[nt][idx_x, idx_y + 1, idx_z] += ω[3] * Fpi
    upper[nt][idx_x + 1, idx_y + 1, idx_z] += ω[4] * Fpi
    upper[nt][idx_x, idx_y, idx_z + 1] += ω[5] * Fpi
    upper[nt][idx_x + 1, idx_y, idx_z + 1] += ω[6] * Fpi
    upper[nt][idx_x, idx_y + 1, idx_z + 1] += ω[7] * Fpi
    upper[nt][idx_x + 1, idx_y + 1, idx_z + 1] += ω[8] * Fpi
    lower[nt][idx_x, idx_y, idx_z] += ω[1]
    lower[nt][idx_x + 1, idx_y, idx_z] += ω[2]
    lower[nt][idx_x, idx_y + 1, idx_z] += ω[3]
    lower[nt][idx_x + 1, idx_y + 1, idx_z] += ω[4]
    lower[nt][idx_x, idx_y, idx_z + 1] += ω[5]
    lower[nt][idx_x + 1, idx_y, idx_z + 1] += ω[6]
    lower[nt][idx_x, idx_y + 1, idx_z + 1] += ω[7]

    return lower[nt][idx_x + 1, idx_y + 1, idx_z + 1] += ω[8]
end

function gathering!(F::Array{T,3}, Fp::Vector{T}, xi, particle_coords; order=2) where {T}
    # unpack tuples
    px, py, pz = particle_coords
    x, y, z = xi
    dxi = (x[2] - x[1], y[2] - y[1], z[2] - z[1])

    # number of particles
    np = length(Fp)

    # TODO think about pre-allocating these 2 buffers
    upper = [zeros(size(F)) for _ in 1:Threads.nthreads()]
    lower = [zeros(size(F)) for _ in 1:Threads.nthreads()]

    # compute ∑ωᵢFᵢ and ∑ωᵢ
    Threads.@threads for i in 1:np
        if !isnan(px[i]) && !isnan(py[i]) && !isnan(pz[i])
            _gathering!(upper, lower, Fp[i], (px[i], py[i], pz[i]), x, y, z, dxi, order)
        end
    end

    # compute Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] =
            sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
            sum(lower[nt][i] for nt in 1:Threads.nthreads())
    end
end

## CUDA 2D

function _gather1!(
    upper::CuDeviceMatrix{T,1},
    lower::CuDeviceMatrix{T,1},
    Fpd::CuDeviceVector{T,1},
    xi,
    dxi,
    p;
    order=4,
) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    px, py = p
    x, y = xi

    @inbounds if idx ≤ length(px)
        # check that the particle is inside the grid
        # isinside(px[idx], py[idx], x, y)

        p_idx = (px[idx], py[idx])

        if !any(isnan, p_idx)
            # indices of lowermost-left corner of
            # the cell containing the particle
            idx_x, idx_y = parent_cell(p_idx, dxi)

            ω1::Float64 = distance_weigth((x[idx_x], y[idx_y]), p_idx; order=order)
            ω2::Float64 = distance_weigth((x[idx_x + 1], y[idx_y]), p_idx; order=order)
            ω3::Float64 = distance_weigth((x[idx_x], y[idx_y + 1]), p_idx; order=order)
            ω4::Float64 = distance_weigth((x[idx_x + 1], y[idx_y + 1]), p_idx; order=order)

            Fpi::Float64 = Fpd[idx]

            CUDA.@atomic upper[idx_x, idx_y] += ω1 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y] += ω2 * Fpi
            CUDA.@atomic upper[idx_x, idx_y + 1] += ω3 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y + 1] += ω4 * Fpi
            CUDA.@atomic lower[idx_x, idx_y] += ω1
            CUDA.@atomic lower[idx_x + 1, idx_y] += ω2
            CUDA.@atomic lower[idx_x, idx_y + 1] += ω3
            CUDA.@atomic lower[idx_x + 1, idx_y + 1] += ω4
        end
    end

    return nothing
end

function _gather2!(Fd::CuDeviceArray{T,2}, upper, lower) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (idx < size(Fd, 1)) && (idy < size(Fd, 2))
        @inbounds Fd[idx, idy] = upper[idx, idy] / lower[idx, idy]
    end

    return nothing
end

function gathering!(
    Fd::CuArray{T,2}, Fpd::CuArray{T,1}, xi, particle_coords; nt=512
) where {T}
    upper = CUDA.zeros(T, size(Fd))
    lower = CUDA.zeros(T, size(Fd))
    x, y = xi
    dxi = (x[2] - x[1], y[2] - y[1])

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    N = length(Fpd)
    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gather1!(
            upper, lower, Fpd, xi, dxi, particle_coords
        )
    end

    # seond and final kernel that computes Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    nx, ny = size(Fd)
    nblocksx = ceil(Int, nx / 32)
    nblocksy = ceil(Int, ny / 32)
    CUDA.@sync begin
        @cuda threads = (32, 32) blocks = (nblocksx, nblocksy) _gather2!(
            Fd, upper, lower
        )
    end
end

## CUDA 3D 

function _gather1!(
    upper::CuDeviceArray{T,3},
    lower::CuDeviceArray{T,3},
    Fpd::CuDeviceVector{T,1},
    xi,
    dxi,
    p;
    order=2,
) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    px, py, pz = p
    x, y, z = xi

    @inbounds if idx ≤ length(px)
        # check that the particle is inside the grid
        # isinside(px[idx], py[idx], pz[idx], x, y, z)

        p_idx = (px[idx], py[idx], pz[idx])

        if !any(isnan, p_idx)
            # indices of lowermost-left corner of
            # the cell containing the particle
            idx_x, idx_y, idx_z = parent_cell(p_idx, dxi)

            ω1::Float64 = distance_weigth((x[idx_x], y[idx_y], z[idx_z]), p_idx; order=order)
            ω2::Float64 = distance_weigth(
                (x[idx_x + 1], y[idx_y], z[idx_z]), p_idx; order=order
            )
            ω3::Float64 = distance_weigth(
                (x[idx_x], y[idx_y + 1], z[idx_z]), p_idx; order=order
            )
            ω4::Float64 = distance_weigth(
                (x[idx_x + 1], y[idx_y + 1], z[idx_z]), p_idx; order=order
            )
            ω5::Float64 = distance_weigth(
                (x[idx_x], y[idx_y], z[idx_z + 1]), p_idx; order=order
            )
            ω6::Float64 = distance_weigth(
                (x[idx_x + 1], y[idx_y], z[idx_z + 1]), p_idx; order=order
            )
            ω7::Float64 = distance_weigth(
                (x[idx_x], y[idx_y + 1], z[idx_z + 1]), p_idx; order=order
            )
            ω8::Float64 = distance_weigth(
                (x[idx_x + 1], y[idx_y + 1], z[idx_z + 1]), p_idx; order=order
            )

            Fpi::Float64 = Fpd[idx]

            CUDA.@atomic upper[idx_x, idx_y, idx_z] += ω1 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y, idx_z] += ω2 * Fpi
            CUDA.@atomic upper[idx_x, idx_y + 1, idx_z] += ω3 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y + 1, idx_z] += ω4 * Fpi
            CUDA.@atomic upper[idx_x, idx_y, idx_z + 1] += ω5 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y, idx_z + 1] += ω6 * Fpi
            CUDA.@atomic upper[idx_x, idx_y + 1, idx_z + 1] += ω7 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y + 1, idx_z + 1] += ω8 * Fpi
            CUDA.@atomic lower[idx_x, idx_y, idx_z] += ω1
            CUDA.@atomic lower[idx_x + 1, idx_y, idx_z] += ω2
            CUDA.@atomic lower[idx_x, idx_y + 1, idx_z] += ω3
            CUDA.@atomic lower[idx_x + 1, idx_y + 1, idx_z] += ω4
            CUDA.@atomic lower[idx_x, idx_y, idx_z + 1] += ω5
            CUDA.@atomic lower[idx_x + 1, idx_y, idx_z + 1] += ω6
            CUDA.@atomic lower[idx_x, idx_y + 1, idx_z + 1] += ω7
            CUDA.@atomic lower[idx_x + 1, idx_y + 1, idx_z + 1] += ω8
        end
    end

    return nothing
end

function _gather2!(
    Fd::CuDeviceArray{T,3}, upper::CuDeviceArray{T,3}, lower::CuDeviceArray{T,3}
) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    idz = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (idx ≤ size(Fd, 1)) && (idy ≤ size(Fd, 2)) && (idz ≤ size(Fd, 3))
        @inbounds Fd[idx, idy, idz] = upper[idx, idy, idz] / lower[idx, idy, idz]
    end

    return nothing
end

function gathering!(
    Fd::CuArray{T,3}, Fpd::CuArray{T,1}, xi, particle_coords; nt=512
) where {T}
    x, y, z = xi
    dxi = (x[2] - x[1], y[2] - y[1], z[2] - z[1])
    upper = CUDA.zeros(T, size(Fd))
    lower = CUDA.zeros(T, size(Fd))

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    N = length(Fpd)
    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gather1!(
            upper, lower, Fpd, xi, dxi, particle_coords
        )
    end

    # second and final kernel that computes Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    nx, ny, nz = size(Fd)
    nblocksx = ceil(Int, nx / 8)
    nblocksy = ceil(Int, ny / 8)
    nblocksz = ceil(Int, nz / 8)
    CUDA.@sync begin
        @cuda threads = (8, 8, 8) blocks = (nblocksx, nblocksy, nblocksz) _gather2!(
            Fd, upper, lower
        )
    end
end
