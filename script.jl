using CUDA
using Pkg
Pkg.activate(".")

using StencilInterpolations

function random_particles(nxcell, x, y, dx, dy, nx, ny)
    px, py = zeros(nxcell, nx - 1, ny - 1), zeros(nxcell, nx - 1, ny - 1)
    for j in 1:(ny - 1), i in 1:(nx - 1)
        # lowermost-left corner of the cell
        x0, y0 = x[i], y[j]
        # cell index
        for l in 1:nxcell
            px[l, i, j] = rand() * dx + x0
            py[l, i, j] = rand() * dy + y0
        end
    end
    return px, py
end

@inline @generated function bilinear_weight(
    a::NTuple{N,T}, b::NTuple{N,T}, dxi::NTuple{N,T}
) where {N,T}
    quote
        val = one(T)
        Base.Cartesian.@nexprs $N i -> one(T) - abs(a[i] - b[i]) / dxi[i]
        return val
    end
end

function foo!(F::CuArray{T,2}, Fp::CuArray{T,N}, xi, particle_coords) where {T,N}
    px, py = particle_coords
    x, y = xi
    dxi = (x[2] - x[1], y[2] - y[1])

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    nxcell, ny, nz = size(px)
    nblocksx = ceil(Int, ny / 32)
    nblocksy = ceil(Int, nz / 32)
    threadsx = 32
    threadsy = 32

    shmem_size = (3 * sizeof(T) * nxcell * threadsx * threadsy)

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _foo!(
            F, Fp, xi, px, py, dxi
        )
    end
end

function _foo!(F, Fp, xi, px, py, dxi)
    icell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jcell = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (icell ≤ size(px, 2)) && (jcell ≤ size(px, 3))

        # unpack tuples
        xc_cell = (xi[1][icell], xi[2][jcell]) # cell center coordinates
        ω, ωxF = 0.0, 0.0 # init weights
        max_xcell = size(px, 1) # max particles per cell

        for i in 1:max_xcell
            p_i = (px[i, icell, jcell], py[i, icell, jcell])
            any(isnan, p_i) && continue # ignore unused allocations
            ω_i = bilinear_weight(xc_cell, p_i, dxi)
            ω += ω_i
            ωxF += ω_i * Fp[i, icell, jcell]
        end

        F[icell + 1, jcell + 1] = ωxF / ω
    end

    return nothing
end

foo!(F, Fp, xi, particle_coords)

function main(; nx=41, ny=41, nxcell=4)
    nx = 41
    ny = 41
    nxcell = 4

    # model domain
    lx = ly = 1
    dx, dy = lx / (nx - 1), ly / (ny - 1)
    dxi = (dx, dy)
    x = LinRange(0, lx, nx)
    y = LinRange(0, ly, ny)

    xc = LinRange(0 - dx / 2, lx * dx / 2, nx + 1)
    yc = LinRange(0 - dy / 2, ly * dy / 2, ny + 1)
    xi = (x, y)

    # random particles
    px, py = random_particles(nxcell, x, y, dx, dy, nx, ny)
    np = prod(size(px))
    Fp = rand(np)
    # field to interpolate
    F = [-sin(2 * yi) * cos(3 * π * xi) for xi in xc, yi in yc]

    # move to CUDA    
    px = CuArray(px)
    py = CuArray(py)
    particle_coords = (px, py)
    F = CUDA.zeros(Float64, nx + 2, ny + 2)
    Fp = CuArray(Fp)

    gathering_xcell!(F, Fp, xi, particle_coords)

    return grid2particle_xcell!(Fp, xi, F, particle_coords)
end
