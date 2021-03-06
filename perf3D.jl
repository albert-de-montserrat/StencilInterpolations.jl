using CSV
using DataFrames
using Statistics
using CUDA
using StencilInterpolations

const viz = false

function random_particles(nxcell, x, y, z, dx, dy, dz, nx, ny, nz)
    ncells = (nx - 1) * (ny - 1) * (nz - 1)
    px, py, pz = zeros(nxcell * ncells), zeros(nxcell * ncells), zeros(nxcell * ncells)
    for i in 1:(nx - 1)
        for j in 1:(ny - 1), k in 1:(nz - 1)
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

function main(nx, ny, nz, nxcell)

    # model domain
    lx = ly = lz = 1
    dx, dy, dz = lx / (nx - 1), ly / (ny - 1), lz / (nz - 1)
    x = LinRange(0, lx, nx)
    y = LinRange(0, ly, ny)
    z = LinRange(0, lz, nz)

    # random particles
    px, py, pz = random_particles(nxcell, x, y, z, dx, dy, dz, nx, ny, nz)
    particle_coords = (px, py, pz)
    N = length(px)

    # field to interpolate
    F = [-sin(2 * zi) * cos(3 * π * xi) for xi in x, _ in y, zi in z]
    F0 = deepcopy(F)

    ## CPU -----------------------------------------------------------------------------------------

    # scattering operation (tri-linear interpolation)
    t_scatter_cpu = @elapsed Fp = scattering((x, y, z), F, particle_coords)

    # gathering operation (inverse distance weighting)
    t_gather_cpu = @elapsed gathering!(F, Fp, (x, y, z), particle_coords)

    # Compute error
    sol = [-sin(2 * zi) * cos(3 * π * xi) for (xi, zi) in zip(px, pz)]
    misfit_scatter = @.(log10(abs(Fp - sol)))
    misfit_gather = @.(log10(abs(F - F0)))

    ## CUDA -----------------------------------------------------------------------------------------

    Fpd = CUDA.zeros(Float64, N)
    Fd = CuArray(F)
    Fd0 = deepcopy(Fd)
    particle_coords_dev = CuArray.(particle_coords)

    # scattering operation (for now just bi-linear interpolation)
    t_scatter_cuda = @elapsed scattering!(Fpd, (x, y, z), Fd, particle_coords_dev)

    # gathering operation (inverse distance weighting)
    fill!(Fd, 0.0)
    t_gather_cuda = @elapsed gathering!(Fd, Fpd, (x, y, z), particle_coords_dev)

    # Compute error
    sol_gpu = CuArray(sol)
    misfit_scatter_cuda = @.(log10(abs(Fpd - sol_gpu)))
    misfit_gather_cuda = @.(log10(abs(Fd - Fd0)))

    println(
        "Finished for Ω ∈ [0,1] × [0,1] × [0,1]; $(nx) × $(ny) × $(nz) nodes; $nxcell particles per cell or $(Float64(N)) particles",
    )

    # Plots CPU
    if viz == true
        xx = vec([x for x in x, y in y, z in z])
        yy = vec([y for x in x, y in y, z in z])
        zz = vec([z for x in x, y in y, z in z])

        f = Figure(; resolution=(2400, 1200), fontsize=20)
        a = Axis(f[1, 1]; aspect=1, title="Analytical")
        scatter!(a, xx, yy, zz; color=vec(F0), colormap=:batlow, markersize=10)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hidexdecorations!(a)

        a = Axis(f[1, 2]; aspect=1, title="Scattered")
        scatter!(a, px, py, pz; color=Fp, colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hidedecorations!(a)

        a = Axis(f[1, 3]; aspect=1, title="log10 error")
        s = scatter!(a, px, py, pz; color=misfit_scatter, colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hidedecorations!(a)

        Colorbar(f[1, 4], s; height=400)

        a = Axis(f[1, 5]; aspect=1, title="Gathered")
        scatter!(a, xx, yy, zz; color=vec(F), colormap=:batlow, markersize=10)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hidedecorations!(a)

        a = Axis(f[1, 6]; aspect=1, title="log10 error")
        s = scatter!(a, xx, yy, zz; color=misfit_gather[:], colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hidedecorations!(a)

        Colorbar(f[1, 7], s; height=400)

        display(f)

        # Plots GPU
        a = Axis(f[2, 1]; aspect=1, title="CUDA")
        scatter!(a, xx, yy, zz; color=vec(F), colormap=:batlow, markersize=10)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)

        a = Axis(f[2, 2]; aspect=1, title="CUDA")
        scatter!(a, px, py; color=Array(Fpd), colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hideydecorations!(a)

        a = Axis(f[2, 3]; aspect=1, title="CUDA")
        s = scatter!(a, px, py; color=Array(misfit_scatter_cuda), colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hideydecorations!(a)
        Colorbar(f[2, 4], s; height=400)

        a = Axis(f[2, 5]; aspect=1, title="CUDA")
        scatter!(a, xx, yy, zz; color=vec(Array(Fd)), colormap=:batlow, markersize=10)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hideydecorations!(a)

        a = Axis(f[2, 6]; aspect=1, title="CUDA")
        s = scatter!(a, xx, yy, zz; color=Array(misfit_gather_cuda[:]), colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        ylims!(0, lz)
        hideydecorations!(a)

        Colorbar(f[2, 7], s; height=400)

        display(f)
    end

    e1_cpu, e1_cuda = mean(Fp .- sol), mean(Fpd .- sol_gpu)
    return t_scatter_cpu, t_scatter_cuda, t_gather_cpu, t_gather_cuda, e1_cpu, e1_cuda
end

function perf_test()
    df = DataFrame(;
        threads=Int16[],
        nx=Int64[],
        ny=Int64[],
        nz=Int64[],
        nxcell=Float64[],
        t_scatter_cpu=Float64[],
        t_scatter_cuda=Float64[],
        t_gather_cpu=Float64[],
        t_gather_cuda=Float64[],
        error_cpu=Float64[],
        error_cuda=Float64[],
    )

    nx = ny = nz = 128
    for nxcell in (4, 15, 25, 50)
        out = main(nx, ny, nz, nxcell)
        push!(df, [Threads.nthreads() nx ny nz nxcell out...])
    end

    return CSV.write("scatter_perf_$(Threads.nthreads())_nxcell_fma_3D.csv", df)
end

perf_test()
