import Pkg; Pkg.activate(".")
using Statistics
using StencilInterpolation
using CUDA
using CSV
using DataFrames

const viz = false

function random_particles(nxcell, x, y, dx, dy, nx, ny)
    ncells = (nx-1)*(ny-1)
    px, py = zeros(nxcell*ncells), zeros(nxcell*ncells)
    dxdy = dx*dy
    for idx_x in 1:nx-1, idx_y in 1:ny-1
        x0, y0 = x[idx_x], y[idx_y]
        
        cell = idx_x + (nx-1) * (idx_y-1)
        for i in 1:nxcell
            px[(cell-1)*nxcell+i], py[(cell-1)*nxcell+i] = rand()*dxdy+x0, rand()*dxdy+y0
        end
    end
    return px, py
end

function main(nx, ny, nxcell)

    # println("Ω ∈ [0,1] × [0,1]; $(nx) × $(nx) nodes; $N particles")
    # println("Ω ∈ [0,1] × [0,1]; $(nx) × $(nx) nodes; $nxcell particles per cell")

    # model domain
    lx = ly = 1
    dx, dy = lx/(nx-1), ly/(ny-1)
    x = LinRange(0, lx, nx)
    y = LinRange(0, ly, ny)

    # random particles
    px, py = random_particles(nxcell, x, y, dx, dy, nx, ny)
    # px, py = rand(N), rand(N)
    particle_coords = (px, py)
    p = (px, py)
    N = length(px)

    # random field
    F = [-sin(2*yi)*cos(3*π*xi) for xi in x, yi in y]

    # Scattering operation (for now just bi-linear interpolation)
    t_cpu = @elapsed Fp = scattering( (x, y), (dx, dy), F, particle_coords);

    # Compute error
    sol = [-sin(2*yi)*cos(3*π*xi) for (xi, yi) in zip(px,py)]
    misfit = @.(log10(abs(Fp-sol)))
    # println("average log10 error = $(mean(misfit))")

    ## CUDA

    Fpd = CUDA.zeros(Float64, N)
    Fd = CuArray(F)

    t_gpu = @elapsed scattering!(Fpd, (x, y), (dx, dy), Fd, CuArray.(particle_coords));

    # Compute error
    sol_gpu = CuArray(sol)
    misfit_gpu = @.(log10(abs(Fpd-sol_gpu)))
    # println("average log10 error = $(mean(misfit))")

    println("Finished for Ω ∈ [0,1] × [0,1]; $(nx) × $(nx) nodes; $nxcell particles per cell or $(Float64(N)) particles")

    # Plots CPU
    if viz == true
        f = Figure(resolution=(1600, 600))
        a = Axis(f[1,1], aspect=1, title="Analytical")
        heatmap!(a, x, y, F, colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)

        a = Axis(f[1, 2], aspect=1, title="Scattered")
        scatter!(a, px, py, color=Fp, colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)

        a = Axis(f[1, 3], aspect=1, title="log10 error")
        s = scatter!(a, px, py, color=misfit, colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        Colorbar(f[1,4], s)
        display(f)

        # Plots GPU
        f = Figure(resolution=(1600, 600))
        a = Axis(f[1,1], aspect=1, title="Analytical")
        heatmap!(a, x, y, F, colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)

        a = Axis(f[1, 2], aspect=1, title="Scattered")
        scatter!(a, px, py, color=Array(Fpd), colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)

        a = Axis(f[1, 3], aspect=1, title="log10 error")
        s = scatter!(a, px, py, color=Array(misfit_gpu), colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        Colorbar(f[1,4], s)
        display(f)
    end
    # return t_cpu, t_gpu, norm(Fp.-sol, 2)*dx*dy, norm(Fpd.-sol_gpu, 2)*dx*dy
    return t_cpu, t_gpu, mean(Fp.-sol), mean(Fpd.-sol_gpu)
end

function perf_test()
    
    df = DataFrame(threads=Int16[], nx=Int64[], ny=Int64[], nxcell=Float64[], t_cpu=Float64[], t_cuda=Float64[], error_cpu=Float64[], error_cuda=Float64[])
    nx = ny = 1000 # 501
    for nxcell in (4, 12, 16, 24, 32, 50)
        t_cpu, t_gpu, misfit, misfit_gpu = main(nx, ny, nxcell)
        push!(df, [Threads.nthreads() nx ny nxcell t_cpu t_gpu misfit misfit_gpu])
    end

    CSV.write("scatter_perf_$(Threads.nthreads())_fma_nxcell.csv", df)
    
end

perf_test()
