using CSV
using DataFrames
using Statistics
using CUDA

import Pkg; Pkg.activate(".")
using StencilInterpolation

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
    F0 = deepcopy(F)

    ## CPU -----------------------------------------------------------------------------------------

    # scattering operation (for now just bi-linear interpolation)
    t_scatter_cpu = @elapsed Fp = scattering( (x, y), (dx, dy), F, particle_coords)

    # gathering operation (inverse distance weighting)
    t_gather_cpu = @elapsed gathering!(F, Fp, (x, y), (dx, dy), particle_coords)

    # Compute error
    sol = [-sin(2*yi)*cos(3*π*xi) for (xi, yi) in zip(px,py)]
    misfit_scatter = @.(log10(abs(Fp-sol)))
    misfit_gather = @.(log10(abs(F-F0)))

    ## CUDA -----------------------------------------------------------------------------------------

    Fpd = CUDA.zeros(Float64, N)
    Fd = CuArray(F)
    Fd0 = deepcopy(Fd)

    # scattering operation (for now just bi-linear interpolation)
    t_scatter_cuda = @elapsed scattering!(Fpd, (x, y), (dx, dy), Fd, CuArray.(particle_coords))

    # gathering operation (inverse distance weighting)
    particle_coords_dev = CuArray.(particle_coords)
    fill!(Fd, 0.0)
    t_gather_cuda = @elapsed gathering!(Fd, Fpd, (x, y), (dx, dy), particle_coords_dev)
       
    # Compute error
    sol_gpu = CuArray(sol)
    misfit_scatter_cuda = @.(log10(abs(Fpd-sol_gpu)))
    misfit_gather_cuda = @.(log10(abs(Fd-Fd0)))

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
        s = scatter!(a, px, py, color=misfit_scatter, colormap=:batlow)
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
        s = scatter!(a, px, py, color=Array(misfit_scatter_cuda), colormap=:batlow)
        xlims!(0, lx)
        ylims!(0, ly)
        Colorbar(f[1,4], s)
        display(f)
    end

    e1_cpu, e1_cuda = mean(Fp.-sol), mean(Fpd.-sol_gpu)
    # return t_scatter_cpu, t_scatter_cuda, norm(Fp.-sol, 2)*dx*dy, norm(Fpd.-sol_gpu, 2)*dx*dy
    return t_scatter_cpu, t_scatter_cuda, t_gather_cpu, t_gather_cuda,  e1_cpu, e1_cuda
end

function perf_test()
    
    df = DataFrame(
        threads=Int16[], nx=Int64[], ny=Int64[], nxcell=Float64[],
        t_scatter_cpu=Float64[], t_scatter_cuda=Float64[], t_gather_cpu=Float64[], t_gather_cuda=Float64[], 
        error_cpu=Float64[], error_cuda=Float64[])
    nx = ny = 512*2
    for nxcell in (4, 25, 50, 75, 100)
        out = main(nx, ny, nxcell)
        push!(df, [Threads.nthreads() nx ny nxcell out...])
    end

    CSV.write("scatter_perf_$(Threads.nthreads())_fma_nxcell.csv", df)
    
end

perf_test()

files = readdir(".")[endswith.(readdir("."), "csv")]

out = [CSV.read(f, DataFrame) for f in files]

## TIMINGS
f = Figure(fontsize=20)

ax1 = Axis(f[1,1], title="scatter", ylabel="seconds", xscale = log10, yscale = log10)
ax2 = Axis(f[2,1], title="gather", xlabel="particles", ylabel="seconds", xscale = log10, yscale = log10)
for data in out
    np = @. (data.nx*data.ny*data.nxcell)
    lines!(ax1, np[2:end],  data.t_scatter_cpu[2:end], linestyle = :solid, linewidth=3, label = "$(data.threads[1]) threads")
    l=lines!(ax2, np[2:end],  data.t_gather_cpu[2:end], linestyle = :solid, linewidth=3, label = "$(data.threads[1]) threads")
end
lines!(ax1, @.(out[end].nx*out[end].ny*out[end].nxcell)[2:end], out[end].t_scatter_cuda[2:end], 
    color = :black, linestyle = :dash, linewidth=3,  label = "RTX3080")
l=lines!(ax2, @.(out[end].nx*out[end].ny*out[end].nxcell)[2:end], out[end].t_gather_cuda[2:end], 
    color = :black, linestyle = :dash, linewidth=3,  label = "RTX3080")

for ax in (ax1, ax2)
    ylims!(ax, (1e-2, 1e2))
end
hidexdecorations!(ax1, grid=false)
f[3,1] = Legend(f, ax2, orientation=:horizontal, framevisible=false)
f

## SPEEDUP
nt = [data.threads[end] for data in out]
su_scatter = [data.t_scatter_cpu[end]./out[end].t_scatter_cuda[end] for data in out][sortperm(nt)]
su_gather = [data.t_gather_cpu[end]./out[end].t_gather_cuda[end] for data in out][sortperm(nt)]
sort!(nt)

f = Figure(fontsize=20)
ax1 = Axis(f[1,1], ylabel="speed up",  xlabel="threads", yscale=log10)
lines!(ax1, nt, su_scatter, linestyle = :solid, linewidth=3, label="scatter")
lines!(ax1, nt, su_gather, linestyle = :solid, linewidth=3, label="gather")
ax1.xticks = nt
ylims!(ax1, (1, 1e2))
axislegend(ax1)
f
