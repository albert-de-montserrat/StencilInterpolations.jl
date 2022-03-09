using CSV
using CairoMakie
using DataFrames
using Statistics
using CUDA

import Pkg; Pkg.activate(".")
using StencilInterpolation

const viz = false

function random_particles(nxcell, x, y, z, dx, dy, dz, nx, ny, nz)
    ncells = (nx-1)*(ny-1)*(nz-1)
    px, py, pz = zeros(nxcell*ncells), zeros(nxcell*ncells), zeros(nxcell*ncells)
    for i in 1:nx-1
        for j in 1:ny-1, k in 1:nz-1
            # lowermost-left corner of the cell
            x0, y0, z0 = x[i], y[j], z[k]
            # cell index
            cell = i + (nx-1)*(j-1) + (nx-1)*(ny-1)*(k-1)
            for l in 1:nxcell
                px[(cell-1)*nxcell + l] = rand()*dx + x0
                py[(cell-1)*nxcell + l] = rand()*dy + y0
                pz[(cell-1)*nxcell + l] = rand()*dz + z0
            end
        end
    end
    return px, py, pz
end

function main(nx, ny, nz, nxcell)

    # model domain
    lx = ly = lz = 1
    dx, dy, dz = lx/(nx-1), ly/(ny-1), lz/(nz-1)
    x = LinRange(0, lx, nx)
    y = LinRange(0, ly, ny)
    z = LinRange(0, lz, nz)

    # random particles
    px, py, pz = random_particles(nxcell, x, y, z, dx, dy, dz, nx, ny, nz)
    particle_coords = (px, py, pz)
    N = length(px)

    # random field
    F = [-sin(2*zi)*cos(3*π*xi) for xi in x, _ in y, zi in z]
    F0 = deepcopy(F)

    ## CPU -----------------------------------------------------------------------------------------

    # scattering operation (tri-linear interpolation)
    t_scatter_cpu = @elapsed Fp = scattering( (x, y, z), (dx, dy, dz), F, particle_coords)

    # gathering operation (inverse distance weighting)
    t_gather_cpu = @elapsed gathering!(F, Fp, (x, y, z), (dx, dy, dz), particle_coords)

    # Compute error
    sol = [-sin(2*zi)*cos(3*π*xi) for (xi, zi) in zip(px, pz)]
    misfit_scatter = @.(log10(abs(Fp-sol)))
    misfit_gather = @.(log10(abs(F-F0)))

    ## CUDA -----------------------------------------------------------------------------------------

    Fpd = CUDA.zeros(Float64, N)
    Fd = CuArray(F)
    Fd0 = deepcopy(Fd)

    # scattering operation (for now just bi-linear interpolation)
    t_scatter_cuda = @elapsed scattering!(Fpd, (x, y, z), (dx, dy, dz), Fd, CuArray.(particle_coords))

    # gathering operation (inverse distance weighting)
    particle_coords_dev = CuArray.(particle_coords)
    fill!(Fd, 0.0)
    t_gather_cuda = @elapsed gathering!(Fd, Fpd, (x, y, z), (dx, dy, dz), particle_coords_dev)
       
    # Compute error
    sol_gpu = CuArray(sol)
    misfit_scatter_cuda = @.(log10(abs(Fpd-sol_gpu)))
    misfit_gather_cuda = @.(log10(abs(Fd-Fd0)))

    println("Finished for Ω ∈ [0,1] × [0,1] × [0,1]; $(nx) × $(ny) × $(nz) nodes; $nxcell particles per cell or $(Float64(N)) particles")

    # Plots CPU
    if viz == true
        f = Figure(resolution=(1600, 600))
        a = Axis(f[1,1], aspect=1, title="Analytical")
        scatter!(a, px, py, pz, color=F, colormap=:batlow,markersize=15)
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
    return t_scatter_cpu, t_scatter_cuda, t_gather_cpu, t_gather_cuda,  e1_cpu, e1_cuda
end

function perf_test()
    
    df = DataFrame(
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
        error_cuda=Float64[]
    )

    nx = ny = nz = 128
    for nxcell in (4, 15, 25, 50)
        out = main(nx, ny, nz, nxcell)
        push!(df, [Threads.nthreads() nx ny nz nxcell out...])
    end

    CSV.write("scatter_perf_$(Threads.nthreads())_nxcell_3D.csv", df)
    
end

perf_test()

files = readdir(".")[endswith.(readdir("."), "3D.csv")]

out = [CSV.read(f, DataFrame) for f in files]

## TIMINGS
f = Figure(fontsize=20)

ax1 = Axis(f[1,1], title="scatter", ylabel="seconds", xscale = log10, yscale = log10)
ax2 = Axis(f[2,1], title="gather", xlabel="particles", ylabel="seconds", xscale = log10, yscale = log10)
for data in out
    np = @. data.nx*data.ny*data.nz*data.nxcell
    lines!(ax1, np,  data.t_scatter_cpu, linestyle = :solid, linewidth=3, label = "$(data.threads[1]) threads")
    l=lines!(ax2, np,  data.t_gather_cpu, linestyle = :solid, linewidth=3, label = "$(data.threads[1]) threads")
end
lines!(ax1, @.(out[end].nx*out[end].ny*out[end].ny*out[end].nxcell), out[end].t_scatter_cuda, 
    color = :black, linestyle = :dash, linewidth=3,  label = "RTX3080")
l=lines!(ax2, @.(out[end].nx*out[end].ny*out[end].ny*out[end].nxcell), out[end].t_gather_cuda, 
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
