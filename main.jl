using Statistics
using StencilInterpolation

# random particles
N = Int(1e6)
px, py = rand(N), rand(N)
particle_coords = (px, py)
p = (px, py)

# model domain
nx = ny = 101
lx = ly = 1
dx, dy = lx/(nx-1), ly/(ny-1)
x = LinRange(0, lx, nx)
y = LinRange(0, ly, ny)

# random field
F = [-sin(2*yi)*cos(3*π*xi) for xi in x, yi in y]

# Scattering operation (for now just bi-linear interpolation)
@time Fp = scattering( (x, y), (dx, dy), F, particle_coords);

# # Compute error
# Fanalytic = [-sin(2*yi)*cos(3*π*xi) for (xi, yi) in zip(px,py)]
# misfit = @.(log10(abs(Fp-Fanalytic)))
# println("average log10 error = $(mean(misfit))")

# # Plots
# f = Figure(resolution=(1600, 600))
# a = Axis(f[1,1], aspect=1, title="Analytical")
# heatmap!(a, x, y, F, colormap=:batlow)
# xlims!(0, lx)
# ylims!(0, ly)

# a = Axis(f[1, 2], aspect=1, title="Scattered")
# scatter!(a, px, py, color=Fp, colormap=:batlow)
# xlims!(0, lx)
# ylims!(0, ly)

# a = Axis(f[1, 3], aspect=1, title="log10 error")
# s = scatter!(a, px, py, color=misfit, colormap=:batlow)
# xlims!(0, lx)
# ylims!(0, ly)
# Colorbar(f[1,4], s)
# display(f)

# CUDA
Fpd = CUDA.zeros(Float64, N)
Fd = CuArray(F)

@time scattering!(Fpd, (x, y), (dx, dy), Fd, CuArray.(particle_coords));