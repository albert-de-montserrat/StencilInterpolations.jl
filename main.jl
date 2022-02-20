using StaticArrays

include("src/utils/Utils.jl")

nx = ny = 11
lx = ly = 1
dx, dy = lx/(nx-1), ly/(ny-1)
x = LinRange(0, lx, nx)
y = LinRange(0, ly, ny)

# random field
T = [rand() for _ in x, _ in y]

N = 100000
px, py = rand(N), rand(N)
particle_coords = (px, py)

p = (px, py)
dxi = (dx, dy)


gather(xi, dxi, F, particle_coords)
gather(xi, dxi, F, particle_coords)