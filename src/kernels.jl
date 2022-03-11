## ND LINEAR INTERPOLATION KERNELS

"""
    lerp(t, v0, v1)

Linear interpolation between `f(x0)=v0` and `f(x1)=v1` at the normalized coordinate t
"""
lerp(t, v0, v1) = fma(t, v1, fma(-t, v0, v0))

"""
    bilinear(tx, ty, v00, v10, v01, v11)

Linear interpolation between `f(x0,y0)=v0` and `f(x1,y1)=v1` at the normalized coordinates `tx`, `ty`
"""
function bilinear(tx, ty, v00, v10, v01, v11) 
    return lerp(ty, lerp(tx, v00, v10), lerp(tx, v01, v11))
end

"""
    trilinear(tx, ty, tz, v000, v100, v001, v101, v010, v110, v011, v111) 

Linear interpolation between `f(x0,y0,z0)=v0` and `f(x1,y1,z1)=v1` at the normalized coordinates `tx`, `ty`, `tz`
"""
function trilinear(tx, ty, tz, v000, v100, v001, v101, v010, v110, v011, v111) 
    return lerp(
        ty, 
        bilinear(tx, tz,  v000, v100, v001, v101), 
        bilinear(tx, tz, v010, v110, v011, v111)
    )
end