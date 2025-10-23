export modfloat
modfloat(a, b) = abs(mod(a + 0.5 * b, b) - 0.5 * b)

include(joinpath(@__DIR__, "PA.jl"))
include(joinpath(@__DIR__, "smooth_periodic.jl"))
include(joinpath(@__DIR__, "crps.jl"))
include(joinpath(@__DIR__, "selection_maps.jl"))
include(joinpath(@__DIR__, "localization.jl"))
include(joinpath(@__DIR__, "viz.jl"))