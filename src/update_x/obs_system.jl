export ObsSystem

import Base: *, size
import LinearAlgebra: mul!

struct ObsSystem
    Nx::Int64
    Ny::Int64
    H::LinearMap
    Cϵ::LinearMap
    # To update the covariance matrix for the state
    CX::Array{LinearMap}
end

function ObsSystem(H::LinearMap, Cϵ::LinearMap, CX::LinearMap)
    Nx = size(CX, 1)
    Ny = size(H, 1)

    return ObsSystem(Nx, Ny, H, Cϵ, [CX])
end

size(sys::ObsSystem) = (sys.Ny, sys.Ny)


function mul!(output::Vector{Float64}, sys::ObsSystem, input::Vector{Float64})

    @unpack Nx, Ny, H, Cϵ, CX = sys
    CX = CX[1]
    output .= Cϵ * input
    output .+= H * (CX * (H' * input))

    return output
end

function (*)(sys::ObsSystem, input::Vector{Float64})
    output = similar(input)
    mul!(output, sys, input)
    return output
end
