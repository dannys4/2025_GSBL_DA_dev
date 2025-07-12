export ObsSystem

import Base: *, size, Matrix
import LinearAlgebra: mul!

mutable struct ObsSystem
    const Nx::Int64
    const Ny::Int64
    const H::LinearMap
    const Cϵ::LinearMap
    # To update the covariance matrix for the state
    CX::Matrix{Float64}
end

function ObsSystem(H::LinearMap, Cϵ::LinearMap, CX::Matrix{Float64}=Matrix{Float64}(undef, 0, 0))
    Nx = size(CX, 1)
    Ny = size(H, 1)

    return ObsSystem(Nx, Ny, H, Cϵ, CX)
end

size(sys::ObsSystem) = (sys.Ny, sys.Ny)


function Base.Matrix(sys::ObsSystem)
    # Suppose we have [y,]
    # Then we get sys =
    # [ Ce + H * CX * H' ]
    @unpack Nx, Ny, H, Cϵ, CX = sys
    out = Matrix(Cϵ)
    mul!(out, H.lmap, (CX * H.lmap'), true, true)
    return Hermitian(out)
end


function mul!(output::Vector{Float64}, sys::ObsSystem, input::Vector{Float64}, alpha=true, beta=false)

    @unpack Nx, Ny, H, Cϵ, CX = sys
    mul!(output, Cϵ, input, alpha, beta)
    mul!(output, H, CX * (H' * input), alpha, true)

    return output
end

function (*)(sys::ObsSystem, input::Vector{Float64})
    output = similar(input)
    mul!(output, sys, input)
    return output
end
