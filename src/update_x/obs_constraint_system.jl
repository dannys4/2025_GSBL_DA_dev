export ObsConstraintSystem

import Base: *, size
import LinearAlgebra: mul!

struct ObsConstraintSystem
    Nx::Int64
    Ny::Int64
    Nz::Int64
    H::LinearMap
    S::LinearMap
    Cθ::LinearMap
    Cϵ::LinearMap
    # To update the covariance matrix for the state
    CX::Array{LinearMap}
    # cache_xx::ArrayPartition{Float64}
    # cache_ys::ObsConstraintVector
end

function ObsConstraintSystem(
    H::LinearMap,
    S::LinearMap,
    Cθ::LinearMap,
    Cϵ::LinearMap,
    CX::LinearMap,
)
    Nx = size(CX, 1)
    Ny = size(H, 1)
    Nz = size(S, 1)

    # cache_xx = ArrayPartition(zeros(Nx), zeros(Nx))
    # cache_ys = ArrayPartition(zeros(Ny), zeros(Nz))

    return ObsConstraintSystem(Nx, Ny, Nz, H, S, Cθ, Cϵ, [CX])
end

size(sys::ObsConstraintSystem) = (sys.Ny + sys.Nz, sys.Ny + sys.Nz)

function mul!(
    output::ObsConstraintVector,
    sys::ObsConstraintSystem,
    input::ObsConstraintVector,
)

    y = observation(input)
    s = constraint(input)

    @unpack Nx, Ny, Nz, H, S, Cθ, Cϵ, CX = sys
    CX = CX[1]

    output.x[1] .= Cϵ * y
    output.x[1] .+= H * (CX * (H' * y))
    output.x[1] .+= H * (CX * (S' * s))

    output.x[2] .= S * (CX * (H' * y))
    output.x[2] .+= Cθ * s
    output.x[2] .+= S * (CX * (S' * s))

    return output
end

function (*)(sys::ObsConstraintSystem, input::ObsConstraintVector)
    output = similar(input)
    mul!(output, sys, input)
    return output
end

function mul!(output::Vector{Float64}, sys::ObsConstraintSystem, input::Vector{Float64})


    @unpack Nx, Ny, Nz, H, S, Cθ, Cϵ, CX = sys
    CX = CX[1]
    idx_y = 1:Ny
    idx_s = Ny+1:Ny+Nz

    y = view(input, idx_y)
    s = view(input, idx_s)

    output[idx_y] .= Cϵ * y
    output[idx_y] .+= H * (CX * (H' * y))
    output[idx_y] .+= H * (CX * (S' * s))

    output[idx_s] .= S * (CX * (H' * y))
    output[idx_s] .+= Cθ * s
    output[idx_s] .+= S * (CX * (S' * s))

    return output
end

function (*)(sys::ObsConstraintSystem, input::Vector{Float64})
    output = similar(input)
    mul!(output, sys, input)
    return output
end
