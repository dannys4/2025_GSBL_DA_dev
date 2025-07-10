export ObsConstraintSystem

import Base: *, size, Matrix
import LinearAlgebra: mul!

# Covariance of (y,s) | x
mutable struct ObsConstraintSystem{
    HT<:LinearMap,ST<:LinearMap,
    CθT<:LinearMap,CϵT<:LinearMap,
    CacheT<:Union{Nothing,Matrix{Float64}}
}
    const Nx::Int64
    const Ny::Int64
    const Nz::Int64
    const H::HT
    const S::ST
    const Cθ::CθT
    const Cϵ::CϵT
    const cache_YS::CacheT
    # To update the covariance matrix for the state
    CX::Matrix{Float64}
end

function ObsConstraintSystem(
    H::LinearMap,
    S::LinearMap,
    Cθ::LinearMap,
    Cϵ::LinearMap;
    cache=true,
)
    Ny, Nx = size(H)
    Nz = size(S, 1)
    cache_YS = cache ? zeros(Ny + Nz, Ny + Nz) : nothing

    # cache_xx = ArrayPartition(zeros(Nx), zeros(Nx))
    # cache_ys = ArrayPartition(zeros(Ny), zeros(Nz))
    CX = Matrix{Float64}(undef, 0, 0)
    return ObsConstraintSystem(Nx, Ny, Nz, H, S, Cθ, Cϵ, cache_YS, CX)
end

size(sys::ObsConstraintSystem) = (sys.Ny + sys.Nz, sys.Ny + sys.Nz)

function mul!(
    output::ObsConstraintVector,
    sys::ObsConstraintSystem,
    input::ObsConstraintVector,
)

    y = observation(input)
    s = constraint(input)

    @unpack H, S, Cθ, Cϵ, CX = sys

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

function _muladd!(Out, A, B)
    mul!(Out, A, B, true, true)
end

function Base.Matrix(sys::ObsConstraintSystem)
    # Suppose we have [y, s]
    # Then we get sys =
    # [ Ce + H * CX * H' |   H * CX * S'    ]
    # [   S * CX * H'    | CT + S * CX * S' ]
    @unpack Nx, Ny, Nz, H, S, Cθ, Cϵ, CX, cache_YS = sys
    H_CX = H.lmap * CX
    C_YY = @view cache_YS[1:Ny, 1:Ny]
    C_YS = @view cache_YS[1:Ny, Ny+1:end]
    C_SS = @view cache_YS[Ny+1:end, Ny+1:end]
    # Off-diagonal
    mul!(C_YS, H_CX, S.lmap')
    # Diagonal
    copy!(C_YY, Cϵ.lmap)
    mul!(C_YY, H_CX, H.lmap', true, true)
    copy!(C_SS, Cθ.lmap)
    mul!(C_SS, S.lmap, Matrix(CX * S'))
    return Hermitian(cache_YS, :U)
end

function mul!(output::AbstractVector{Float64}, sys::ObsConstraintSystem, input::AbstractVector{Float64})


    @unpack Nx, Ny, Nz, H, S, Cθ, Cϵ, CX = sys
    CX = CX[1]
    idx_y = Base.OneTo(Ny)
    idx_s = Ny+1:Ny+Nz

    y = @view input[idx_y]
    s = @view input[idx_s]

    out_y = @view output[idx_y]
    out_s = @view output[idx_s]

    tmp_X1 = Vector{Float64}(undef, size(CX, 1))
    tmp_X2 = similar(tmp_X1)
    mul!(out_y, Cϵ, y)
    mul!(out_s, Cθ, s)

    mul!(tmp_X1, H', y)
    mul!(tmp_X2, CX, tmp_X1)
    _muladd!(out_y, H, tmp_X2)
    _muladd!(out_s, S, tmp_X2)

    mul!(tmp_X1, S', s)
    mul!(tmp_X2, CX, tmp_X1)
    _muladd!(out_y, H, tmp_X2)
    _muladd!(out_s, S, tmp_X2)

    return output
end

function (*)(sys::ObsConstraintSystem, input::Vector{Float64})
    output = similar(input)
    mul!(output, sys, input)
    return output
end
