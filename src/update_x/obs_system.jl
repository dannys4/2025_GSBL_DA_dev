export ObsSystem

using Base: *, size, Matrix
using LinearMaps: _unsafe_mul!, MulStyle, issymmetric, ishermitian
import LinearAlgebra: mul!

mutable struct ObsSystem{M,W} <: LinearMaps.LinearMap{Float64}
    const Nx::Int64
    const Ny::Int64
    const H::LinearMap
    const Cϵ::LinearMap
    # To update the covariance matrix for the state
    CX::M
    # Workspaces for iterative scheme
    workspace::W
end

LinearMaps.MulStyle(::ObsSystem) = FiveArg()
LinearMaps.issymmetric(::ObsSystem) = true
LinearMaps.ishermitian(::ObsSystem) = true

function ObsSystem(H::LinearMap, Cϵ::LinearMap, CX::T=Matrix{Float64}(undef, 0, 0); use_workspace=false, sparse_pattern=nothing) where {T}
    Ny, Nx = size(H)
    if use_workspace
        H_T_X = isnothing(sparse_pattern) ? zeros(Nx) : sparsevec(sparse_pattern, ones(length(sparse_pattern)), Nx)
        workspace = (CX_H_T_X=zeros(Nx), H_T_X=H_T_X)
    else
        workspace = nothing
    end
    return ObsSystem{T,typeof(workspace)}(Nx, Ny, H, Cϵ, CX, workspace)
end

# function modify_CX!(sys::ObsSystem{M}, CX::M) where {M}
#     sys.CX = CX
# end

# function modify_CX!(sys::LinearMap, CX)

# end

function Base.show(io::IO, sys::ObsSystem)
    print(io, "Observation system of size $(sys.Ny)")
end

Base.size(sys::ObsSystem) = (sys.Ny, sys.Ny)


function Base.Matrix(sys::ObsSystem)
    # Suppose we have [y,]
    # Then we get sys =
    # [ Ce + H * CX * H' ]
    @unpack Nx, Ny, H, Cϵ, CX = sys
    out = Matrix(Cϵ)
    mul!(out, H.lmap, (CX * H.lmap'), true, true)
    return Hermitian(out)
end


function LinearMaps._unsafe_mul!(output, sys::ObsSystem, input, alpha, beta)
    @unpack Nx, Ny, H, Cϵ, CX, workspace = sys
    @unpack CX_H_T_X, H_T_X = workspace

    # y = H C H' + Γ
    mul!(output, Cϵ, input, alpha, beta)
    mul!(H_T_X, H', input)
    mul!(CX_H_T_X, CX, H_T_X)
    mul!(output, H, CX_H_T_X, alpha, true)
    return output
end

# mul!(output, sys::ObsSystem, input) = mul!(output, sys, input, true, false)

# function (*)(sys::ObsSystem, input::Vector{Float64})
#     output = similar(input)
#     mul!(output, sys, input)
#     return output
# end

# (sys::ObsSystem)(output, input) = mul!(output, sys, input, true, false)
# (sys::ObsSystem)(output, input, alpha, beta) = mul!(output, sys, input, alpha, beta)