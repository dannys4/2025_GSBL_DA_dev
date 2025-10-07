export ObsSystem

using Base: *, size, Matrix
using LinearMaps: _unsafe_mul!, issymmetric, ishermitian
import LinearMaps
import LinearAlgebra: mul!

mutable struct ObsSystem{M,W,HT,EpsT} <: LinearMaps.LinearMap{Float64}
    const Nx::Int64
    const Ny::Int64
    const H::HT
    const Cϵ::EpsT
    # To update the covariance matrix for the state
    CX::M
    # Workspaces for iterative scheme
    workspace::W
end

LinearMaps.MulStyle(::ObsSystem) = FiveArg()
LinearMaps.issymmetric(::ObsSystem) = true
LinearMaps.ishermitian(::ObsSystem) = true

function ObsSystem(H, Cϵ, CX=Matrix{Float64}(undef, 0, 0); use_workspace=false, sparse_pattern=nothing, cg_tol=1e-6)
    Ny, Nx = size(H)
    if use_workspace
        H_T_X = isnothing(sparse_pattern) ? zeros(Nx) : sparsevec(sparse_pattern, ones(length(sparse_pattern)), Nx)
        workspace = (CX_H_T_X=zeros(Nx), H_T_X=H_T_X)
    else
        workspace = nothing
    end
    return ObsSystem(Nx, Ny, H, Cϵ, CX, workspace)
end

function Base.show(io::IO, sys::ObsSystem)
    print(io, "Observation system of size $(sys.Ny)")
end

Base.size(sys::ObsSystem) = (sys.Ny, sys.Ny)


function Base.Matrix(sys::ObsSystem)
    # Suppose we have [y,]
    # Then we get sys =
    # [ Ce + H * CX * H' ]
    @unpack Nx, Ny, H, Cϵ, CX = sys
    HM = Matrix(H)
    out = copy(Matrix(Cϵ))
    mul!(out, HM, (CX * HM'), true, true)
    return Hermitian(Matrix(out))
end

function LinearAlgebra.Diagonal(sys::ObsSystem)
    @unpack Ny, H, Cϵ, CX = sys
    out = zeros(Ny)
    for j in eachindex(out)
        h_j = H[j, :]
        out[j] = Cϵ[j, j] + __inner_prod(CX, h_j)
    end
    Diagonal(out)
end

function obs_sys_mul!(output, sys::ObsSystem, input, alpha, beta)
    @unpack Nx, Ny, H, Cϵ, CX, workspace = sys
    @unpack CX_H_T_X, H_T_X = workspace
    # y = H C H' + Γ
    mul!(output, Cϵ, input, alpha, beta)
    mul!(H_T_X, H', input)
    mul!(CX_H_T_X, CX, H_T_X)
    mul!(output, H, CX_H_T_X, alpha, true)
    return output
end

LinearMaps._unsafe_mul!(out::AbstractVector, sys::ObsSystem, inp::AbstractVector, alpha=true, beta=false) = obs_sys_mul!(out, sys, inp, alpha, beta)
LinearMaps._unsafe_mul!(out::AbstractMatrix, sys::ObsSystem, inp::AbstractMatrix, alpha=true, beta=false) = obs_sys_mul!(out, sys, inp, alpha, beta)