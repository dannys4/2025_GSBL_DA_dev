export ObsConstraintSystem

import Base: *, size, Matrix
import LinearAlgebra: mul!
using LinearMaps: _unsafe_mul!

struct ObsConstraintSysCache{MT}
    C_YY::MT
    C_YS::MT
    C_SS::MT
    function ObsConstraintSysCache(Ny, Nz)
        cache_sys = zeros(Ny + Nz, Ny + Nz)
        C_YY = @view cache_sys[1:Ny, 1:Ny]
        C_YS = @view cache_sys[1:Ny, Ny+1:end]
        C_SS = @view cache_sys[Ny+1:end, Ny+1:end]
        new{typeof(C_YY)}(C_YY, C_YS, C_SS), cache_sys
    end
end

# Covariance of (y,s) | x
mutable struct ObsConstraintSystem{
    CXT,
    HT<:LinearMap,ST<:LinearMap,
    CθT<:LinearMap,CϵT<:LinearMap,
    CacheT<:Union{Nothing,ObsConstraintSysCache{<:AbstractMatrix}},
    SysCacheT<:Union{Nothing,Matrix{Float64}},
    MatVecCacheT<:Union{Nothing,NTuple{2,Vector{Float64}}}
} <: LinearMaps.LinearMap{Float64}
    # To update the covariance matrix for the state
    CX::CXT
    const Nx::Int64
    const Ny::Int64
    const Nz::Int64
    const H::HT
    const S::ST
    const Cθ::CθT
    const Cϵ::CϵT
    const cache_YS::CacheT
    const cache_sys::SysCacheT
    const cache_matvec::MatVecCacheT
end

function ObsConstraintSystem(
    H::LinearMap,
    S::LinearMap,
    Cθ::LinearMap,
    Cϵ::LinearMap,
    CX::T=Matrix{Float64}(undef, 0, 0);
    cache_matrix=true,
    isiterative=false,
) where {T}
    Ny, Nx = size(H)
    Nz = size(S, 1)
    cache_YS, cache_sys = cache_matrix ? ObsConstraintSysCache(Ny, Nz) : (nothing, nothing)
    cache_matvec = isiterative ? (Vector{Float64}(undef, Nx), Vector{Float64}(undef, Nx)) : nothing
    return ObsConstraintSystem(CX, Nx, Ny, Nz, H, S, Cθ, Cϵ, cache_YS, cache_sys, cache_matvec)
end

Base.size(sys::ObsConstraintSystem) = (sys.Ny + sys.Nz, sys.Ny + sys.Nz)

function _muladd!(Out, A, B)
    mul!(Out, A, B, true, true)
end

function mul!(
    output::ObsConstraintVector,
    sys::ObsConstraintSystem,
    input::ObsConstraintVector,
)

    y = observation(input)
    s = constraint(input)

    @unpack H, S, Cθ, Cϵ, CX = sys

    fill!(output, zero(eltype(output)))

    out_y = output.x[1]
    out_s = output.x[1]
    tmp_X1 = Vector{Float64}(undef, size(CX, 1))
    tmp_X2 = similar(tmp_X1)

    mul!(out_y, Cϵ, y)
    mul!(out_s, Cθ, s)

    mul!(tmp_X1, H', y)
    mul!(tmp_X2, CX, tmp_1)
    _muladd!(out_y, H, tmp_X2)
    _muladd!(out_s, S, tmp_X2)

    mul!(tmp_X1, S', s)
    mul!(tmp_X2, CX, tmp_X1)
    _muladd!(out_y, H, tmp_X2)
    _muladd!(out_s, S, tmp_X2)

    # output.x[1] .= Cϵ * y
    # output.x[1] .+= H * (CX * (H' * y))
    # output.x[1] .+= H * (CX * (S' * s))

    # output.x[2] .= S * (CX * (H' * y))
    # output.x[2] .+= Cθ * s
    # output.x[2] .+= S * (CX * (S' * s))

    return output
end

function (*)(sys::ObsConstraintSystem, input::ObsConstraintVector)
    output = similar(input)
    mul!(output, sys, input)
    return output
end

function initialize_sys_diag_block!(C::AbstractMatrix, Cϵ::LinearMaps.UniformScalingMap)
    # C .= zero(eltype(C))
    C[diagind(C)] .= Cϵ.λ
end

function initialize_sys_diag_block!(C_YY::AbstractMatrix, Cϵ::LinearMaps.WrappedMap)
    copy!(C_YY, Cϵ.lmap)
end

function Base.Matrix(sys::ObsConstraintSystem)
    # Suppose we have [y, s]
    # Then we get sys =
    # [ Ce + H * CX * H' |   H * CX * S'    ]
    # [   S * CX * H'    | CT + S * CX * S' ]
    @unpack Nx, Ny, Nz, H, S, Cθ, Cϵ, CX, cache_sys, cache_YS = sys
    H_CX = H * CX
    (; C_YY, C_YS, C_SS) = cache_YS
    fill!(cache_sys, zero(eltype(cache_sys)))
    # Off-diagonal
    mul!(C_YS, H_CX, S.lmap')
    # Diagonal
    initialize_sys_diag_block!(C_YY, Cϵ)
    mul!(C_YY, H_CX, Matrix(H'), true, true)

    initialize_sys_diag_block!(C_SS, Cθ)
    mul!(C_SS, S.lmap, CX * S', true, true)
    # cache_sys shares memory with C_YY,C_YS,C_SS
    return Hermitian(cache_sys, :U)
end

function __inner_prod(C, x)
    x' * C * x
end

function LinearAlgebra.Diagonal(sys::ObsConstraintSystem)
    @unpack Nx, Ny, Nz, H, S, Cθ, Cϵ, CX = sys
    out = zeros(Ny + Nz)
    out_y, out_s = @view(out[1:Ny]), @view(out[Ny+1:end])
    for j in eachindex(out_y)
        h_j = H[j, :]
        out_y[j] = Cϵ[j, j] + __inner_prod(CX, h_j)
    end
    for j in eachindex(out_s)
        s_j = S.lmap[j, :]
        out_s[j] = Cθ[j, j] + __inner_prod(CX, s_j)
    end
    Diagonal(out)
end

function LinearMaps._unsafe_mul!(
    output,
    sys::ObsConstraintSystem,
    input,
)
    @unpack Nx, Ny, Nz, H, S, Cθ, Cϵ, CX, cache_matvec = sys
    tmpx1, tmpx2 = cache_matvec
    idx_y = Base.OneTo(Ny)
    idx_s = Ny+1:Ny+Nz

    y = @view input[idx_y]
    s = @view input[idx_s]

    out_y = @view output[idx_y]
    out_s = @view output[idx_s]

    mul!(tmpx1, H', y)

    mul!(tmpx1, S', s, true, true)

    mul!(tmpx2, CX, tmpx1)

    mul!(out_y, Cϵ, y)
    mul!(out_s, Cθ, s)
    mul!(out_y, H, tmpx2, true, true)
    mul!(out_s, S, tmpx2, true, true)
    return output
end

# function (*)(sys::ObsConstraintSystem, input::Vector{Float64})
#     output = similar(input)
#     mul!(output, sys, input)
#     return output
# end
