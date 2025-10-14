export EmpiricalCov
using LinearMaps: _unsafe_mul!, issymmetric, ishermitian
import LinearMaps
using Base: size
import Base: *

struct EmpiricalCov <: AbstractEmpiricalCov
    Nx::Int64
    Ne::Int64
    X::Matrix{Float64}
    μX::Vector{Float64}
    CX::Union{Nothing,Matrix{Float64}}
end

function EmpiricalCov(X::Matrix{Float64}; with_matrix=true)
    Nx, Ne = size(X)
    μX = mean(X; dims=2)[:, 1]

    CX = nothing

    if with_matrix
        CX = cov(X')
    end
    return EmpiricalCov(Nx, Ne, X, μX, CX)
end

function Base.Matrix(C::EmpiricalCov)
    return C.CX
end

function LinearMaps._unsafe_mul!(v::AbstractVector{Float64}, Ĉ::EmpiricalCov, u::AbstractVector{Float64})
    @unpack Nx, Ne, X, μX, CX = Ĉ
    if isnothing(CX)
        fill!(v, zero(eltype(v)))
        for i = 1:Ne
            xi = view(X, :, i)
            v .+= (xi - μX) * dot(xi - μX, u)
        end
        v .*= inv(Ne - 1)
    else
        mul!(v, CX, u)
    end
    return v
end
