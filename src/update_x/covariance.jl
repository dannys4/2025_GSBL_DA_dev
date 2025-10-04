export EmpiricalCov, LocalizedEmpiricalCov
using LinearMaps: _unsafe_mul!, MulStyle, issymmetric, ishermitian
using Base: size
abstract type AbstractEmpiricalCov <: LinearMaps.LinearMap{Float64} end

LinearMaps.issymmetric(::AbstractEmpiricalCov) = true
LinearMaps.ishermitian(::AbstractEmpiricalCov) = true
LinearMaps.MulStyle(::AbstractEmpiricalCov) = LinearMaps.FiveArg()
Base.size(C::AbstractEmpiricalCov) = (C.Nx, C.Nx)

# In this script, we develop a matrix-free formulation for the action of an empirical covariance matrix on a state

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

struct LocalizedEmpiricalCov{
    LT,CT<:Union{Nothing,<:AbstractMatrix{Float64}},W
} <: AbstractEmpiricalCov

    Nx::Int64
    Ne::Int64
    center_X::Matrix{Float64}
    μX::Vector{Float64}
    Loc::LT
    CX::Union{Nothing,Matrix{Float64}}
    CXloc::CT
    workspace::W
end

function localization_elementwise_mul(A::SparseMatrixCSC, B::AbstractMatrix)
    C = similar(A)
    nonzero_idxs = findall(!iszero, A)
    @inbounds begin
        Trixi.@threaded for sp_idx in eachindex(nonzero_idxs)
            c_idx = nonzero_idxs[sp_idx]
            C.nzval[sp_idx] = A.nzval[sp_idx] * B[c_idx]
        end
    end
    C
end

function localization_elementwise_mul(A::LinearMaps.WrappedMap, B::AbstractMatrix)
    localization_elementwise_mul(A.lmap, B)
end

function localization_elementwise_mul(A::AbstractMatrix, B::AbstractMatrix)
    A .* B
end

function LocalizedEmpiricalCov(X::Matrix{Float64}, Loc::Localization; with_matrix=true, workspace_sparsity=nothing)
    Nx, Ne = size(X)
    μX = vec(mean(X; dims=2))
    center_X = X .- μX

    CX = nothing
    CXloc = nothing
    workspace = nothing

    if with_matrix
        CX = (center_X * center_X') / (Ne - 1)
        CXloc = localization_elementwise_mul(Loc.ρX, CX)
    else
        X_mul_U = isnothing(workspace_sparsity) ? similar(μX) : sparsevec(workspace_sparsity, ones(length(workspace_sparsity)), length(μX))
        Localize_Mul = similar(μX)
        workspace = (; X_mul_U, Localize_Mul)
    end

    return LocalizedEmpiricalCov(Nx, Ne, center_X, μX, Loc, CX, CXloc, workspace)
end

function __inner_prod(C::LocalizedEmpiricalCov, u::AbstractVector)
    ret = zero(eltype(u))
    T = promote_type(eltype(u), eltype(C.center_X))
    X_mul_U = similar(u, T)
    herm_loc = C.Loc.ρX.lmap
    for i in 1:C.Ne
        xi = @view C.center_X[:, i]
        if u isa SparseVector
            for (u_idx, x_idx) in enumerate(u.nzind)
                # @info "" X_mul_U.nzval[u_idx] xi[x_idx] u.nzval[u_idx]
                X_mul_U.nzval[u_idx] = xi[x_idx] * u.nzval[u_idx]
            end
        else
            for idx in eachindex(X_mul_U, xi, u)
                X_mul_U[idx] = xi[idx] * u[idx]
            end
        end
        ret += dot(X_mul_U, herm_loc, X_mul_U)
    end
    ret / (C.Ne - 1)
end

function cov_mul!(
    v,
    Ĉ::LocalizedEmpiricalCov,
    u,
    α, β
)
    if isnothing(Ĉ.CX)
        # @assert α && !β
        if β isa Bool
            # If beta = false, fill with zeros
            # Otherwise, v should stay as-is
            β || fill!(v, zero(eltype(v)))
        else
            # If beta is not a bool, just straight-up multiply
            v .*= β
        end
        (; X_mul_U, Localize_Mul) = Ĉ.workspace
        # Using https://pi.math.cornell.edu/~ajt/presentations/HadamardProduct.pdf, slide 4
        # (A ⊙ ∑ u_j v_j^T) x = ∑ D_{u_j} A D_{v_j} x
        # = ∑ u_j ⊙ (A (v_j ⊙ x))
        X_mul_U = similar(u, promote_type(eltype(u), eltype(Ĉ.center_X)))
        @inbounds for i = 1:Ĉ.Ne
            xi = @view Ĉ.center_X[:, i]
            # Recall that xi is centered in constructor.
            # v .+= Diagonal(xi) * (Ĉ.Loc.ρX * (xi .* u))
            if u isa SparseVector
                for (u_idx, x_idx) in enumerate(u.nzind)
                    # @info "" X_mul_U.nzval[u_idx] xi[x_idx] u.nzval[u_idx]
                    X_mul_U.nzval[u_idx] = xi[x_idx] * u.nzval[u_idx]
                end
            else
                X_mul_U .= xi .* u
            end
            mul!(Localize_Mul, Ĉ.Loc.ρX, X_mul_U, α, false)
            for state_idx in eachindex(v)
                v[state_idx] = muladd(xi[state_idx], Localize_Mul[state_idx], v[state_idx])
            end
        end
        v .*= inv(Ĉ.Ne - 1)
    else
        mul!(v, Ĉ.CXloc, u, α, β)
    end
    return v
end

function LinearMaps._unsafe_mul!(v::AbstractMatrix, Ĉ::LocalizedEmpiricalCov, u::AbstractMatrix, alpha, beta)
    cov_mul!(v, Ĉ, u, alpha, beta)
end

function LinearMaps._unsafe_mul!(v::AbstractMatrix, Ĉ::LocalizedEmpiricalCov, u::AbstractMatrix)
    cov_mul!(v, Ĉ, u, true, false)
end

function LinearMaps._unsafe_mul!(v::AbstractVector, Ĉ::LocalizedEmpiricalCov, u::AbstractVector, alpha, beta)
    cov_mul!(v, Ĉ, u, alpha, beta)
end

function LinearMaps._unsafe_mul!(v::AbstractVector, Ĉ::LocalizedEmpiricalCov, u::AbstractVector)
    cov_mul!(v, Ĉ, u, true, false)
end

function (*)(Ĉ::LocalizedEmpiricalCov, u::AbstractVector{Float64})
    v = similar(u)
    mul!(v, Ĉ, u)
    return v
end

function (*)(Ĉ::LocalizedEmpiricalCov, u::SparseVector{Float64})
    v = Vector{Float64}(undef, length(u))
    mul!(v, Ĉ, u)
    return v
end

function Base.Matrix(C::LocalizedEmpiricalCov)
    isnothing(C.CXloc) ? C * I(size(C, 1)) : C.CXloc
end
