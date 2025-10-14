import LinearMaps

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
        workspace_size = size(center_X)
        X_mul_U = if isnothing(workspace_sparsity)
            similar(μX, workspace_size)
        else
            ArgumentError("TODO: Support sparse workspace")
            # sparsevec(workspace_sparsity, ones(length(workspace_sparsity)), length(μX))
        end
        Localize_Mul = similar(μX, workspace_size)
        workspace = (; X_mul_U, Localize_Mul)
    end

    return LocalizedEmpiricalCov(Nx, Ne, center_X, μX, Loc, CX, CXloc, workspace)
end

nz_iterator(x::AbstractVector) = zip(eachindex(x), x)
nz_iterator(x::SparseVector) = zip(x.nzind, x.nzval)

function __inner_prod(C::LocalizedEmpiricalCov, u::AbstractVector)
    ret = zero(eltype(u))
    localization_mat = C.Loc.ρX.lmap
    ret = Vector{Float64}(undef, C.Ne)
    tmp = similar(u, Float64)
    for ens_idx in 1:C.Ne
        fill!(tmp, 0.)
        xi = @view C.center_X[:, ens_idx]
        # tmp_i = @view X_mul_U[:, ens_idx]
        for (state_idx, u_val) in nz_iterator(u)
            tmp[state_idx] = xi[state_idx] * u_val
        end
        ret[ens_idx] = dot(tmp, localization_mat, tmp)
    end
    sum(ret) / (C.Ne - 1)
end

function cov_mul!(
    v,
    Ĉ::LocalizedEmpiricalCov,
    u,
    α, β
)
    @inbounds if isnothing(Ĉ.CX)
        if iszero(β)
            fill!(v, zero(eltype(v)))
        elseif !isone(β)
            lmul!(β, v)
        end
        (; X_mul_U, Localize_Mul) = Ĉ.workspace
        # Using https://pi.math.cornell.edu/~ajt/presentations/HadamardProduct.pdf, slide 4
        # (A ⊙ ∑ u_j v_j^T) x = ∑ D_{u_j} A D_{v_j} x
        # = ∑ u_j ⊙ (A (v_j ⊙ x))
        fill!(X_mul_U, zero(eltype(X_mul_U)))
        for ens_idx in axes(X_mul_U, 2)
            for (state_idx, u_val) in nz_iterator(u)
                X_mul_U[state_idx, ens_idx] = Ĉ.center_X[state_idx, ens_idx] * u_val
            end
        end
        mul!(Localize_Mul, Ĉ.Loc.ρX, X_mul_U, α, false)
        for state_idx in eachindex(v)
            for ens_idx in axes(Ĉ.center_X, 2)
                v[state_idx] = muladd(Ĉ.center_X[state_idx, ens_idx], Localize_Mul[state_idx, ens_idx], v[state_idx])
            end
            v[state_idx] /= Ĉ.Ne - 1
        end
    else
        mul!(v, Ĉ.CXloc, u, α, β)
    end
    return v
end

function LinearMaps._unsafe_mul!(v::AbstractMatrix, Ĉ::LocalizedEmpiricalCov, u::AbstractMatrix, alpha=true, beta=false)
    cov_mul!(v, Ĉ, u, alpha, beta)
end

function LinearMaps._unsafe_mul!(v::AbstractVector, Ĉ::LocalizedEmpiricalCov, u::AbstractVector, alpha=true, beta=false)
    cov_mul!(v, Ĉ, u, alpha, beta)
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
