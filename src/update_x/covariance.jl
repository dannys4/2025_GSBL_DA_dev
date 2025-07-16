export EmpiricalCov, LocalizedEmpiricalCov
abstract type AbstractEmpiricalCov end

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

function mul!(v::AbstractVector{Float64}, Ĉ::EmpiricalCov, u::AbstractVector{Float64})
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

function (*)(Ĉ::EmpiricalCov, u::AbstractVector{Float64})
    v = similar(u)
    mul!(v, Ĉ, u)
    return v
end

struct LocalizedEmpiricalCov{
    LT<:Localization,CT<:Union{Nothing,<:AbstractMatrix{Float64}},W
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

function LocalizedEmpiricalCov(X::Matrix{Float64}, Loc::Localization; with_matrix=true)
    Nx, Ne = size(X)
    μX = vec(mean(X; dims=2))
    center_X = copy(X)
    @. center_X = center_X - μX

    CX = nothing
    CXloc = nothing
    workspace = nothing

    if with_matrix
        CX = center_X * center_X'
        CXloc = Loc.ρX .* CX
    else
        workspace = (similar(μX), similar(μX))
    end

    return LocalizedEmpiricalCov(Nx, Ne, center_X, μX, Loc, CX, CXloc, workspace)
end

function mul!(
    v::AbstractVector{Float64},
    Ĉ::LocalizedEmpiricalCov,
    u::AbstractVector{Float64},
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
        tmp = Ĉ.workspace[1]
        tmp_loc = Ĉ.workspace[2]
        # Using https://pi.math.cornell.edu/~ajt/presentations/HadamardProduct.pdf, slide 4
        # (A ⊙ ∑ u_j v_j^T) x = ∑ D_{u_j} A D_{v_j} x
        # = ∑ u_j ⊙ (A (v_j ⊙ x))
        for i = 1:Ĉ.Ne
            xi = @view Ĉ.center_X[:, i]
            # Recall that xi is centered in constructor.
            # v .+= Diagonal(xi) * (Ĉ.Loc.ρX * (xi .* u))
            @. tmp = xi * u
            mul!(tmp_loc, Ĉ.Loc.ρX, tmp, α, false)
            for state_idx in eachindex(v)
                v[state_idx] = muladd(xi[state_idx], tmp_loc[state_idx], v[state_idx])
            end
            # @show "we haven't applied localization yet, more a placeholder for now"
        end
        v .*= inv(Ĉ.Ne - 1)
    else
        mul!(v, Ĉ.CXloc, u, α, β)
    end
    return v
end

mul!(v, Ĉ::LocalizedEmpiricalCov, u) = mul!(v, Ĉ, u, true, false)

function (*)(Ĉ::LocalizedEmpiricalCov, u::AbstractVector{Float64})
    v = similar(u)
    mul!(v, Ĉ, u)
    return v
end

function Base.Matrix(C::LocalizedEmpiricalCov)
    return C.CXloc
end
