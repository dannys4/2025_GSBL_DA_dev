export EmpiricalCov, LocalizedEmpiricalCov

# In this script, we develop a matrix-free formulation for the action of an empirical covariance matrix on a state

struct EmpiricalCov{M<:AbstractMatrix{Float64},V<:AbstractVector{Float64}, C<:Union{Nothing,M}}
    Nx::Int64
    Ne::Int64
    X::M
    μX::V
    CX::C
end

function EmpiricalCov(X::Matrix{Float64}; with_matrix = true)
    Nx, Ne = size(X)
    μX = mean(X; dims = 2)[:, 1]
    if with_matrix == true
        CX = cov(X')
    else
        CX = nothing
    end
    return EmpiricalCov(Nx, Ne, X, μX, CX)
end

function mul!(
    v::AbstractVector{Float64},
    Ĉ::EmpiricalCov,
    u::AbstractVector{Float64},
    α::Real,
    β::Real,
)
    @unpack Nx, Ne, X, μX, CX = Ĉ
    if typeof(CX) == Nothing
        fill!(v, zero(eltype(v)))
        for i = 1:Ne
            xi = view(X, :, i)
            v .+= (xi - μX) * dot(xi - μX, u)
        end
        v .*= α * inv(Ne - 1)
    else
        mul!(v, CX, u, α, β)
    end
    return v
end

function (*)(Ĉ::EmpiricalCov, u::AbstractVector{Float64})
    v = similar(u)
    mul!(v, Ĉ, u, true, false)
    return v
end

struct LocalizedEmpiricalCov{M<:AbstractMatrix{Float64},V<:AbstractVector{Float64},C<:Union{Nothing,<:AbstractMatrix{Float64}}}
    Nx::Int64
    Ne::Int64
    X::M
    μX::V
    CX::C
    Loc::Localization
    ρX::Union{SparseMatrixCSC{Float64,Int64},Matrix{Float64}}
    CXloc::Union{Nothing,SparseMatrixCSC{Float64,Int64}}
end

function LocalizedEmpiricalCov(X::AbstractMatrix{Float64}, Loc::Localization; with_matrix = true)
    Nx, Ne = size(X)
    μX = mean(X; dims = 2)[:, 1]

    ρX = sparse(Locgaspari((Nx, Nx), Loc.L, Loc.Gxx))

    if with_matrix == true
        CX = cov(X')
        CXloc = ρX .* CX
    else
        CX = nothing
        CXloc = nothing
    end

    return LocalizedEmpiricalCov(Nx, Ne, X, μX, CX, Loc, ρX, CXloc)
end

function mul!(
    v::AbstractVector{Float64},
    Ĉ::LocalizedEmpiricalCov,
    u::AbstractVector{Float64},
    α::Real,
    β::Real,
)

    if typeof(Ĉ.CX) == Nothing
        fill!(v, zero(eltype(v)))

        # Using https://pi.math.cornell.edu/~ajt/presentations/HadamardProduct.pdf
        for i = 1:Ĉ.Ne
            xi = view(Ĉ.X, :, i)
            v .+= Diagonal(xi - Ĉ.μX) * (Ĉ.ρX * ((xi - Ĉ.μX) .* u))
            # @show "we haven't applied localization yet, more a placeholder for now"
        end
        v .*= α / (Ĉ.Ne - 1)
    else
        mul!(v, Ĉ.CXloc, u, α, β)
    end
    return v
end

function (*)(Ĉ::LocalizedEmpiricalCov, u::AbstractVector{Float64})
    v = similar(u)
    mul!(v, Ĉ, u, true, false)
    return v
end
