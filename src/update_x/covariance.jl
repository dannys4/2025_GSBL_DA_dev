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
    LT<:Localization,CT<:Union{Nothing,<:AbstractMatrix{Float64}}
} <: AbstractEmpiricalCov

    Nx::Int64
    Ne::Int64
    X::Matrix{Float64}
    μX::Vector{Float64}
    Loc::LT
    CX::Union{Nothing,Matrix{Float64}}
    CXloc::CT
end

function LocalizedEmpiricalCov(X::Matrix{Float64}, Loc::Localization; with_matrix=true)
    Nx, Ne = size(X)
    μX = mean(X; dims=2)[:, 1]

    CX = nothing
    CXloc = nothing

    if with_matrix
        CX = cov(X, dims=2)
        CXloc = Loc.ρX .* CX
    end

    return LocalizedEmpiricalCov(Nx, Ne, X, μX, Loc, CX, CXloc)
end

function mul!(
    v::AbstractVector{Float64},
    Ĉ::LocalizedEmpiricalCov,
    u::AbstractVector{Float64},
    α=true,
    β=false
)

    if isnothing(Ĉ.CX)
        @assert α && !β
        fill!(v, zero(eltype(v)))

        # Using https://pi.math.cornell.edu/~ajt/presentations/HadamardProduct.pdf
        for i = 1:Ĉ.Ne
            xi = view(Ĉ.X, :, i)
            v .+= Diagonal(xi - Ĉ.μX) * (Ĉ.Loc.ρX * ((xi - Ĉ.μX) .* u))
            # @show "we haven't applied localization yet, more a placeholder for now"
        end
        v .*= inv(Ĉ.Ne - 1)
    else
        mul!(v, Ĉ.CXloc, u, α, β)
    end
    return v
end

function (*)(Ĉ::LocalizedEmpiricalCov, u::AbstractVector{Float64})
    v = similar(u)
    mul!(v, Ĉ, u)
    return v
end

function Base.Matrix(C::LocalizedEmpiricalCov)
    return C.CXloc
end
