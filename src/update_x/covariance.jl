export EmpiricalCov, LocalizedEmpiricalCov

# In this script, we develop a matrix-free formulation for the action of an empirical covariance matrix on a state

struct EmpiricalCov
    Nx::Int64
    Ne::Int64
    X::Matrix{Float64}
    μX::Vector{Float64}
    CX::Union{Nothing,Matrix{Float64}}
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

function mul!(v::AbstractVector{Float64}, Ĉ::EmpiricalCov, u::AbstractVector{Float64})
    @unpack Nx, Ne, X, μX, CX = Ĉ
    if typeof(CX) == Nothing
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

struct LocalizedEmpiricalCov
    Nx::Int64
    Ne::Int64
    X::Matrix{Float64}
    μX::Vector{Float64}
    CX::Union{Nothing,Matrix{Float64}}
    Loc::Localization
    ρX::Union{SparseMatrixCSC{Float64,Int64},Matrix{Float64}}
    CXloc::Union{Nothing,SparseMatrixCSC{Float64,Int64}}
end

function LocalizedEmpiricalCov(X::Matrix{Float64}, Loc::Localization; with_matrix = true)
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
)

    if typeof(Ĉ.CX) == Nothing
        fill!(v, zero(eltype(v)))

        # Using https://pi.math.cornell.edu/~ajt/presentations/HadamardProduct.pdf
        for i = 1:Ĉ.Ne
            xi = view(Ĉ.X, :, i)
            v .+= Diagonal(xi - Ĉ.μX) * (Ĉ.ρX * ((xi - Ĉ.μX) .* u))
            # @show "we haven't applied localization yet, more a placeholder for now"
        end
        v .*= inv(Ĉ.Ne - 1)
    else
        mul!(v, Ĉ.CXloc, u)
    end
    return v
end

function (*)(Ĉ::LocalizedEmpiricalCov, u::AbstractVector{Float64})
    v = similar(u)
    mul!(v, Ĉ, u)
    return v
end
