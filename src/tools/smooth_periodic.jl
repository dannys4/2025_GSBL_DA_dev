export SmoothPeriodic, regenerate!

struct SmoothPeriodic
    N::Int64
    Nvar::Int64
    L::Float64
    α::Float64
    ĉ::Vector{ComplexF64}
    is_dirichlet::Bool
end

function SmoothPeriodic(x::Vector{Float64}, α; L=1.0, Nvar::Int64=1, is_dirichlet=false)
    N = length(x)
    ĉ = zeros(ComplexF64, Nvar * N)

    for k = 1:N
        for l = 1:Nvar
            noise = (im * randn()) + (is_dirichlet ? 0. : randn())
            ĉ[(l-1)*N+k] = noise * exp(-0.5 * k^α)
        end
    end

    return SmoothPeriodic(N, Nvar, L, α, ĉ, is_dirichlet)
end

function SmoothPeriodic(N::Int64, α; is_dirichlet=false, L=1.0, Nvar::Int64=1)
    ĉ = zeros(ComplexF64, Nvar * N)

    for k = 1:N
        for l = 1:Nvar
            noise = (im * randn()) + (is_dirichlet ? 0. : randn())
            ĉ[(l-1)*N+k] = noise * exp(-0.5 * k^α)
        end
    end

    return SmoothPeriodic(N, Nvar, L, α, ĉ, is_dirichlet)
end

(f::SmoothPeriodic)(x::Real) =
    sum(k -> real(f.ĉ[k] * exp(im * 2 * π * (k - 1) * x / f.L)), 1:f.N)


function (f::SmoothPeriodic)(out::AbstractVector, xgrid::AbstractVector)
    @assert length(xgrid) == f.N

    @assert length(out) == length(xgrid) * f.Nvar

    for (i, xi) in enumerate(xgrid)
        for k = 1:f.N
            for l = 1:f.Nvar
                out[(l-1)*f.N+i] +=
                    real(f.ĉ[(l-1)*f.N+k] * exp(im * 2 * π * (k - 1) * xi / f.L))
            end
        end
    end
    return out
end

function (f::SmoothPeriodic)(xgrid::AbstractVector)
    out = zeros(length(xgrid) * f.Nvar)
    f(out, xgrid)
    return out
end

function regenerate!(f::SmoothPeriodic)
    for k = 1:f.N
        for l = 1:f.Nvar
            noise = im * randn() +  + (f.is_dirichlet ? 0. : randn())
            f.ĉ[(l-1)*f.N+k] = noise * exp(-0.5 * k^f.α)
        end
    end
end
