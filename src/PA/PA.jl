export PolyAnnil

import Base: *
import LinearAlgebra: mul!

struct PolyAnnil
    x::Vector{Float64}
    m::Int64
    P::SparseMatrixCSC{Float64,Int64}
end

# Build the operator
# x: nodes
# m: order of the PA operator
function PolyAnnil_single(x::Vector, m::Int64; istruncated=false, isperiodic=false, periodic_limits=nothing)
    r = ceil(Int64, m / 2)
    if isperiodic
        isa(periodic_limits, NTuple{2,<:Real}) || throw(ArgumentError("Expected periodic limits (x_min, x_max), got $periodic_limits"))
        istruncated || throw(ArgumentError("If isperiodic=true, must have istruncated=true"))
        x_min, x_max = periodic_limits
        x = vcat((x_min - x_max) .+ x[end-r+1:end], x, (x_max - x_min) .+ x[1:r])
    end

    n = length(x)

    PA = zeros(n, n)
    # Discard the nodes near the edges
    xidx = r+1:n-r

    for i in xidx
        xi = x[i]
        idxi = i-r:(i+r-isodd(m))
        ngbi = x[idxi]
        mi = length(ngbi)
        @assert mi == m + 1 # There is one more point that the degree m of the PA operator
        # Store the location of the points
        qi = 0.0
        for j in idxi
            xj = x[j]
            # cj as formulated in https://doi.org/10.1137/S0036142903435259 Thm 3.2
            ωj_xi = prod(xj - x[k] for k in idxi if k != j)
            cj_xi = factorial(m) / ωj_xi
            # Only add to qi if grid points are to right of xi
            if xj >= xi
                qi += cj_xi
            end
            PA[i, j] = cj_xi
        end
        PA[i, :] ./= qi
    end
    PA = istruncated ? PA[xidx, :] : PA # TODO: Fix this in case of untruncated periodic?
    if isperiodic
        dst = @view PA[1:r, end-(m-iseven(m)):end-r]
        copy!(dst, @view(PA[1:r, 1:r]))
        dst = @view PA[end-r+1:end, r+1:2r]
        copy!(dst, @view(PA[end-r+1:end, end-r+1:end]))
        PA = PA[:, xidx]
    end
    sparse(PA)
end

function PolyAnnil(x::Vector, m::Int; Nvar::Int=1, kwargs...)
    P = PolyAnnil_single(x, m; kwargs...)
    PolyAnnil(x, m, Nvar == 1 ? P : kron(P, I(Nvar)))
end

mul!(s::AbstractVector, P::PolyAnnil, x::AbstractVector) = mul!(s, P.P, x)
mul!(s::AbstractVector, P::PolyAnnil, x::AbstractVector, alpha, beta) = mul!(s, P.P, x, alpha, beta)

(*)(P::PolyAnnil, x) = P.P * x