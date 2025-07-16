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
function PolyAnnil(x::Vector, m::Int64; Nvar::Int64=1, istruncated=false)#, dist::Distances.UnionMetric; isperiodic = true)
    n = length(x)

    PA = zeros(Nvar * n, Nvar * n)
    #     Σdist = pairwise(dist, x)
    r = ceil(Int64, m / 2)

    # Discard the nodes near the edges
    xidx = r+1:n-r

    Threads.@threads for i in xidx
        xi = x[i]
        if iseven(m)
            idxi = i-r:i+r
        else
            idxi = i-r:i+r-1
        end
        ngbi = x[idxi]
        mi = length(ngbi)
        @assert mi == m + 1 #There is one more point that the degree m of the PA operator
        # Store the location of the points
        qi = 0.0
        for j in idxi
            xj = x[j]
            # abs(xj - xi) <= 1e-11 && continue
            ωj_xi = prod(xj - x[k] for k in idxi if k != j)
            cj_xi = factorial(m) / ωj_xi
            if xj > xi + 1e-10
                qi += cj_xi
            end
            for k = 1:Nvar
                PA[(k-1)*n+i, (k-1)*n+j] = cj_xi
            end
        end
        for k = 1:Nvar
            PA[(k-1)*n+i, :] ./= qi
        end
    end
    trunc_PA = istruncated ? PA[unroll(xidx, n, Nvar), :] : PA
    return PolyAnnil(x, m, sparse(trunc_PA))
end

mul!(s::AbstractVector, P::PolyAnnil, x::AbstractVector) = mul!(s, P.P, x)
mul!(s::AbstractVector, P::PolyAnnil, x::AbstractVector, alpha, beta) = mul!(s, P.P, x, alpha, beta)

(*)(P::PolyAnnil, x) = P.P * x