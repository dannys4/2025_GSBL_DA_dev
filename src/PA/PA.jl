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
function PolyAnnil(x::Vector, m::Int64; Nvar::Int64 = 1, istruncated = false)#, dist::Distances.UnionMetric; isperiodic = true)
    n = length(x)

    PA = zeros(Nvar * n, Nvar * n)
    #     Σdist = pairwise(dist, x)
    r = ceil(Int64, m / 2)

    # Discard the nodes near the edges
    xidx = r+1:n-r

    for i in xidx
        xi = x[i]
        #         idxi = partialsortperm(view(Σdist,i,:), 1:m)
        if mod(m, 2) == 0
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
            ωj_xi = prod(xj - x[k] for k in idxi if k != j)
            cj_xi = factorial(m) / ωj_xi
            if xj >= xi
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

    if istruncated
        return PolyAnnil(x, m, sparse(PA[unroll(xidx, n, Nvar), :]))
    else
        return PolyAnnil(x, m, sparse(PA))
    end
end

mul!(s::AbstractVector, P::PolyAnnil, x::AbstractVector) = mul!(s, P.P, x)

(*)(P::PolyAnnil, x) = P.P * x

#function barycentric_weights(nodes)
# n_nodes = length(nodes)
# weights = ones(n_nodes)

# for j in 2:n_nodes, k in 1:(j - 1)
#     weights[k] *= nodes[k] - nodes[j]
#     weights[j] *= nodes[j] - nodes[k]
# end

# for j in 1:n_nodes
#     weights[j] = 1 / weights[j]
# end

# return weights
# end

# Routine from Trixi.jl to compute weights.
# function barycentric_weights(nodes)
#     n_nodes = length(nodes)
#     weights = ones(n_nodes)

#     for j in 2:n_nodes, k in 1:(j - 1)
#         weights[k] *= nodes[k] - nodes[j]
#         weights[j] *= nodes[j] - nodes[k]
#     end

#     for j in 1:n_nodes
#         weights[j] = 1 / weights[j]
#     end

#     return weights
# end
