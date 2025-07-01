export CRPS_gaussian, CRPS_quadrature, CRPS, weight_sum_reduction

"""
CRPS calculation for a univariate Gaussian distribution
"""
function CRPS_gaussian(p::Normal{Float64}, x)
    μ = mean(p)
    σ = std(p)
    z = (x - μ) / σ
    return σ * (z * (2 * cdf(Normal(), z) - 1) + 2 * pdf(Normal(), z) - 1 / sqrt(π))
end

"""
CRPS calculation based on quadrature rule
"""
function CRPS_quadrature(X::AbstractVector, x::Float64)
    ΦX = ecdf(X)
    Φx = ecdf([x])
    return quadgk(t -> (ΦX(t) - Φx(t))^2, -Inf, Inf)[1]
end

"""
Fastest implementation of CRPS
"""
function CRPS(X::AbstractVector, x::Float64)
    N = length(X)

    term1 = 0.0
    term2 = 0.0
    @inbounds for i = 1:N
        xi = X[i]
        term1 += abs(xi - x)
        # Exploit symmetry and multiply off-diagonal entries by 2.
        for j = i+1:N
            xj = X[j]
            term2 += 2 * abs(xj - xi)
        end
    end

    term1 *= 1 / N
    term2 *= 1 / N^2

    return term1 - 0.5 * term2
end


function CRPS(X::AbstractMatrix, x::AbstractVector, which_norm::Symbol=:norm2, space_weights::AbstractVector=fill(1 / length(x), length(x)))
    d, N = size(X)
    @assert length(space_weights) == d
    abs_fcn, rt_fcn = nothing, nothing
    if which_norm == :norm2
        abs_fcn, rt_fcn = abs2, sqrt
    elseif which_norm == :norm1
        abs_fcn, rt_fcn = abs, identity
    else
        throw(ArgumentError("Which norm: only :norm2 and :norm1 are supported"))
    end
    rmse = 0.
    std_diff = 0.
    for ens_idx in 1:N
        sq_diff = zero(eltype(x))
        for dim_idx in 1:d
            sq_diff += space_weights[dim_idx] * abs_fcn(X[dim_idx, ens_idx] - x[dim_idx])
        end
        rmse += rt_fcn(sq_diff)
        for ens_idx_prime in ens_idx+1:N
            var_diff = zero(eltype(x))
            for dim_idx in 1:d
                var_diff += 2space_weights[dim_idx] * abs_fcn(X[dim_idx, ens_idx] - X[dim_idx, ens_idx_prime])
            end
            std_diff += rt_fcn(var_diff)
        end
    end
    rmse /= N
    std_diff /= N * (N - 1)
    crps = rmse - std_diff / 2
    (; crps, rmse, std_diff)
end

# Return CRPS of a trajectory at each time
function CRPS(X::AbstractVector{<:AbstractMatrix}, truth::AbstractMatrix, space_weights::AbstractVector=ones(1 / size(truth, 1), size(truth, 1)))
    T = size(truth, 2)
    @assert length(X) == T + 1
    map(enumerate(X[2:end])) do (time_idx, Xi)
        sum(axes(truth, 1)) do space_idx
            space_weights[space_idx] * CRPS(@view(Xi[space_idx, :]), truth[space_idx, time_idx])
        end
    end
end

function weight_sum_reduction(x::AbstractVector, fcn::Function, weights::AbstractVector=fill(1 / length(x), length(x)))
    d = length(x)
    @assert length(weights) == d
    return sum(zip(x, weights)) do (x_elem, wt_elem)
        fcn(x_elem) * wt_elem
    end
end