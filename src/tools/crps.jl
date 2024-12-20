export CRPS_gaussian, CRPS_quadrature, CRPS 

"""
CRPS calculation for a univariate Gaussian distribution
"""
function CRPS_gaussian(p::Normal{Float64}, x)
    μ = mean(p)
    σ = std(p)
    z = (x-μ)/σ
    return σ*(z*(2*cdf(Normal(), z) - 1) + 2*pdf(Normal(), z) - 1/sqrt(π))
end

"""
CRPS calculation based on quadrature rule
"""
function CRPS_quadrature(X::AbstractVector, x::Float64)
    ΦX = ecdf(X)
    Φx = ecdf([x])
    return quadgk(t->(ΦX(t) - Φx(t))^2, -Inf, Inf)[1]
end

"""
Fastest implementation of CRPS
"""
function CRPS(X::AbstractVector, x::Float64)
    N = length(X)
    
    term1 = 0.0
    term2 = 0.0
    @inbounds for i=1:N
        xi = X[i]
        term1 += abs(xi - x)
        # Exploit symmetry and multiply off-diagonal entries by 2.
        for j=i+1:N
            xj = X[j]
            term2 += 2*abs(xj - xi)
        end
    end
    
    term1 *= 1/N
    term2 *= 1/N^2
    
    return term1 - 0.5*term2
end

