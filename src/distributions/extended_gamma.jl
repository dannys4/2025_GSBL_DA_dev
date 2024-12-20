export log_integrand_extended_dist,
    integrand_extended_dist,
    unnorm_logpdf!,
    unnorm_logpdf,
    unnorm_pdf!,
    unnorm_pdf,
    Znorm!,
    Znorm,
    update_Znorm!,
    cdf!,
    cdf,
    ccdf!,
    ccdf,
    ExtendedGamma

log_integrand_extended_dist(α, κ, p, q, t) = (α - 1) * log(t) - t^κ / p - q / t

integrand_extended_dist(α, κ, p, q, t) = exp(log_integrand_extended_dist(α, κ, p, q, t))

unnorm_logpdf!(out, α, κ, p, q, t) = out .= log_integrand_extended_dist(α, κ, p, q, t)
unnorm_pdf!(out, α, κ, p, q, t) = out .= integrand_extended_dist(α, κ, p, q, t)

# Routine to compute the un-normalized probability density function
function unnorm_logpdf(α, κ, p, q, t)
    out = zeros(eltype(t), 1)
    unnorm_logpdf!(out, α, κ, p, q, t)
    return out
end

# Routine to compute the normalizing constant of the probability density function
function Znorm!(out, α, κ, p, q)
    quadgk!((b, a) -> unnorm_pdf!(b, α, κ, p, q, a), out, 0.0, Inf)
    nothing
end

# Routine to compute the normalizing constant of the probability density function
function Znorm(α, κ, p, q)
    out = zeros(eltype(t), 1)
    Znorm!(out, α, κ, p, q)
    return out
end

"""
ExtendedGamma

A structure for the extended (generalized) Gamma distrubution given by

```math
f(x; \\alpha, \\kappa, p , q) = C x^{\\alpha - 1} \\exp(-\\frac{x^\\kappa}{p} - \\frac{q}{x}), 
\\quad x > 0
```
"""
struct ExtendedGamma <: ContinuousUnivariateDistribution
    α::Float64
    κ::Float64
    p::Float64
    q::MVector{1,Float64}
    # Normalizing constant
    Z::MVector{1,Float64}
    cache::MVector{1,Float64}
end

function ExtendedGamma(α, κ, p, q)
    @assert typeof(q) <: Real "q should be a real number."
    cache = MVector{1,Float64}(0.0)
    # Compute normalizing constant
    Znorm!(cache, α, κ, p, q)
    Zdist = copy(cache)
    return ExtendedGamma(α, κ, p, MVector{1,Float64}(q), Zdist, cache)
end

function show(io::IO, dist::ExtendedGamma)
    print(io, "ExtendedGamma (α=$(dist.α), κ=$(dist.κ), p=$(dist.p), q = $(dist.q[1]))")
end

integrand_extended_dist(dist::ExtendedGamma, t) =
    integrand_extended_dist(dist.α, dist.κ, dist.p, dist.q[1], t)

unnorm_logpdf!(out, dist::ExtendedGamma, t) =
    unnorm_logpdf!(out, dist.α, dist.κ, dist.p, dist.q[1], t)

function unnorm_logpdf(dist::ExtendedGamma, t)
    unnorm_logpdf!(dist.cache, dist, t)
    return dist.cache
end

function logpdf!(out, dist::ExtendedGamma, t)
    unnorm_logpdf!(out, dist, t)
    out .-= log(dist.Z[1])
    nothing
end

function Distributions.logpdf(dist::ExtendedGamma, t)
    logpdf!(dist.cache, dist, t)
    return dist.cache[1]
end

unnorm_pdf!(out, dist::ExtendedGamma, t) =
    unnorm_pdf!(out, dist.α, dist.κ, dist.p, dist.q[1], t)

function unnorm_pdf(dist::ExtendedGamma, t)
    unnorm_pdf!(dist.cache, dist, t)
    return dist.cache[1]
end

function Distributions.pdf!(out, dist::ExtendedGamma, t)
    unnorm_pdf!(out, dist.cache, dist, t)
    out ./= dist.Z
    nothing
end

# function Distributions.pdf(dist::ExtendedGamma, t)
#     Distributions.pdf!(dist.cache, dist, t)
#     return dist.cache
# end

Znorm!(out, dist::ExtendedGamma) = Znorm!(out, dist.α, dist.κ, dist.p, dist.q[1])

function Znorm(dist::ExtendedGamma)
    Znorm!(dist.cache, dist)
    return dist.cache
end

update_Znorm!(dist::ExtendedGamma) = Znorm!(dist.Z, dist)

function cdf!(out, dist::ExtendedGamma, x)
    quadgk!((b, a) -> unnorm_pdf!(b, dist, a), out, 0.0, x)[1]
    out ./= dist.Z[1]
    nothing
end

function Distributions.cdf(dist::ExtendedGamma, x)
    cdf!(dist.cache, dist, x)
    return dist.cache
end

# Complementary of the CDF to 1
function ccdf!(out, dist::ExtendedGamma, x)
    quadgk!((b, a) -> unnorm_pdf!(b, dist, a), out, x, Inf)[1]
    out ./= dist.Z[1]
    nothing
end

# Complementary of the CDF to 1
function Distributions.ccdf(dist::ExtendedGamma, x)
    ccdf!(dist.cache, dist, x)
    return dist.cache
end
