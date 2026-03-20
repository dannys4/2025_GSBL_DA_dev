using Distributions, Random
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
            noise = im * randn() + (f.is_dirichlet ? 0. : randn())
            f.ĉ[(l-1)*f.N+k] = noise * exp(-0.5 * k^f.α)
        end
    end
end

struct SmoothSigmoid{D_shift<:UnivariateDistribution,D_scale<:UnivariateDistribution}
    shifts::Vector{Float64}
    scales::Vector{Float64}
    x_lo::Float64
    x_hi::Float64
    shift_dist::D_shift
    scale_dist::D_scale
    left_vals::Vector{Float64}
    right_vals::Vector{Float64}
    function SmoothSigmoid(
        x_lo::Real, x_hi::Real,
        left_vals::AbstractVector{Float64}, right_vals::AbstractVector{Float64};
        shift_dist::Union{Nothing, <:UnivariateDistribution} = nothing,
        scale_dist::Union{Nothing, <:UnivariateDistribution} = nothing,
        shift_mean::Float64 = 0.5, shift_scale::Float64 = 0.15
    )
        length(left_vals) == length(right_vals) || throw(ArgumentError("Need the left and right values to be same length"))
        Nvar = length(left_vals)
        if isnothing(shift_dist)
            shift_dist = (x_hi - x_lo) * Truncated(Normal(shift_mean, shift_scale), 0., 1.) + x_lo
        end
        if isnothing(scale_dist)
            scale_dist = LogNormal(0, 1/2)
        end
        shifts, scales = rand(shift_dist, Nvar), rand(scale_dist, Nvar)
        D_1, D_2 = typeof(shift_dist), typeof(scale_dist)
        new{D_1, D_2}(shifts, scales, x_lo, x_hi, shift_dist, scale_dist, collect(left_vals), collect(right_vals))
    end
end

function regenerate!(f::SmoothSigmoid)
    rand!(f.shift_dist, f.shifts)
    rand!(f.scale_dist, f.scales)
end

function sigmoid(t)
    (1 - (t > 0 ? exp(-t) / (1 + exp(- t)) : 1/(1 + exp(t))))
end

function (f::SmoothSigmoid)(out::AbstractVector, xgrid::AbstractVector)
    (;shifts, scales) = f
    Nvar = length(shifts)

    length(out) == length(xgrid) * Nvar || throw(ArgumentError("Invalid length of argument `out`."))
    sigmoids_l = sigmoid.(scales .* (f.x_lo .- shifts))
    sigmoids_r = sigmoid.(scales .* (f.x_hi .- shifts))
    is_flipped = f.left_vals .< f.right_vals
    u_lo, u_hi = [[s[j] for s in minmax.(f.left_vals, f.right_vals)] for j in 1:2]

    for (i, xi) in enumerate(xgrid)
        for var_idx in 1:Nvar
            shift, scale = shifts[var_idx], scales[var_idx]
            s_l, s_r = sigmoids_l[var_idx], sigmoids_r[var_idx]
            u_lo_var, u_hi_var = u_lo[var_idx], u_hi[var_idx]
            s_eval = (sigmoid(scale * (xi - shift)) - s_l) / (s_r - s_l)
            if !is_flipped[var_idx]
                s_eval = 1 - s_eval
            end
            out[Nvar * (i-1) + var_idx] = (u_hi_var - u_lo_var) * s_eval + u_lo_var
        end
    end
    return out
end

function (f::SmoothSigmoid)(xgrid::AbstractVector)
    Nvar = length(f.shifts)
    out = similar(xgrid, Nvar * length(xgrid))
    f(out, xgrid)
end