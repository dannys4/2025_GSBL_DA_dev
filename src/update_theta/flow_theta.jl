export FlowTheta, rhs_theta! #, updateFlowTheta!

abstract type AbstractFlowTheta end

struct FlowTheta <: AbstractFlowTheta
    r::Float64
    β::Float64
    ϑ::Float64
    η::Float64
    t0::Float64
    tf::Float64
    φ0::Vector{Float64}
    φprob::ODEProblem
    φ::ODESolution
end

struct NegGammaFlowTheta <: AbstractFlowTheta
    denom::Float64
    ϑ::Float64
end

function rhs_theta!(dφ, φ, p, t)
    r = p[]
    dφ[] = 2 * t * φ[] / (2 * r^2 * (φ[]^(r + 1)) + t^2)
end

function FlowTheta(dist::GeneralizedGamma; Ne=1, t0=0, tf=1e6)
    r = dist.r
    β = dist.β
    ϑ = dist.ϑ

    # if r == -1
    #     denom = 2 * (1 + β) + Ne
    #     return NegGammaFlowTheta(denom, ϑ)
    # end

    η = r * β - (Ne + 2) / 2

    # Check conditions for validity of ODE approach
    if r < 0 || r > (1 + Ne / 2) / β
        ArgumentError("The ODE approach is not valid in this setting")
    end

    φ0 = [(η / r)^(1 / r)]

    @assert φ0[] >= 0 "The initial condition cannot be negative, got $(φ0[])"

    φprob = ODEProblem(rhs_theta!, φ0, (t0, tf), [r])

    φ = solve(φprob, Vern9())

    FlowTheta(r, β, ϑ, η, t0, tf, φ0, φprob, φ)
end

# Convenient evaluation routine
function (flow::FlowTheta)(t)
    if t < flow.t0 || t > flow.tf
        ArgumentError("t=$t is out of the bounds ($(flow.t0),$(flow.tf))")
    end

    flow.φ(t)[]
end

function (flow::NegGammaFlowTheta)(t)
    return muladd(t, t, 2) / flow.denom
end