export FlowTheta, rhs_theta!, updateFlowTheta!

struct FlowTheta
    r::Float64
    β::Float64
    ϑ::Float64
    η::Float64
    t0::Float64
    tf::Float64
    φ0::Vector{Float64}
    φprob::ODEProblem
    φ::Vector{ODESolution}
end


function rhs_theta!(dφ, φ, p, t)
    r = p[1]
    dφ[1] = 2 * t * φ[1] / (2 * r^2 * φ[1]^(r + 1) + t^2)
end

function FlowTheta(dist::GeneralizedGamma; Ne = 1, t0 = 0, tf = 1e6)
    r = dist.r
    β = dist.β
    ϑ = dist.ϑ
    η = r * β - (Ne + 2) / 2

    # Check conditions for validity of ODE approach 
    @assert (r < 0 && η < -(Ne + 2) / 2) || (r > 0 && η > 0) "The ODE approach is not valid in this setting "

    φ0 = [(η / r)^(1 / r)]

    @assert φ0[1] >= 0 "The initial condition cannot be negative"


    φprob = ODEProblem(rhs_theta!, φ0, (t0, tf), [r])

    φ = solve(φprob, Feagin14())

    return FlowTheta(r, β, ϑ, η, t0, tf, φ0, φprob, [φ])
end

"""
    updateFlowTheta!(flow::FlowTheta, φ0::Vector{Float64})

TBW
"""
function updateFlowTheta!(flow::FlowTheta, φ0::Vector{Float64})
    copy!(flow.φ0, φ0)
    remake(flow.φprob, u0 = φ0)
    flow.φ[1] = solve(flow.φprob, Tsit5())
end

# Convenient evaluation routine
function (flow::FlowTheta)(t)
    if t < flow.t0 || t > flow.tf
        error(string(t) * " is out of the bounds")
    else
        # @show t
        return flow.φ[1](t)[1]
    end
end
