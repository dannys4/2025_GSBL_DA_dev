import TransportBasedInference: SeqFilter

export HEnKF, update_x!

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical 
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct HEnKF <: SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Structure for observation and constraint"
    sys::ObsConstraintSystem

    "GeneralizedGamma distribution"
    dist::GeneralizedGamma

    "Flow theta"
    flow::FlowTheta

    "Penalization coefficients θ associated with the regularization term"
    θ::Vector{Float64}

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is θ shared"
    isθshared::Bool

    "Boolean: is the linear system solved with an iterative solver"
    isiterative::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function HEnKF(
    G::Function,
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsConstraintSystem,
    dist::GeneralizedGamma,
    θ::Union{Vector{Float64},Matrix{Float64}},
    Δtdyn,
    Δtobs;
    isiterative = false,
    isfiltered = false
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    flow = FlowTheta(dist; Ne = Ne)

    if typeof(θ) <: Vector{Float64}
        isθshared = true
    elseif typeof(θ) <: Matrix{Float64}
        isθshared = false
    end

    return HEnKF(
        G,
        ϵy,
        sys,
        dist,
        flow,
        θ,
        Δtdyn,
        Δtobs,
        isθshared,
        isiterative,
        isfiltered
    )
end

# If no filtering function is provided, use the identity in the constructor.
function HEnKF(
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsConstraintSystem,
    dist::GeneralizedGamma,
    θ::Vector{Float64},
    Δtdyn,
    Δtobs,
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    flow = FlowTheta(dist; Ne = Ne)

    if typeof(θ) <: Vector{Float64}
        isθshared = true
    elseif typeof(θ) <: Matrix{Float64}
        isθshared = false
    end

    return HEnKF(x -> x, ϵy, sys, dist, flow, θ, Δtdyn, Δtobs, isθshared, false, false)
end

function Base.show(io::IO, enkf::HEnKF)
    println(
        io,
        "Hierarchical ensemble Kalman filter with
        iterative solver = $(enkf.isiterative) and
        filtered = $(enkf.isfiltered)",
    )
end
