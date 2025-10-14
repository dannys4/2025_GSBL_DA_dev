import TransportBasedInference2

export EnKF

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct EnKF <: SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Structure for observation"
    sys::ObsSystem

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is the linear system solved with an iterative solver"
    isiterative::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function EnKF(
    G::Function,
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsSystem,
    Δtdyn,
    Δtobs;
    isiterative=false,
    isfiltered=false,
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return EnKF(G, ϵy, sys, Δtdyn, Δtobs, isiterative, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function EnKF(Ne::Int64, ϵy::InflationType, sys::ObsSystem, Δtdyn, Δtobs)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return EnKF(x -> x, ϵy, sys, Δtdyn, Δtobs, false, false)
end

function Base.show(io::IO, enkf::EnKF)
    println(
        io,
        "Ensemble Kalman filter with
iterative solver = $(enkf.isiterative) and
filtered = $(enkf.isfiltered)",
    )

end

getĈX(::EnKF, X; with_matrix=true) = EmpiricalCov(X; with_matrix)
