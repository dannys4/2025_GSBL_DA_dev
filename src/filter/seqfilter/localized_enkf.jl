export LocEnKF

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct LocEnKF{
    GT<:Function,
    ET<:InflationType,
    ObsT<:ObsSystem,
    LT<:Localization
} <: TransportBasedInference2.SeqFilter
    "Filter function"
    G::GT

    "Standard deviations of the measurement noise distribution"
    ϵy::ET

    "Structure for observation"
    sys::ObsT

    "Localization structure"
    Loc::LT

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is the linear system solved with an iterative solver"
    isiterative::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool

    "Tolerance for Conjugate Gradient, if isiterative"
    cg_tol::Float64
end

function LocEnKF(
    G::Function,
    ϵy::InflationType,
    sys::ObsSystem,
    Loc::Localization,
    Δtdyn,
    Δtobs;
    isiterative=false,
    isfiltered=false,
    cg_tol=1e-6
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocEnKF{}(G, ϵy, sys, Loc, Δtdyn, Δtobs, isiterative, isfiltered, cg_tol)
end

# If no filtering function is provided, use the identity in the constructor.
function LocEnKF(
    ϵy::InflationType,
    sys::ObsSystem,
    Loc::Localization,
    Δtdyn,
    Δtobs;
    isiterative=false,
    isfiltered=false,
    cg_tol=1e-6
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocEnKF{}(identity, ϵy, sys, Loc, Δtdyn, Δtobs, isiterative, isfiltered, cg_tol)
end

function Base.show(io::IO, enkf::LocEnKF)
    println(
        io,
        "Localized ensemble Kalman filter with iterative = $(enkf.isiterative) and filtered = $(enkf.isfiltered)",
    )
end
getĈX(enkf::LocEnKF, X; with_matrix=true, workspace_sparsity=nothing) = LocalizedEmpiricalCov(X, enkf.Loc; with_matrix, workspace_sparsity)
