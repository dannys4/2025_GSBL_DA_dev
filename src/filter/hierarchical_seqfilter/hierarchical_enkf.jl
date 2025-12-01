export HEnKF, update_x!

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct HEnKF{
    ThetaT<:AbstractFlowTheta,
    ThetaSpaceT<:Union{Vector{Float64},Matrix{Float64}}
} <: HierarchicalSeqFilter

    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Structure for observation and constraint"
    sys::ObsConstraintSystem

    "Perturbed Observation Workspace"
    obs_workspace::Matrix{Float64}

    "GeneralizedGamma distribution"
    dist::GeneralizedGamma

    "Flow theta"
    flow::ThetaT

    "Penalization coefficients θ associated with the regularization term"
    θ::ThetaSpaceT

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is the linear system solved with an iterative solver"
    isiterative::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool

    "Number of optimization (IAS) steps"
    Niter::Int

    "Relative tolerance of IAS optimization"
    rtolθ::Float64

    "Use Ensemble Kalman inversion while finding θ"
    useEnKIOpt::Bool
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
    isiterative=false,
    isfiltered=false,
    Niter::Int=40,
    rtolθ::Float64=1e-4,
    useEnKIOpt::Bool=false,
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    is_θ_shared = θ isa Vector{Float64}
    flow = FlowTheta(dist; Ne=is_θ_shared ? Ne : 1)

    useEnKIOpt && @assert (is_θ_shared) "If state is stochastic, must have shared θ"

    return HEnKF(
        G,
        ϵy,
        sys,
        dist,
        flow,
        θ,
        Δtdyn,
        Δtobs,
        isiterative,
        isfiltered,
        Niter,
        rtolθ,
        useEnKIOpt,
    )
end

# If no filtering function is provided, use the identity in the constructor.
function HEnKF(
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsConstraintSystem,
    dist::GeneralizedGamma,
    θ::Union{Vector{Float64},Matrix{Float64}},
    Δtdyn,
    Δtobs;
    Niter::Int=40,
    rtolθ::Float64=1e-4,
    useEnKIOpt::Bool=false,
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    is_θ_shared = θ isa Vector{Float64}
    flow = FlowTheta(dist; Ne=is_θ_shared ? Ne : 1)

    useEnKIOpt && @assert is_θ_shared "If state is stochastic, must have shared θ"

    return HEnKF(
        identity,
        ϵy,
        sys,
        dist,
        flow,
        θ,
        Δtdyn,
        Δtobs,
        false,
        false,
        Niter,
        rtolθ,
        useEnKIOpt,
    )
end

function Base.show(io::IO, enkf::HEnKF)
    println(
        io,
        "Hierarchical ensemble Kalman filter with
        iterative solver = $(enkf.isiterative) and
        filtered = $(enkf.isfiltered)",
    )
end

getĈX(::HEnKF, X, Nx, Ny; with_matrix=true) = EmpiricalCov(X; with_matrix)
