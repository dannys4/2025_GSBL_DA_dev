export HLocEnKF, update_x!

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct HLocEnKF{
    LT<:Union{<:Localization,Nothing},
    ThetaT<:AbstractFlowTheta,
    ThetaSpaceT<:Union{Vector{Float64},Matrix{Float64}},
    GT<:Function,
    ET<:InflationType,
    ObsT<:ObsConstraintSystem,
} <: HierarchicalSeqFilter
    "Filter function"
    G::GT

    "Standard deviations of the measurement noise distribution"
    ϵy::ET

    "Structure for observation and constraint"
    sys::ObsT

    "Localization structure"
    Loc::LT

    "Perturbed observation workspace"
    obs_workspace::Matrix{Float64}

    "GeneralizedGamma distribution"
    dist::GeneralizedGamma{Float64}

    "Flow theta"
    flow::ThetaT

    "Penalization coefficients θ associated with the regularization term. Vector if shared, Matrix if not"
    θ::ThetaSpaceT

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is the linear system solved with an iterative solver"
    isiterative::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool

    "Number of optimization (IAS) iterations"
    Niter::Int

    "Optimization relative tolerance"
    rtolθ::Float64

    "Initialization of θ in IAS"
    θinit::Float64

    "Use Ensemble Kalman inversion while finding θ"
    useEnKIOpt::Bool

    "Tolerance for Conjugate Gradient if isiterative"
    cg_tol::Float64
end

function HLocEnKF(
    G::Function,
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsConstraintSystem,
    Loc::Localization,
    dist::GeneralizedGamma,
    θ::Union{Vector{Float64},Matrix{Float64}},
    Δtdyn,
    Δtobs;
    isiterative=false,
    isfiltered=false,
    Niter::Int=40,
    rtolθ::Float64=1e-4,
    θinit::Float64=1.,
    useEnKIOpt::Bool=false,
    cg_tol=1e-6
)
    # @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"
    obs_workspace = Matrix{Float64}(undef, sys.Ny, Ne)

    is_θ_shared = θ isa Vector{Float64}
    flow = FlowTheta(dist; Ne=is_θ_shared ? Ne : 1)

    useEnKIOpt && @assert (is_θ_shared) "If state is stochastic, must have shared θ"

    return HLocEnKF(
        G,
        ϵy,
        sys,
        Loc,
        obs_workspace,
        dist,
        flow,
        θ,
        Δtdyn,
        Δtobs,
        isiterative,
        isfiltered,
        Niter,
        rtolθ,
        θinit,
        useEnKIOpt,
        cg_tol
    )
end

# If no filtering function is provided, use the identity in the constructor.
function HLocEnKF(
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsConstraintSystem,
    Loc::Localization,
    dist::GeneralizedGamma,
    θ::Union{Vector{Float64},Matrix{Float64}},
    Δtdyn,
    Δtobs;
    Niter::Int=10,
    rtolθ::Float64=1e-4,
    θinit::Float64=1.,
    useEnKIOpt::Bool=false,
    cg_tol=1e-6
)
    # @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"
    obs_workspace = Matrix{Float64}(undef, sys.Ny, Ne)

    is_θ_shared = θ isa Vector{Float64}
    flow = FlowTheta(dist; Ne=is_θ_shared ? Ne : 1)

    useEnKIOpt && @assert is_θ_shared "If state is stochastic, must have shared θ"

    return HLocEnKF(
        identity,
        ϵy,
        sys,
        Loc,
        obs_workspace,
        dist,
        flow,
        θ,
        Δtdyn,
        Δtobs,
        false,
        false,
        Niter,
        rtolθ,
        θinit,
        useEnKIOpt,
        cg_tol
    )
end

function Base.show(io::IO, enkf::HLocEnKF)
    println(
        io,
        "Hierarchical localized ensemble Kalman filter with
        iterative solver = $(enkf.isiterative) and
        filtered = $(enkf.isfiltered)",
    )
end

getĈX(enkf::HLocEnKF{<:Localization}, X; kwargs...) = LocalizedEmpiricalCov(X, enkf.Loc; kwargs...)