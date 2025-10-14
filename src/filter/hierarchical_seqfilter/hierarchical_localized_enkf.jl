export HLocEnKF, update_x!

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct HLocEnKF{
    ThetaT<:AbstractFlowTheta,
    GT<:Function,
    ET<:InflationType,
    ObsT<:ObsConstraintSystem,
    LT<:Localization
} <: HierarchicalSeqFilter
    "Filter function"
    G::GT

    "Standard deviations of the measurement noise distribution"
    ϵy::ET

    "Structure for observation and constraint"
    sys::ObsT

    "Localization structure"
    Loc::LT

    "GeneralizedGamma distribution"
    dist::GeneralizedGamma{Float64}

    "Flow theta"
    flow::ThetaT

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

    flow = FlowTheta(dist; Ne=Ne)

    isθshared = (θ isa Vector)

    return HLocEnKF(
        G,
        ϵy,
        sys,
        Loc,
        dist,
        flow,
        θ,
        Δtdyn,
        Δtobs,
        isθshared,
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
    Niter::Int=40,
    rtolθ::Float64=1e-4,
    θinit::Float64=1.,
    useEnKIOpt::Bool=false,
    cg_tol=1e-6
)
    # @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    flow = FlowTheta(dist; Ne=Ne)

    isθshared = (θ isa Vector)
    useEnKIOpt && @assert isθshared "If state is stochastic, expected θ to be shared"
    return HLocEnKF(
        x -> x,
        ϵy,
        sys,
        Loc,
        dist,
        flow,
        θ,
        Δtdyn,
        Δtobs,
        isθshared,
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

function getĈX_op(enkf::HierarchicalSeqFilter, X::AbstractMatrix; kwargs...)
    ĈX = getĈX(enkf, X; kwargs...)
    ĈX_mat = Matrix(ĈX)
    if isnothing(ĈX_mat)
        return ĈX
    else
        return LinearMap(ĈX_mat)
    end
end

getĈX(enkf::HLocEnKF, X; kwargs...) = LocalizedEmpiricalCov(X, enkf.Loc; kwargs...)