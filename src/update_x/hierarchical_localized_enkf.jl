export HLocEnKF, update_x!

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical 
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct HLocEnKF <: HierarchicalSeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Structure for observation and constraint"
    sys::ObsConstraintSystem

    "Localization structure"
    Loc::Localization

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

    "Number of optimization (IAS) iterations"
    Niter::Int

    "Optimization relative tolerance"
    rtolθ::Float64

    "Initialization of θ in IAS"
    θinit::Float64

    "Use Ensemble Kalman inversion while finding θ"
    useEnKIOpt::Bool
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
    isiterative = false,
    isfiltered = false,
    Niter::Int = 40,
    rtolθ::Float64 = 1e-4,
    θinit::Float64 = 1.,
    useEnKIOpt::Bool = false,
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    flow = FlowTheta(dist; Ne = Ne)

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
        useEnKIOpt
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
    Niter::Int = 40,
    rtolθ::Float64 = 1e-4,
    θinit::Float64 = 1.,
    useEnKIOpt::Bool = false
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    flow = FlowTheta(dist; Ne = Ne)

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
        useEnKIOpt
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


function (enkf::HierarchicalSeqFilter)(
    X_forecast,
    ystar::Array{Float64,1},
    t::Float64
)
    X_analysis = deepcopy(X_forecast)
    X_forecast_loop = enkf.useEnKIOpt ? X_analysis : X_forecast
    if enkf.isθshared
        # Initial guess?
        fill!(enkf.θ, enkf.θinit)
        
        θold = zero(enkf.θ)
        for _ = 1:enkf.Niter
            copy!(θold, enkf.θ)

            # Update x 
            update_x!(enkf, X_forecast_loop, enkf.θ, ystar, t, X_analysis)

            # Update theta
            update_θ!(enkf, X_analysis, enkf.θ, ystar, t)

            if norm(enkf.θ - θold) / norm(θold) < enkf.rtolθ
                break
            end
        end
    else
        enkf.θ .= rand(enkf.dist, enkf.sys.Ns, enkf.sys.Ne)
        θold = zero(enkf.θ)

        for _ = 1:enkf.Niter
            copy!(θold, enkf.θ)

            # Update theta
            update_θ!(enkf, X_analysis, enkf.θ, ystar, t)

            # Update x 
            update_x!(enkf, X_forecast_loop, enkf.θ, ystar, t, X_analysis)

            if norm(enkf.θ - θold) / norm(θold) < enkf.rtolθ
                break
            end
        end
    end
    update_x!(enkf, X_forecast, enkf.θ, ystar, t, X_forecast)
    return X_forecast, enkf.θ
end

getĈX(enkf::HLocEnKF, X, Nx, Ny; with_matrix=true) = LocalizedEmpiricalCov(X[Ny+1:Ny+Nx, :], enkf.Loc; with_matrix)