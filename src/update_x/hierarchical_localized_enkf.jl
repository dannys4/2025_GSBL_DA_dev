import TransportBasedInference2: SeqFilter

export HLocEnKF, update_x!

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical 
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct HLocEnKF <: SeqFilter
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

    "Number of optimization (ALS) iterations"
    Niter::Int

    "Optimization relative tolerance"
    rtolθ::Float64
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
    rtolθ::Float64 = 1e-4
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    flow = FlowTheta(dist; Ne = Ne)

    if typeof(θ) <: Vector{Float64}
        isθshared = true
    elseif typeof(θ) <: Matrix{Float64}
        isθshared = false
    end

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
        rtolθ
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
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    flow = FlowTheta(dist; Ne = Ne)

    if typeof(θ) <: Vector{Float64}
        isθshared = true
    elseif typeof(θ) <: Matrix{Float64}
        isθshared = false
    end

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
        rtolθ
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


function (enkf::Union{HEnKF,HLocEnKF})(
    X,
    ystar::Array{Float64,1},
    t::Float64
)

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)

    if enkf.isθshared
        # Initial guess?
        enkf.θ .= one(enkf.sys.Nz)
        
        θold = zero(enkf.θ)
        for n = 1:enkf.Niter
            copy!(θold, enkf.θ)

            # Update x 
            update_x!(enkf, X, enkf.θ, ystar, t)

            # Update theta
            update_θ!(enkf, X, enkf.θ, ystar, t)

            if norm(enkf.θ - θold) / norm(θold) < enkf.rtolθ
                break
            end
        end
    else
        enkf.θ .= rand(enkf.dist, enkf.sys.Ns, enkf.sys.Ne)
        θold = zero(enkf.θ)
        for n = 1:enkf.Niter
            copy!(θold, enkf.θ)

            # Update theta
            update_θ!(enkf, X, enkf.θ, ystar, t)

            # Update x 
            update_x!(enkf, X, enkf.θ, ystar, t)

            if norm(enkf.θ - θold) / norm(θold) < enkf.rtolθ
                break
            end
        end
    end
    return X, enkf.θ
end
