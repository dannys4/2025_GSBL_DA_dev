import TransportBasedInference: SeqFilter

export LocEnKF

"""
$(TYPEDEF)

A structure for the variational formulation of the hierarchical 
stochastic ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct LocEnKF <: SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Structure for observation"
    sys::ObsSystem

    "Localization structure"
    Loc::Localization

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is the linear system solved with an iterative solver"
    isiterative::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function LocEnKF(
    G::Function,
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsSystem,
    Loc::Localization,
    Δtdyn,
    Δtobs;
    isiterative = false,
    isfiltered = false,
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocEnKF(G, ϵy, sys, Loc, Δtdyn, Δtobs, isiterative, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LocEnKF(
    Ne::Int64,
    ϵy::InflationType,
    sys::ObsSystem,
    Loc::Localization,
    Δtdyn,
    Δtobs,
)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocEnKF(x -> x, ϵy, sys, Loc, Δtdyn, Δtobs, false, false)
end

function Base.show(io::IO, enkf::LocEnKF)
    println(
        io,
        "Localized ensemble Kalman filter with iterative = $(enkf.isiterative) and filtered = $(enkf.isfiltered)",
    )
end

function update_x!(enkf::LocEnKF, X, ystar::Vector{Float64}, t)


    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)

    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if typeof(enkf.ϵy) <: AdditiveInflation
        E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    # ĈX = EmpiricalCov(X[Ny+1:Ny+Nx,:])
    ĈX = LocalizedEmpiricalCov(X[Ny+1:Ny+Nx, :], enkf.Loc; with_matrix = true)
    ĈX_op = FunctionMap{Float64,true}(
        (y, x) -> mul!(y, ĈX, x),
        Nx;
        issymmetric = true,
        isposdef = false,
    )

    # Update covariance matrix
    enkf.sys.CX[1] = ĈX_op

    sys_op = LinearMaps.FunctionMap{Float64,true}(
        (y, x) -> mul!(y, enkf.sys, x),
        Ny;
        issymmetric = true,
        isposdef = true,
    )

    if enkf.isiterative == false
        sys_mat = zeros(Ny, Ny)

        ei = zeros(Ny)
        for i = 1:Ny
            fill!(ei, 0.0)
            ei[i] = 1.0
            sys_mat[:, i] = sys_op * ei
        end

        sys_mat = factorize(Symmetric(sys_mat))
    end

    # Compute Kalman-update in a matrix-free way

    yi = zeros(Ny)
    δi = zeros(Nx)

    for i = 1:Ne
        xi = view(X, Ny+1:Ny+Nx, i)

        mul!(yi, enkf.sys.H, xi)
        @assert isapprox(yi, enkf.sys.H * xi, atol = 1e-8)

        yi .+= E[:, i] - ystar

        if enkf.isiterative == false
            yi .= sys_mat \ yi
        else
            # Invert sys_op
            # ldiv!(ys_i, sys_mat, ys_i)
            # @show typeof(ys_i)
            # @show cg(sys_op, ys_i; log = true)[2]/
            # @show cg(sys_op, yi; log = true, reltol = 1e-3)
            cg!(yi, sys_op, copy(yi); log = false, reltol = 1e-3)
        end

        δi .= enkf.sys.H' * yi

        xi .+= -(ĈX * δi)
    end
end

function (enkf::LocEnKF)(X, ystar::Array{Float64,1}, t::Float64)

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)

    # Update x 
    update_x!(enkf, X, ystar, t)

    return X
end
