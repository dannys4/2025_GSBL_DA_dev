import TransportBasedInference2: SeqFilter

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
    isiterative = false,
    isfiltered = false,
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

function update_x!(enkf::EnKF, X, ystar::Vector{Float64}, t)


    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)

    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if enkf.ϵy isa AdditiveInflation
        E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    ĈX = getĈX(enkf, X, Nx, Ny)
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

    if !enkf.isiterative
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

        if !enkf.isiterative
            yi .= sys_mat \ yi
        else
            # Invert sys_op
            cg!(yi, sys_op, copy(yi); log = false, reltol = 1e-3)
        end

        δi .= enkf.sys.H' * yi

        xi .+= -(ĈX * δi)
    end
end

function (enkf::EnKF)(X, ystar::Array{Float64,1}, t::Float64)

    # Update x 
    update_x!(enkf, X, ystar, t)

    return X
end

getĈX(::EnKF, X, Nx, Ny; with_matrix=true) = EmpiricalCov(X[Ny+1:Ny+Nx, :];with_matrix)