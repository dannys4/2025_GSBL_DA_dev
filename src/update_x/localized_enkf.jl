import TransportBasedInference2: SeqFilter

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
    isiterative=false,
    isfiltered=false,
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

function update_x!(enkf::LocEnKF, X_forecast, ystar::AbstractVector{Float64}, t, X_analysis)
    Ny = size(ystar, 1)
    Nx = size(X_forecast, 1)
    Ne = size(X_forecast, 2)

    # Use in-place updates
    @assert X_forecast === X_analysis
    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    errs = repeat(-ystar, 1, Ne)
    if enkf.ϵy isa AdditiveInflation
        if has_nonzero_mean(enkf.ϵy)
            errs .+= enkf.ϵy.m
        end
        errs_samp = zeros(Ny)
        for j in axes(errs, 2)
            randn!(errs_samp)
            mul!(@view(errs[:, j]), enkf.ϵy.σ, errs_samp, true, true)
        end
        # E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    # Only form covariance mat iff enkf is not iterative
    ĈX = getĈX(enkf, X_forecast, Nx, Ny; with_matrix=!enkf.isiterative)

    if enkf.isiterative
        enkf.sys.CX = ĈX

        sys_op = LinearMap{Float64}(
            (y, x) -> mul!(y, enkf.sys, x),
            Ny;
            issymmetric=true,
            isposdef=false,
        )
    else
        # Update covariance matrix
        enkf.sys.CX = ĈX.CXloc

        # @show cond(sys_mat)
        sys_mat = bunchkaufman!(Matrix(enkf.sys))
    end


    # Compute Kalman-update in a matrix-free way

    # yi = zeros(Ny)
    iterative_RHS = enkf.isiterative ? similar(ystar) : nothing
    δi = zeros(Nx)

    for i = 1:3
        @info "" i
        err_i = @view errs[:, i]
        xi = @view X_analysis[:, i]

        # yi .= enkf.sys.H * xi + E[:, i] - ystar
        mul!(err_i, enkf.sys.H, xi, true, true)
        @info "Calc yi."
        # @assert isapprox(yi, enkf.sys.H * xi, atol=1e-8)


        if enkf.isiterative
            # Invert sys_op
            copy!(iterative_RHS, err_i)
            cg!(err_i, sys_op, iterative_RHS; log=false, verbose=true, reltol=1e-2)
        else
            # yi .= sys_mat \ yi
            ldiv!(sys_mat, yi)
        end
        @info "solved."

        # δi .= enkf.sys.H' * yi
        mul!(δi, enkf.sys.H', err_i)
        @info "finished δi"

        # xi .-= ĈX * δi
        mul!(xi, ĈX, δi, true, -1)
    end
end

function (enkf::LocEnKF)(X, ystar::AbstractVector{Float64}, t::Float64)

    # Update x 
    update_x!(enkf, X, ystar, t, X)

    return X
end

getĈX(enkf::LocEnKF, X, Nx, Ny; with_matrix=true) = LocalizedEmpiricalCov(X, enkf.Loc; with_matrix)