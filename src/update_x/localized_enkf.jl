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

function update_x!(enkf::LocEnKF, X_forecast::AbstractMatrix{Float64}, ystar::AbstractVector{Float64}, t, X_analysis, verbose)
    Ny = length(ystar)
    Nx, Ne = size(X_forecast)

    # Use in-place updates
    @assert X_forecast === X_analysis

    # Generate observational noise samples
    errs = repeat(ystar, 1, Ne)
    if enkf.ϵy isa AdditiveInflation
        if has_nonzero_mean(enkf.ϵy)
            errs .-= enkf.ϵy.m
        end
        errs_samp = zeros(Ny)
        for j in axes(errs, 2)
            randn!(errs_samp)
            mul!(@view(errs[:, j]), enkf.ϵy.σ, errs_samp, true, true)
        end
        # E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    # Only form covariance mat iff enkf is not iterative
    # workspace_sparsity = findall(isnan, enkf.sys.H' * fill(NaN, size(enkf.sys.H, 1)))
    ĈX_op = getĈX(enkf, X_forecast, Nx, Ny; with_matrix=!enkf.isiterative)
    # Update covariance matrix
    enkf.sys.CX = ĈX_op

    if enkf.isiterative
        # Creates linear map object
        sys_op = enkf.sys
    else
        # @show cond(sys_mat)
        sys_mat = bunchkaufman!(Matrix(enkf.sys))
    end
    verbose && @info "Made sys op."

    # Compute Kalman-update in a matrix-free way

    # yi = zeros(Ny)
    iterative_RHS = enkf.isiterative ? similar(ystar) : nothing
    δi = zeros(Nx)
    time_start = Base.time_ns()
    for i = 1:Ne
        verbose && @info "Ensemble member $i"
        err_i = @view errs[:, i]
        xi = @view X_analysis[:, i]

        # mul!(err_i, enkf.sys.H, xi, true, true)
        err_i .-= enkf.sys.H * xi
        yi = err_i
        verbose && @info "Calc'ed yi."

        if enkf.isiterative
            # Invert sys_op
            copy!(iterative_RHS, yi)
            fill!(yi, zero(eltype(yi)))
            cg!(yi, sys_op, iterative_RHS; log=false, verbose=false, reltol=enkf.cg_tol)
        else
            # yi .= sys_mat \ yi
            ldiv!(sys_mat, yi)
        end
        verbose && @info "solved."

        # mul!(δi, enkf.sys.H', yi)
        δi .= enkf.sys.H' * yi
        verbose && @info "finished δi."

        xi .= xi + ĈX_op * δi
        # mul!(xi, ĈX, δi, true, true)
    end
    time_elapsed = (Base.time_ns() - time_start) / 1.0e9
    verbose && @info "Took $(time_elapsed)s"
end

function (enkf::LocEnKF)(X, ystar::AbstractVector{Float64}, t::Float64, verbose::Bool)
    # Update x
    update_x!(enkf, X, ystar, t, X, verbose)

    return X
end

getĈX(enkf::LocEnKF, X, Nx, Ny; with_matrix=true, workspace_sparsity=nothing) = LocalizedEmpiricalCov(X, enkf.Loc; with_matrix, workspace_sparsity)