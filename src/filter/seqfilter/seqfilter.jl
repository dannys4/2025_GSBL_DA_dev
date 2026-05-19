
function update_x!(enkf::SeqFilter, X_forecast::AbstractMatrix{Float64}, ystar::AbstractVector{Float64}, t, X_analysis, verbose)
    Ny = length(ystar)
    Nx, Ne = size(X_forecast)

    # Use in-place updates
    @assert X_forecast === X_analysis

    # Generate observational noise samples
    errs = repeat(ystar, 1, Ne)
    enkf.ϵy(errs, t, true)

    # Only form covariance mat iff enkf is not iterative
    # workspace_sparsity = findall(isnan, enkf.sys.H' * fill(NaN, size(enkf.sys.H, 1)))
    ĈX_op = getĈX(enkf, X_forecast; with_matrix=!enkf.isiterative)
    # Update covariance matrices
    enkf.sys.CX = ĈX_op
    enkf.sys.Cϵ = LinearMap(get_cov(enkf.ϵy, t))

    if enkf.isiterative
        # Creates linear map object
        sys_op = enkf.sys
        precond = Diagonal(sys_op)
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
            cg!(yi, sys_op, iterative_RHS; log=false, verbose=false, reltol=enkf.cg_tol, Pl=precond)
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

function (enkf::SeqFilter)(X, ystar::AbstractVector{Float64}, t::Float64, verbose::Bool)
    # Update x
    update_x!(enkf, X, ystar, t, X, verbose)

    return X
end
