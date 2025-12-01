# In this version, there is a jump coefficient θ for each ensemble member

function update_x!(
    enkf::HierarchicalSeqFilter,
    X_forecast,
    perturbed_obs::Matrix{Float64},
    ĈX_op,
    θ::Matrix{Float64},
    t,
    X_analysis,
    verbose::Bool
)
    verbose && @info "x being updated"

    # Update weight vector θ
    if !(enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64})
        ArgumentError("Wrong type for Cθ")
    end

    Ny = size(perturbed_obs, 1)
    Nx = size(X_forecast, 1)
    Ne = size(X_forecast, 2)
    Ne = size(X_forecast, 2)
    Nz = enkf.sys.Nz

    @assert size(θ) == (Nz, Ne)
    @assert size(perturbed_obs, 2) == Ne

    verbose && @info "noise sampled"

    # Update covariance matrix
    enkf.sys.CX = ĈX_op
    verbose && @info "CX copied"

    fill!(enkf.sys.Cθ.lmap.diag, 0.)

    if enkf.isiterative
        sys_op = enkf.sys
        precond = Diagonal(sys_op)
    else
        sys_mat = Hermitian(Matrix(enkf.sys))
        diag_theta_idx = diagind(sys_mat)[end-Nz+1:end]
        sys_mat_ref = sys_mat[diag_theta_idx]
    end
    verbose && @info "Got sys op"

    # Compute Kalman-update in a matrix-free way
    ys_i = ObsConstraintVector(Ny, Nz)
    tmp = zeros(length(ys_i))
    solve_out = similar(tmp)

    δi = zeros(Nx)

    if X_analysis !== X_forecast
        copy!(X_analysis, X_forecast)
    end

    for i = 1:Ne
        verbose && @info "Ensemble member $i"
        fill!(ys_i, zero(eltype(ys_i)))
        fill!(δi, zero(eltype(δi)))
        xi_forecast = @view X_forecast[:, i]
        xi_analysis = @view X_analysis[:, i]

        θi = @view θ[:, i]
        yi = observation(ys_i)
        si = constraint(ys_i)

        obs_i = @view perturbed_obs[:, i]

        copy!(yi, obs_i)
        mul!(yi, enkf.sys.H, xi_forecast, -1, true)
        mul!(si, enkf.sys.S, xi_forecast, -1, false)
        copy!(tmp, ys_i)

        verbose && @info "Start solve"

        if enkf.isiterative
            copy!(enkf.sys.Cθ.lmap.diag, θi)
            # Invert sys_op
            cg!(solve_out, sys_op, tmp; log=false, verbose=false, reltol=enkf.cg_tol, Pl=precond)
            copy!(ys_i, solve_out)
        else
            for (diag_idx, mat_idx) in enumerate(diag_theta_idx)
                sys_mat[mat_idx] = sys_mat_ref[diag_idx] + θi[diag_idx]
            end
            # Not worth factorizing because matrix changes.
            solve_out .= sys_mat \ tmp
            copy!(ys_i, solve_out)
        end

        # δi .= enkf.sys.H' * observation(ys_i) + enkf.sys.S' * constraint(ys_i)
        # xi_analysis .= xi_analysis + ĈX_op * δi
        mul!(δi, enkf.sys.H', observation(ys_i))
        mul!(δi, enkf.sys.S', constraint(ys_i), true, true)
        mul!(xi_analysis, ĈX_op, δi, true, true)
    end
end
