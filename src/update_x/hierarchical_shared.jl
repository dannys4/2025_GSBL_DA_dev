# In this version, there is a jump coefficient θ shared across the ensemble members

function update_x!(
    enkf::HierarchicalSeqFilter,
    X_forecast,
    perturbed_obs::Matrix{Float64},
    ĈX_op,
    θ::Vector{Float64},
    t,
    X_analysis,
    verbose::Bool
)
    verbose && @info "x being updated"
    @assert enkf.isθshared

    # Update weight vector θ
    if !(enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64})
        ArgumentError("Wrong type for Cθ")
    end

    Ny = size(perturbed_obs, 1)
    Nx = size(X_forecast, 1)
    Ne = size(X_forecast, 2)
    Ne = size(X_forecast, 2)
    Nz = enkf.sys.Nz

    @assert size(θ, 1) == Nz
    @assert size(perturbed_obs, 2) == Ne

    verbose && @info "noise sampled"

    # Update covariance matrix
    enkf.sys.CX = ĈX_op
    verbose && @info "CX copied"
    copy!(enkf.θ, θ)
    verbose && @info "theta copied"
    copy!(enkf.sys.Cθ.lmap.diag, θ)
    verbose && @info "Getting sys op"

    if enkf.isiterative
        sys_op = enkf.sys
        precond = Diagonal(sys_op)
    else
        sys_mat = bunchkaufman!(Matrix(enkf.sys))
    end
    verbose && @info "Got sys op"

    # Compute Kalman-update in a matrix-free way
    ys_i = ObsConstraintVector(Ny, Nz)
    tmp = zeros(length(ys_i))

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
        yi = observation(ys_i)
        si = constraint(ys_i)

        obs_i = @view perturbed_obs[:, i]
        yi .= obs_i - enkf.sys.H * xi_forecast
        si .= -enkf.sys.S * xi_forecast
        copy!(tmp, ys_i)
        verbose && @info "Start solve"
        if enkf.isiterative
            # Invert sys_op
            cg_out = copy(tmp)
            cg!(cg_out, sys_op, tmp; log=false, verbose=false, reltol=enkf.cg_tol, Pl=precond)
            copy!(ys_i, cg_out)
        else
            ldiv!(sys_mat, tmp)
            copy!(ys_i, tmp)
        end
        # mul!(δi, enkf.sys.H', observation(ys_i))
        # mul!(δi, enkf.sys.S', constraint(ys_i), true, true)
        # mul!(xi_analysis, ĈX_op, δi, true, true)
        δi .= enkf.sys.H' * observation(ys_i) + enkf.sys.S' * constraint(ys_i)
        xi_analysis .= xi_analysis + ĈX_op * δi
    end
end
