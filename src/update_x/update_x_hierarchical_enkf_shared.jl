# In this version, there is a jump coefficient θ shared across the eNzemble members

function update_x!(
    enkf::HierarchicalSeqFilter,
    X_forecast,
    ĈX_op,
    θ::Vector{Float64},
    ystar::Vector{Float64},
    t,
    X_analysis,
    verbose::Bool
)
    @assert enkf.isθshared

    # Update weight vector θ
    if !(enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64})
        ArgumentError("Wrong type for Cθ")
    end

    Ny = size(ystar, 1)
    Nx = size(X_forecast, 1)
    Ne = size(X_forecast, 2)
    Ne = size(X_forecast, 2)
    Nz = enkf.sys.Nz

    @assert size(θ, 1) == Nz
    @assert size(ystar, 1) == Ny

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
    end

    # Update covariance matrix
    enkf.sys.CX = ĈX_op.lmap
    copy!(enkf.θ, θ)
    copy!(enkf.sys.Cθ.lmap.diag, θ)

    if enkf.isiterative
        sys_op = LinearMaps.FunctionMap{Float64,true}(
            (y, x) -> mul!(y, enkf.sys, x),
            Ny + Nz;
            issymmetric=true,
            isposdef=true,
        )
    else
        sys_mat = bunchkaufman!(Matrix(enkf.sys))
    end


    # Compute Kalman-update in a matrix-free way
    ys_i = ObsConstraintVector(Ny, Nz)
    tmp = zeros(length(ys_i))

    δi = zeros(Nx)

    if X_analysis !== X_forecast
        copy!(X_analysis, X_forecast)
    end

    for i = 1:Ne
        fill!(ys_i, zero(eltype(ys_i)))
        fill!(δi, zero(eltype(δi)))
        xi_forecast = @view X_forecast[:, i]
        xi_analysis = @view X_analysis[:, i]
        yi = observation(ys_i)
        si = constraint(ys_i)

        err_i = @view errs[:, i]
        # copy!(yi, err_i)

        # mul!(yi, enkf.sys.H, xi_forecast, -1, true)
        # mul!(si, enkf.sys.S, xi_forecast, -1, false)
        yi .= err_i - enkf.sys.H * xi_forecast
        si .= -enkf.sys.S * xi_forecast
        copy!(tmp, ys_i)
        if enkf.isiterative
            # Invert sys_op
            cg!(ys_i, sys_op, tmp; log=false, reltol=1e-3)
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
