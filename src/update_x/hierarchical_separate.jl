# In this version, there is a jump coefficient θ per ensemble member

function update_x!(
    enkf::HierarchicalSeqFilter,
    X_forecast,
    θ::Matrix{Float64},
    perturbed_obs::Matrix{Float64},
    t,
    X_analysis,
    verbose::Bool
)
    @assert !enkf.isθshared

    Ny = size(ystar, 1)
    Nx = size(X_forecast, 1) - Ny
    Ne = size(X_forecast, 2)
    Ns = enkf.sys.Ns

    @assert size(θ, 1) == Ns
    @assert size(θ, 2) == Ne
    @assert size(perturbed_obs, 1) == Ny
    @assert size(perturbed_obs, 2) == Ne

    # Generate observational noise samples
    # E = zeros(Ny, Ne)
    # if enkf.ϵy isa AdditiveInflation
    #     E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    # end

    si = zeros(enkf.sys.Ns)

    ĈX = getĈX(enkf, X_forecast, Nx, Ny)

    ĈX_op = FunctionMap{Float64,true}(
        (y, x) -> mul!(y, ĈX, x),
        Nx;
        issymmetric=true,
        isposdef=false,
    )

    # Update covariance matrix
    enkf.sys.CX = ĈX_op

    enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64} || throw(ArgumentError("Wrong type of theta!"))

    sys_op = LinearMaps.FunctionMap{Float64,true}(
        (y, x) -> mul!(y, enkf.sys, x),
        Ny + Ns;
        issymmetric=true,
        isposdef=true,
    )

    if X_forecast !== X_analysis
        copy!(X_analysis, X_forecast)
    end

    if !enkf.isiterative
        sys_mat = factorize(Symmetric(Matrix(sys_op)))
    end


    for i = 1:Ne
        # Compute Kalman-update in a matrix-free way

        ys_i = ObsConstraintVector(Ny, Ns)
        tmp = ObsConstraintVector(Ny, Ns)

        δi = zeros(Nx)

        xi = @view X_analysis[:, i]
        obs_i = @view perturbed_obs[:, i]
        yi = observation(ys_i)
        si = constraint(ys_i)

        copy!(yi, obs_i)
        mul!(yi, enkf.sys.H, xi, -1, true)
        # @assert isapprox(ys_i.x[1], enkf.sys.H * xi, atol=1e-8)

        mul!(si, enkf.sys.S, xi)

        if enkf.isiterative
            # Invert sys_op
            cg!(ys_i, sys_op, copy(ys_i); log=false, reltol=enkf.cg_tol)
        else
            ldiv!(sys_mat, ys_i)
        end

        δi .= enkf.sys.H' * observation(ys_i)
        δi .+= enkf.sys.S' * constraint(ys_i)

        xi .+= (ĈX * δi)
    end
end
