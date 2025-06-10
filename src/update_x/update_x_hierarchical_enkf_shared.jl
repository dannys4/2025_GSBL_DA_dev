# In this version, there is a jump coefficient θ shared across the eNzemble members

function update_x!(
    enkf::HierarchicalSeqFilter,
    X_forecast,
    θ::Vector{Float64},
    ystar::Vector{Float64},
    t,
    X_analysis
)
    # @assert X_forecast !== X_analysis
    @assert enkf.isθshared

    Ny = size(ystar, 1)
    Nx = size(X_forecast, 1) - Ny
    Ne = size(X_forecast, 2)
    Ne = size(X_forecast, 2)
    Nz = enkf.sys.Nz

    @assert size(θ, 1) == Nz
    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if enkf.ϵy isa AdditiveInflation
        E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    si = zeros(enkf.sys.Nz)

    ĈX = getĈX(enkf, X_forecast, Nx, Ny)

    ĈX_op = FunctionMap{Float64,true}(
        (y, x) -> mul!(y, ĈX, x),
        Nx;
        issymmetric = true,
        isposdef = false,
    )

    # Update covariance matrix
    enkf.sys.CX[1] = ĈX_op

    # Update weight vector θ
    enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64} || throw(ArgumentError("Wrong type for Cθ"))

    enkf.θ .= θ
    enkf.sys.Cθ.lmap.diag .= θ

    sys_op = LinearMaps.FunctionMap{Float64,true}(
        (y, x) -> mul!(y, enkf.sys, x),
        Ny + Nz;
        issymmetric = true,
        isposdef = true,
    )

    if !enkf.isiterative
        sys_mat = zeros(Ny + Nz, Ny + Nz)

        ei = zeros(Ny + Nz)
        for i = 1:Ny+Nz
            fill!(ei, 0.0)
            ei[i] = 1.0
            sys_mat[:, i] = sys_op * ei
        end

        # @show cond(sys_mat)
        sys_mat = factorize(Symmetric(sys_mat))
    end


    # Compute Kalman-update in a matrix-free way
    ys_i = ObsConstraintVector(Ny, Nz)
    tmp = ObsConstraintVector(Ny, Nz)

    δi = zeros(Nx)

    if X_analysis !== X_forecast
        copy!(view(X_analysis, Ny+1:Ny+Nx, :), view(X_forecast, Ny+1:Ny+Nx, :))
    end

    for i = 1:Ne
        xi = view(X_analysis, Ny+1:Ny+Nx, i)
        yi = observation(ys_i)
        si = constraint(ys_i)

        mul!(yi, enkf.sys.H, xi)
        @assert isapprox(ys_i.x[1], enkf.sys.H * xi, atol = 1e-8)

        yi .+= E[:, i] - ystar

        mul!(si, enkf.sys.S, xi)

        tmp.x[1] .= ys_i.x[1]
        tmp.x[2] .= ys_i.x[2]
        if enkf.isiterative
            # Invert sys_op
            cg!(ys_i, sys_op, tmp; log = false, reltol = 1e-3)
        else
            ys_i .= sys_mat \ tmp
        end
        mul!(δi, enkf.sys.H', observation(ys_i))
        mul!(δi, enkf.sys.S', constraint(ys_i), true, true)
        mul!(xi, ĈX, δi, -1, true)
    end
end
