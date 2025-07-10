# In this version, there is a jump coefficient θ shared across the eNzemble members

function update_x!(
    enkf::HierarchicalSeqFilter,
    X_forecast,
    ĈX_op,
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

    # Update covariance matrix
    enkf.sys.CX = ĈX_op.lmap

    # Update weight vector θ
    enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64} || throw(ArgumentError("Wrong type for Cθ"))

    enkf.θ .= θ
    enkf.sys.Cθ.lmap.diag .= θ

    if enkf.isiterative
        sys_op = LinearMaps.FunctionMap{Float64,true}(
            (y, x) -> mul!(y, enkf.sys, x),
            Ny + Nz;
            issymmetric=true,
            isposdef=true,
        )
    else
        # @show cond(sys_mat)
        sys_mat = bunchkaufman!(Matrix(enkf.sys))
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
        # @assert isapprox(ys_i.x[1], enkf.sys.H * xi, atol=1e-8)

        yi .+= E[:, i] - ystar

        mul!(si, enkf.sys.S, xi)

        if enkf.isiterative
            tmp.x[1] .= ys_i.x[1]
            tmp.x[2] .= ys_i.x[2]
            # Invert sys_op
            cg!(ys_i, sys_op, tmp; log=false, reltol=1e-3)
        else
            ldiv!(sys_mat, Array(ys_i))
        end
        mul!(δi, enkf.sys.H', observation(ys_i))
        mul!(δi, enkf.sys.S', constraint(ys_i), true, true)
        mul!(xi, ĈX_op, δi, -1, true)
    end
end
