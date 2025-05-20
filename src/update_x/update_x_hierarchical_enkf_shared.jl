# In this version, there is a jump coefficient θ shared across the eNzemble members

function update_x!(
    enkf::HierarchicalSeqFilter,
    X,
    θ::Vector{Float64},
    ystar::Vector{Float64},
    t,
)

    @assert enkf.isθshared == true

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)
    Ne = size(X, 2)
    Nz = enkf.sys.Nz

    @assert size(θ, 1) == Nz
    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if enkf.ϵy isa AdditiveInflation
        E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    si = zeros(enkf.sys.Nz)

    ĈX = getĈX(enkf, X, Nx, Ny)

    ĈX_op = FunctionMap{Float64,true}(
        (y, x) -> mul!(y, ĈX, x),
        Nx;
        issymmetric = true,
        isposdef = false,
    )

    # Update covariance matrix
    enkf.sys.CX[1] = ĈX_op

    # Update weight vector θ
    if typeof(enkf.sys.Cθ) <: LinearMaps.LinearMaps.WrappedMap{Float64}
        enkf.θ .= θ
        enkf.sys.Cθ.lmap.diag .= θ
    else
        error("Wrong type for Cθ")
    end

    sys_op = LinearMaps.FunctionMap{Float64,true}(
        (y, x) -> mul!(y, enkf.sys, x),
        Ny + Nz;
        issymmetric = true,
        isposdef = true,
    )

    if enkf.isiterative == false
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


    for i = 1:Ne
        xi = view(X, Ny+1:Ny+Nx, i)
        yi = observation(ys_i)
        si = constraint(ys_i)

        mul!(yi, enkf.sys.H, xi)
        @assert isapprox(ys_i.x[1], enkf.sys.H * xi, atol = 1e-8)

        yi .+= E[:, i] - ystar

        mul!(si, enkf.sys.S, xi)

        tmp.x[1] .= ys_i.x[1]
        tmp.x[2] .= ys_i.x[2]

        if enkf.isiterative == false
            ys_i .= sys_mat \ tmp
        else

            # Invert sys_op
            cg!(ys_i, sys_op, tmp; log = false, reltol = 1e-3)

        end

        δi .= enkf.sys.H' * observation(ys_i)
        δi .+= enkf.sys.S' * constraint(ys_i)

        xi .+= -(ĈX * δi)
    end
end
