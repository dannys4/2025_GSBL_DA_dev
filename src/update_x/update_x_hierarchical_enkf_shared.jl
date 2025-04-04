# In this version, there is a jump coefficient θ shared across the eNzemble members

function update_x!(
    enkf::Union{HEnKF,HLocEnKF},
    X,
    θ::Vector{Float64},
    ystar::Vector{Float64},
    t,
)
    @assert enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64}
    @assert enkf.isθshared

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

    ĈX = getĈX(enkf, X, Ny, Nx)

    ĈX_op = FunctionMap{Float64,true}(
        (y, x) -> mul!(y, ĈX, x),
        Nx;
        issymmetric = true,
        isposdef = false,
    )

    # Update covariance matrix
    enkf.sys.CX[1] = ĈX_op

    # Update weight vector θ
    enkf.θ .= θ
    enkf.sys.Cθ.lmap.diag .= θ

    sys_op = LinearMaps.FunctionMap{Float64,true}(
        (y, x) -> mul!(y, enkf.sys, x),
        Ny + Nz;
        issymmetric = true,
        isposdef = true,
    )

    # if !enkf.isiterative
    #     # sys_mat = zeros(Ny + Nz, Ny + Nz)

    #     # ei = zeros(Ny + Nz)
    #     # for i = 1:Ny+Nz
    #     #     ei[i] = 1.0
    #     #     sys_mat[:, i] = sys_op * ei
    #     #     ei[i] = 0.0
    #     # end

    #     sys_mat = factorize(Hermitian(Matrix(sys_op)))
    # end

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

        if !enkf.isiterative
            ys_i .= sys_mat \ tmp
        else
            # Invert sys_op
            cg!(ys_i, sys_op, tmp; log = false, reltol = 1e-3)
        end
        δi .= enkf.sys.H' * observation(ys_i)
        δi .+= enkf.sys.S' * constraint(ys_i)
        xi .+= -(ĈX * δi)
        # mul!(δi, enkf.sys.H', observation(ys_i), true, false)
        # C <- S^T * ys_i + C
        # mul!(δi, enkf.sys.S', constraint(ys_i), true, true)
        # xi <- xi -(ĈX * δi)
        # mul!(xi, ĈX, δi, -1, true)
        # xi .+= -(ĈX * δi)
        # δi .= 0
    end
end
