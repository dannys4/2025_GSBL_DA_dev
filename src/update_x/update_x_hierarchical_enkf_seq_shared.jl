# In this version, there is a jump coefficient θ shared across the ensemble members, 
# and we first assimilate the observations before to assimilate the regularization term

function update_x!(
    enkf::Union{HEnKF,HLocEnKF},
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
    Ns = enkf.sys.Ns

    @assert size(θ, 1) == Ns
    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if typeof(enkf.ϵy) <: AdditiveInflation
        E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    si = zeros(enkf.sys.Ns)

    if typeof(enkf) <: HEnKF
        ĈX = EmpiricalCov(X[Ny+1:Ny+Nx, :]; with_matrix = true)
    elseif typeof(enkf) <: HLocEnKF
        ĈX = LocalizedEmpiricalCov(X[Ny+1:Ny+Nx, :], enkf.Loc; with_matrix = true)
    end

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
        Ny + Ns;
        issymmetric = true,
        isposdef = true,
    )

    if enkf.isiterative == false
        sys_mat = zeros(Ny + Ns, Ny + Ns)

        ei = zeros(Ny + Ns)
        for i = 1:Ny+Ns
            fill!(ei, 0.0)
            ei[i] = 1.0
            sys_mat[:, i] = sys_op * ei
        end

        # @show cond(sys_mat)
        sys_mat = factorize(Symmetric(sys_mat))
    end


    # Compute Kalman-update in a matrix-free way

    ys_i = ObsConstraintVector(Ny, Ns)
    tmp = ObsConstraintVector(Ny, Ns)

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
            # ldiv!(ys_i, sys_mat, ys_i)
            # @show typeof(ys_i)
            # @show cg(sys_op, ys_i; log = true)[2]/
            # @show cg(sys_op, tmp; log = true, reltol = 1e-3)
            cg!(ys_i, sys_op, tmp; log = false, reltol = 1e-3)

        end

        δi .= enkf.sys.H' * observation(ys_i)
        δi .+= enkf.sys.S' * constraint(ys_i)

        xi .+= -(ĈX * δi)
    end
end
