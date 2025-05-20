# In this version, there is a jump coefficient θ per ensemble member

function update_x!(
    enkf::HierarchicalSeqFilter,
    X,
    θ::Matrix{Float64},
    ystar::Vector{Float64},
    t,
)

    @assert enkf.isθshared == false

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)
    Ns = enkf.sys.Ns

    @assert size(θ, 1) == Ns
    @assert size(θ, 2) == Ne
    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if enkf.ϵy isa AdditiveInflation
        E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    si = zeros(enkf.sys.Ns)

    ĈX = getĈX(enkf, X, Nx, Ny)

    ĈX_op = FunctionMap{Float64,true}(
        (y, x) -> mul!(y, ĈX, x),
        Nx;
        issymmetric = true,
        isposdef = false,
    )

    # Update covariance matrix
    enkf.sys.CX[1] = ĈX_op


    for i = 1:Ne
        # Update weight vector θ
        if typeof(enkf.sys.Cθ) <: LinearMaps.LinearMaps.WrappedMap{Float64}
            enkf.θ[:, i] .= θ[:, i]
            enkf.sys.Cθ.lmap.diag .= θ[:, i]
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
                sys_mat[:, i] .= sys_op * ei
            end

            sys_mat = factorize(Symmetric(sys_mat))
        end

        # Compute Kalman-update in a matrix-free way

        ys_i = ObsConstraintVector(Ny, Ns)
        tmp = ObsConstraintVector(Ny, Ns)

        δi = zeros(Nx)

        xi = view(X, Ny+1:Ny+Nx, i)
        yi = observation(ys_i)
        si = constraint(ys_i)

        mul!(yi, enkf.sys.H, xi)
        @assert isapprox(ys_i.x[1], enkf.sys.H * xi, atol = 1e-8)

        yi .+= E[:, i] - ystar

        mul!(si, enkf.sys.S, xi)

        if enkf.isiterative == false
            # Invert sys_op
            ldiv!(ys_i, sys_mat, ys_i)
        else
            cg!(ys_i, sys_op, copy(ys_i); log = false, reltol = 1e-3)
        end

        δi .= enkf.sys.H' * observation(ys_i)
        δi .+= enkf.sys.S' * constraint(ys_i)

        xi .+= -(ĈX * δi)
    end
end
