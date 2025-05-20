using LinearAlgebra
# In this version, there is a jump coefficient θ shared across the ensemble members, 
# and we first assimilate the observations before to assimilate the regularization term

function update_x!(
    enkf::HierarchicalSeqFilter,
    X,
    θ::Vector{Float64},
    ystar::Vector{Float64},
    t,
)

    @assert enkf.isθshared

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)
    Ne = size(X, 2)
    Ns = enkf.sys.Ns

    @assert size(θ, 1) == Ns
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
        issymmetric=true,
        isposdef=false,
    )

    # Update covariance matrix
    enkf.sys.CX[1] = ĈX_op

    if !(enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64})
        throw(ArgumentError("Wrong type for Cθ"))
    end

    # Update weight vector θ
    copy!(enkf.θ, θ)
    copy!(enkf.sys.Cθ.lmap.diag, θ)

    # enkf.sys is an ObservationConstraintSystem
    # This is then the application of the covariance of (y,θ)|x
    # i.e., the matrix inverted in kalman gain
    sys_op = LinearMaps.FunctionMap{Float64,true}(
        (y, x) -> mul!(y, enkf.sys, x),
        Ny + Ns;
        issymmetric=true,
        isposdef=true,
    )

    if !enkf.isiterative
        sys_mat = zeros(Ny + Ns, Ny + Ns)

        ei = zeros(Ny + Ns)
        for i = 1:Ny+Ns
            ei[i] = 1.0
            sys_mat[:, i] = sys_op * ei
            ei[i] = 0.0
        end
        sys_mat = factorize(Hermitian(sys_mat))
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
        @assert isapprox(ys_i.x[1], enkf.sys.H * xi, atol=1e-8)

        # Difference of *noised* forecast observation and actual observation
        yi .+= E[:, i] - ystar

        # PA operator
        mul!(si, enkf.sys.S, xi)

        # Observation
        tmp.x[1] .= ys_i.x[1]
        # Constraint
        tmp.x[2] .= ys_i.x[2]

        if enkf.isiterative
            # Invert sys_op
            cg!(ys_i, sys_op, tmp; log=false, reltol=1e-3)
        else
            ys_i .= sys_mat \ tmp
        end

        δi .= enkf.sys.H' * observation(ys_i)
        δi .+= enkf.sys.S' * constraint(ys_i)

        xi .+= -(ĈX * δi)
    end
end

"""
    matmuladd!(C,A,B, α)
Calculate \$ A \\times B\$ and add it to C inplace
"""
matmuladd!(C,A,B) = mul!(C, A, B, true, true)

function update_x_fixed_θ_laplace!(
    enkf::HierarchicalSeqFilter,
    X,
    θ:Vector{Float64},
    ystar::Vector{Float64},
    t
)

    @assert enkf.isθshared && enkf.isStateStochastic

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)
    Ne = size(X, 2)
    Ns = enkf.sys.Ns

    @assert size(θ, 1) == Ns
    @assert size(ystar, 1) == Ny

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if enkf.ϵy isa AdditiveInflation
        E .= enkf.ϵy.σ * randn(Ny, Ne) .+ enkf.ϵy.m
    end

    ĈX = getĈX(enkf, X, Nx, Ny)

    ĈX_op = FunctionMap{Float64,true}(
        (y, x) -> mul!(y, ĈX, x),
        Nx;
        issymmetric=true,
        isposdef=false,
    )

    # Update covariance matrix
    enkf.sys.CX[1] = ĈX_op

    if !(enkf.sys.Cθ isa LinearMaps.LinearMaps.WrappedMap{Float64})
        throw(ArgumentError("Wrong type for Cθ"))
    end

    # Update weight vector θ
    copy!(enkf.θ, θ)
    copy!(enkf.sys.Cθ.lmap.diag, θ)

    # enkf.sys is an ObservationConstraintSystem
    # This is then the application of the covariance of (y,θ)|x
    # i.e., the matrix inverted in kalman gain
    sys_op = LinearMaps.FunctionMap{Float64,true}(
        (y, x) -> mul!(y, enkf.sys, x),
        Ny + Ns;
        issymmetric=true,
        isposdef=true,
    )

    @assert !enkf.isiterative "Stochastic GSBL laplace only supports full solving"
    if !enkf.isiterative
        sys_mat = zeros(Ny + Ns, Ny + Ns)

        ei = zeros(Ny + Ns)
        for i = 1:Ny+Ns
            ei[i] = 1.0
            sys_mat[:, i] = sys_op * ei
            ei[i] = 0.0
        end
        sys_mat = factorize(Hermitian(sys_mat))
    end


    # Get CX Matrix
    # TODO: Inefficient placeholder for now
    CX_mat = zeros(Nx)
    ei = zeros(Nx)
    for i in 1:Nx
        ei[i] = 1.0
        mul!(view(CX_mat,:,i), ĈX_op, ei)
        ei[i] = 0.0
    end
    # TODO: Add inflation?
    Herm_CX_mat = Hermitian(CX_mat)
    sqrt_CX = Hermitian(sqrt(Herm_CX_mat))
    Theta_half = Diagonal(sqrt.(θ))
    
    # Compute Kalman-update in a matrix-free way
    ys_i = ObsConstraintVector(Ny, Ns)
    tmp = ObsConstraintVector(Ny, Ns)
    pre_kalman = zeros(Nx)
    pre_C_hat_mul = zeros(Nx)

    for i = 1:Ne
        xi = view(X, Ny+1:Ny+Nx, i)
        yi = observation(ys_i)
        si = constraint(ys_i)
        tmp_yi = observation(tmp)
        tmp_si = constraint(tmp)

        # Careful about aliasing!!
        state_noise, theta_noise = pre_C_hat_mul, tmp_si # z_1, z_3 aliases for now
        randn!(state_noise)
        randn!(theta_noise)

        E[:, i] .+= ystar
        ldiv!(enkf.ϵy.Σ, @view(E[:,i])) # Calculate Gamma^{-1} (y^* + Gamma^{1/2} z_2)
        ldiv!(Theta_half, theta_noise) # Calculate Theta^{-1/2} z_3. Done with z_3 alias

        # xi is storage for v2
        matmuladd!(xi, sqrt_CX, state_noise) # Done with z_1 alias
        
        # pre_C_hat_mul is $v_2$ in math, where we just used it as a temporary memory space above
        mul!(pre_C_hat_mul, enkf.sys.H', @view(E[:,i]))
        matmuladd!(pre_C_hat_mul, enkf.sys.S', theta_noise) # v_2 = H^T Gamma^{-1} (y^* + Gamma^{1/2} z_2) + S^T Theta^{-1/2} z_3

        # v2 = xi + \widetilde{C}z_1 + \widehat{C} v1
        matmuladd!(xi, Herm_CX_mat, pre_C_hat_mul)
        # Keep v2 around unchanged! v1 can be changed, though
        mul!(tmp_yi, enkf.sys.H, xi)
        mul!(tmp_si, enkf.sys.S, xi)
        ys_i .= sys_mat \ tmp
        
        mul!(pre_C_hat_mul, enkf.sys.H', yi)
        matmuladd!(pre_C_hat_mul, enkf.sys.S', si)
        
        # Analysis: x_a = v2 - \widehat{C}H^T A^{-1} H v2
        mul!(xi, Herm_CX_mat, pre_C_hat_mul, -1, true)
    end
end