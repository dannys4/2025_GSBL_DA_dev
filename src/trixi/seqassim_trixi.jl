export seqassim_trixi

# Write the seqassim routine for the LikEnRF with further options to output the dof estimate over time and store previous joint samples

"""
		seqassim_trixi(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, X, Ny, Nx, t0::Float64)

Generic API for sequential data assimilation for any sequential filter of parent type `SeqFilter`.
"""
function seqassim_trixi(
    data::SyntheticData,
    J::Int64,
    ϵx::InflationType,
    algo::SeqFilter,
    X,
    Ny::Int64,
    Nx::Int64,
    t0::Float64,
    sys::TrixiSystem;
    solver = SSPRK43(),
    ode_kwargs...
)

    Ne = size(X, 2)

    statehist = Matrix{Float64}[]
    push!(statehist, deepcopy(X[Ny+1:Ny+Nx, :]))

    if algo isa HierarchicalSeqFilter
        θhist = Vector{Float64}[]
        push!(θhist, algo.θ)
    end

    n0 = ceil(Int64, t0 / algo.Δtobs) + 1
    Acycle = n0:n0+J-1
    tspan = (t0, t0 + algo.Δtobs)

    x_itp = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    # create x_itp for each thread
    x_quad = similar(x_itp)

    prob = semidiscretize(sys.semi, tspan)

    # prints a summary of the simulation setup and resets the timers
    # summary_callback = SummaryCallback()

    # analyse the solution in regular intervals and prints the results
    # analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

    # handles the re-calculation of the maximum Δt after each time step
    stepsize_callback = StepsizeCallback(cfl = 0.2)

    # collect all callbacks such that they can be passed to the ODE solver
    callbacks = CallbackSet(stepsize_callback)

    # Run filtering algorithm
    @showprogress for i = 1:length(Acycle)

        # Forecast
        tspan = (t0 + (i - 1) * algo.Δtobs, t0 + i * algo.Δtobs)

        function prob_func(prob, j, repeat)
            # At this point, the vector x is provided at the Gauss-Legendre nodes
            vec2sol!(x_quad, @view(X[Ny+1:Ny+Nx, j]), sys.equations; g = prim2cons)
            # We need to move them to the Lobatto-Legendre nodes
            get_interp_node_vals!(sys.dg.basis, x_itp, x_quad)

            remake(prob, u0 = x_itp, tspan = tspan)
        end

        ensemble_prob = EnsembleProblem(
            prob,
            output_func = (sol, i) -> (sol[end], false),
            prob_func = prob_func,
        )

        sim = solve(
            ensemble_prob,
            solver,
            adaptive = true,
            EnsembleSerial(),
            trajectories = Ne,
            dense = false,
            save_everystep = false,
            callback = stepsize_callback;
            ode_kwargs...
            # dtmax = (tspan[2] - tspan[1])/100
        )

        @inbounds for i = 1:Ne
            # Interpolate the solution from the solver back to the Gauss-Legendre nodes and reshaping
            # mul!(x_itp, sys.dg.basis.Vq, sim[i])
            get_quadrature_node_vals!(sys.dg.basis, sim[i], x_quad)
            sol2vec!(view(X, Ny+1:Ny+Nx, i), x_quad, sys.equations; g = cons2prim)
        end

        # Assimilation # Get real measurement # Fix this later # Things are shifted in data.yt
        ystar = data.yt[:, Acycle[i]]
        # Replace at some point by realobserve(model.h, t0+i*model.Δtobs, ens)
        # Perform inflation for each ensemble member
        ϵx(X, Ny + 1, Ny + Nx)

        # Compute measurements

        # Generate posterior samples.
        # Note that the additive inflation of the observation is applied within the sequential filter.
        if algo isa HierarchicalSeqFilter
            X, θ = algo(X, ystar, t0 + i * algo.Δtobs - t0)
        else
            X = algo(X, ystar, t0 + i * algo.Δtobs - t0)
        end

        # Filter state
        if algo.isfiltered
            for i = 1:Ne
                statei = view(X, Ny+1:Ny+Nx, i)
                statei .= algo.G(statei)
            end
        end

        push!(statehist, copy(X[Ny+1:Ny+Nx, :]))

        if algo isa HierarchicalSeqFilter
            push!(θhist, copy(θ))
        end
    end
    if algo isa HierarchicalSeqFilter
        return statehist, θhist
    else
        return statehist
    end
end
