export seqassim_trixi_euler

# Write the seqassim routine for the LikEnRF with further options to output the dof estimate over time and store previous joint samples


"""
		seqassim_trixi(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, X, Ny, Nx, t0::Float64)

Generic API for sequential data assimilation for any sequential filter of parent type `SeqFilter`.
"""
function seqassim_trixi_euler(
    data::SyntheticData,
    J::Int64,
    ϵx::InflationType,
    algo::SeqFilter,
    X,
    Ny::Int64,
    Nx::Int64,
    t0::Float64,
    sys::TrixiSystem,
)

    Ne = size(X, 2)

    step = ceil(Int, algo.Δtobs / algo.Δtdyn)

    statehist = Matrix{Float64}[]
    push!(statehist, deepcopy(X[Ny+1:Ny+Nx, :]))

    if algo isa HierarchicalSeqFilter
        θhist = Vector{Float64}[]
        push!(θhist, algo.θ)
    end


    n0 = ceil(Int64, t0 / algo.Δtobs) + 1
    Acycle = n0:n0+J-1
    tspan = (t0, t0 + algo.Δtobs)

    x_ode = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    tmp = zero(x_ode)

    prob = semidiscretize(sys.semi, tspan)

    # prints a summary of the simulation setup and resets the timers
    # summary_callback = SummaryCallback()

    # analyse the solution in regular intervals and prints the results
    # analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

    # handles the re-calculation of the maximum Δt after each time step
    stepsize_callback = StepsizeCallback(cfl = 0.4)

    # collect all callbacks such that they can be passed to the ODE solver
    callbacks = CallbackSet(stepsize_callback)

    # Run filtering algorithm
    @showprogress for i = 1:length(Acycle)

        # Forecast
        tspan = (t0 + (i - 1) * algo.Δtobs, t0 + i * algo.Δtobs)

        function prob_func(prob, j, repeat)
            # At this point, the vector x is provided at the Gauss-Legendre nodes
            vec2sol!(x_ode, X[Ny+1:Ny+Nx, i], sys.equations)

            # We need to move them to the Lobatto-Legendre nodes
            mul!(x_ode, sys.dg.basis.Pq, x_ode)

            remake(prob, u0 = x_ode, tspan = tspan)
        end

        ensemble_prob = EnsembleProblem(
            prob,
            output_func = (sol, i) -> (sol[end], false),
            prob_func = prob_func,
        )

        sim = solve(
            ensemble_prob,
            SSPRK43(),
            adaptive = true,
            EnsembleThreads(),
            trajectories = Ne,
            dense = false,
            save_everystep = false,
            callback = callbacks,
        )

        @inbounds for i = 1:Ne
            # Interpolate the solution from the solver back to the Gauss-Legendre nodes and reshaping
            mul!(x_ode, sys.dg.basis.Vq, sim[i])
            sol2vec!(view(X, Ny+1:Ny+Nx, i), x_ode, sys.equations)
            # X[Ny+1:Ny+Nx, i] .= vcat(x_ode...)
        end

        # Collect observation from the true system y⋆_t from data.yt. 
        # Note that the indexing is shifted as there are no observations collected at t = 0.
        # The first observation is collected at Δt_obs. 

        ystar = data.yt[:, Acycle[i]]

        # Perform inflation for each ensemble member
        ϵx(X, Ny + 1, Ny + Nx)

        # Generate posterior samples.
        # Note that the observation noise is applied within the sequential filter.
        if algo isa HierarchicalSeqFilter
            X, θ = algo(X, ystar, t0 + i * algo.Δtobs - t0)
        else
            X = algo(X, ystar, t0 + i * algo.Δtobs - t0)
        end

        # Filter state
        if algo.isfiltered == true
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
