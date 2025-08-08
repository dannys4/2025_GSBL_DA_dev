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
    ode_solver=SSPRK43(),
    cfl=0.2,
    store_state_path=nothing,
    verbose=false,
    ode_kwargs...
)
    Ne = size(X, 2)
    Δtobs = algo.Δtobs
    statehist = Matrix{Float64}[]
    push!(statehist, copy(X))

    if algo isa HierarchicalSeqFilter
        θhist = Vector{Float64}[]
        push!(θhist, algo.θ)
    end

    n0 = ceil(Int64, t0 / Δtobs) + 1
    Acycle = n0:n0+J-1
    tspan = (t0, t0 + Δtobs)

    x_itp = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    x_quad = similar(x_itp)

    prob = semidiscretize(sys.semi, tspan)

    # prints a summary of the simulation setup and resets the timers
    # summary_callback = SummaryCallback()

    # analyse the solution in regular intervals and prints the results
    # analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

    # handles the re-calculation of the maximum Δt after each time step
    stepsize_callback = StepsizeCallback(; cfl)

    # collect all callbacks such that they can be passed to the ODE solver
    callbacks = CallbackSet(stepsize_callback)

    # Check if we assimilate the initial state
    assimilate_first = size(data.yt, 2) == length(data.tt) + 1
    # if assimilate_first
    #     ystar = @view(data.yt[:, 1])
    #     if algo isa HierarchicalSeqFilter
    #         X, θ = algo(X, ystar, 0.)
    #     else
    #         X = algo(X, ystar, 0.)
    #     end


    #     # Filter state
    #     if algo.isfiltered
    #         for i = 1:Ne
    #             statei = @view X[:, i]
    #             statei .= algo.G(statei)
    #         end
    #     end
    #     if isnothing(store_state_path)
    #         push!(statehist, copy(X))
    #     else
    #         @save joinpath(store_state_path, "ens0_$(now()).jld2") X
    #     end

    #     if algo isa HierarchicalSeqFilter
    #         push!(θhist, copy(θ))
    #     end
    # end
    output_func = (sol, i) -> (sol[end], false)
    # Run filtering algorithm
    J > 0 && @showprogress "Filtering using $(typeof(algo))..." for i = 1:length(Acycle)

        # Forecast
        tspan = (t0 + (i - 1) * Δtobs, t0 + i * Δtobs)
        function prob_func(prob, j, repeat)
            # At this point, the vector x is provided at the Gauss-Legendre nodes
            vec2sol!(x_quad, @view(X[:, j]), sys.equations)
            # We need to move them to the Lobatto-Legendre nodes
            get_interp_node_vals!(sys.dg, x_quad, x_itp)
            remake(prob, u0=x_itp, tspan=tspan)
        end

        ensemble_prob = EnsembleProblem(prob; output_func, prob_func=prob_func)

        sim = solve(
            ensemble_prob,
            ode_solver,
            adaptive=true,
            EnsembleSerial(),
            trajectories=Ne,
            dense=false,
            save_everystep=false,
            callback=stepsize_callback;
            ode_kwargs...
        )

        @inbounds for i = 1:Ne
            # Interpolate the solution from the solver back to the Gauss-Legendre nodes and reshaping
            get_quadrature_node_vals!(sys.dg, x_quad, sim[i])
            sol2vec!(@view(X[:, i]), x_quad, sys.equations; g=cons2prim)
        end

        # Assimilation # Get real measurement # Fix this later # Things are shifted in data.yt
        ystar = data.yt[:, Acycle[i]+assimilate_first]
        # Replace at some point by realobserve(model.h, t0+i*model.Δtobs, ens)
        # Perform inflation for each ensemble member
        ϵx(X, 1, Nx)

        # Compute measurements

        # Generate posterior samples.
        # Note that the additive inflation of the observation is applied within the sequential filter.
        if algo isa HierarchicalSeqFilter
            X, θ = algo(X, ystar, i * algo.Δtobs, verbose)
        else
            X = algo(X, ystar, i * algo.Δtobs, verbose)
        end

        # Filter state
        if algo.isfiltered
            for i = 1:Ne
                statei = @view X[:, i]
                statei .= algo.G(statei)
            end
        end

        if isnothing(store_state_path)
            push!(statehist, copy(X))
        else
            @save joinpath(store_state_path, "ens$(i)_$(now()).jld2") X
        end

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
