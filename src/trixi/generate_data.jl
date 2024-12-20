export generate_data_trixi

function generate_data_trixi(model::Model, x0, J::Int64, sys::TrixiSystem)

    @assert model.Nx == size(x0, 1) "Error dimension of the input"
    xt = zeros(model.Nx, J)

    x = deepcopy(x0)
    x_ode = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)

    yt = zeros(model.Ny, J)
    tt = zeros(J)

    t0 = 0.0

    step = ceil(Int, model.Δtobs / model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)

    prob = semidiscretize(sys.semi, tspan)

    # prints a summary of the simulation setup and resets the timers
    # summary_callback = SummaryCallback()

    # analyse the solution in regular intervals and prints the results
    # analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

    # handles the re-calculation of the maximum Δt after each time step
    stepsize_callback = StepsizeCallback(cfl = 0.1)

    # collect all callbacks such that they can be passed to the ODE solver
    callbacks = CallbackSet(stepsize_callback)


    @showprogress for i = 1:J
        # Run dynamics and save results
        tspan = (t0 + (i - 1) * model.Δtobs, t0 + i * model.Δtobs)

        # At this point, the vector x is provided at the Gauss-Legendre nodes
        vec2sol!(x_ode, x, sys.equations; g = prim2cons)
        # for i in eachindex(x_ode)
        #     x_ode[i] = SVector(x[i])
        # end
        # We need to move them to the Lobatto-Legendre nodes
        mul!(x_ode, sys.dg.basis.Pq, x_ode)

        prob = remake(prob, u0 = deepcopy(x_ode), tspan = tspan)

        sol = solve(
            prob,
            SSPRK43(),
            # dt = model.Δtdyn,
            adaptive = true,
            dense = false,
            save_everystep = false,
            callback = callbacks,
        )

        # Interpolate the solution from the solver back to the Gauss-Legendre nodes and reshaping
        mul!(x_ode, sys.dg.basis.Vq, sol.u[end])

        sol2vec!(x, x_ode, sys.equations; g = cons2prim)

        # x .= vcat(x_ode...)

        model.ϵx(x)

        # Collect observations
        tt[i] = deepcopy(i * model.Δtobs)
        xt[:, i] = deepcopy(x)
        yt[:, i] = deepcopy(model.F.h(x, tt[i]))

        if typeof(model.ϵy) <: AdditiveInflation
            yt[:, i] .+= model.ϵy.m + model.ϵy.σ * randn(model.Ny)
        end
    end

    return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end
