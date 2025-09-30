export generate_data_trixi

function generate_data_trixi(model::Model, x0, Tf::Int64, sys::TrixiSystem; ode_solver=SSPRK43(), cfl=0.2, record_first=false, ode_kwargs...)

    @assert model.Nx == size(x0, 1) "Error dimension of the input"
    xt = zeros(model.Nx, Tf)

    x = deepcopy(x0)
    # First is for the interpolation points, second is for the quadrature points
    x_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    x_itp = similar(x_quad)

    yt = zeros(model.Ny, Tf + record_first)
    tt = zeros(Tf)

    t0 = 0.0

    # step = ceil(Int, model.Δtobs / model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)

    prob = semidiscretize(sys.semi, tspan)

    # prints a summary of the simulation setup and resets the timers
    # summary_callback = SummaryCallback()

    # analyse the solution in regular intervals and prints the results
    # analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

    # handles the re-calculation of the maximum Δt after each time step
    stepsize_callback = StepsizeCallback(; cfl)

    # collect all callbacks such that they can be passed to the ODE solver
    # callbacks = CallbackSet(stepsize_callback)

    if record_first
        yt[:, 1] .= model.F.h(x0, 0.)
        if model.ϵy isa AdditiveInflation
            if has_nonzero_mean(model.ϵy)
                # yt[:, 1] .+= model.ϵy.m + model.ϵy.σ * randn(model.Ny)
                mul!(@view(y[:, 1]), model.ϵy.σ, randn(model.Ny))
            end
        end
    end

    @showprogress for time_idx = 1:Tf
        # Run dynamics and save results
        tspan = (t0 + (time_idx - 1) * model.Δtobs, t0 + time_idx * model.Δtobs)

        # At this point, the vector x is provided at the Gauss-Legendre nodes
        vec2sol!(x_quad, x, sys.equations; g=prim2cons)
        # x_quad exists on quadrature nodes. Move to itp points
        get_interp_node_vals!(sys.dg, x_quad, x_itp)

        prob = remake(prob, u0=x_itp, tspan=tspan)
        sol = solve(
            prob,
            ode_solver;
            # dt = model.Δtdyn,
            adaptive=true,
            dense=false,
            save_everystep=false,
            callback=stepsize_callback,
            ode_kwargs...
        )

        # Interpolate the solution from the solver back to the quadrature nodes and reshaping
        get_quadrature_node_vals!(sys.dg, x_quad, sol.u[end])
        sol2vec!(x, x_quad, sys.equations; g=cons2prim)

        model.ϵx(x)

        # Collect observations
        tt[time_idx] = time_idx * model.Δtobs
        copy!(@view(xt[:, time_idx]), x)
        copy!(@view(yt[:, time_idx+record_first]), model.F.h(x, tt[time_idx]))

        if model.ϵy isa AdditiveInflation
            if has_nonzero_mean(model.ϵy)
                y[:, time_idx+record_first] .+= model.ϵy.m
            end
            mul!(@view(yt[:, time_idx+record_first]), model.ϵy.σ, randn(model.Ny), true, true)
        end
    end
    return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end
