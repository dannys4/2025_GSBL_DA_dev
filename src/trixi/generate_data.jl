export generate_data_trixi

function generate_timestep!(u_quad, u_itp, u, prob, tspan, sys, ode_solver; ode_transforms = nothing, ode_kwargs...)
    # Run dynamics and save results

    to_solver_transform = isnothing(ode_transforms) ? identity : ode_transforms.to_solver_transform
    from_solver_transform = isnothing(ode_transforms) ? identity : ode_transforms.from_solver_transform

    # At this point, the vector x is provided at the Gauss-Legendre nodes
    vec2sol!(u_quad, u, sys.equations; g=(x,eqns)->prim2cons(to_solver_transform(x), eqns))
    # x_quad exists on quadrature nodes. Move to itp points
    get_interp_node_vals!(sys.dg, u_quad, u_itp)

    prob = remake(prob, u0=u_itp, tspan=tspan)
    sol = solve(
        prob,
        ode_solver;
        ode_kwargs...
    )

    # Interpolate the solution from the solver back to the quadrature nodes and reshaping
    get_quadrature_node_vals!(sys.dg, u_quad, sol.u[end])
    sol2vec!(u, u_quad, sys.equations; g=from_solver_transform ∘ cons2prim)
end

function generate_data_trixi(model::Model, u0, Tf::Int64, sys::TrixiSystem; (true_soln!)=nothing, ode_solver=SSPRK43(), cfl=0.2, ode_kwargs...)
    @assert model.Nx == size(u0, 1) "Error dimension of the input"
    ut = zeros(model.Nx, Tf)

    u = deepcopy(u0)
    # First is for the interpolation points, second is for the quadrature points
    u_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    u_itp = similar(u_quad)

    bt = zeros(model.Ny, Tf)
    tt = zeros(Tf)

    t0 = 0.0

    # step = ceil(Int, model.Δtobs / model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)

    prob = semidiscretize(sys.semi, tspan)
    xgrid = vec(sys.mesh.md.xq)

    # handles the re-calculation of the maximum Δt after each time step
    stepsize_callback = StepsizeCallback(; cfl)

    # collect all callbacks such that they can be passed to the ODE solver
    @showprogress for time_idx = 1:Tf
        next_t = t0 + time_idx * model.Δtobs
        if isnothing(true_soln!)
            tspan = (t0 + (time_idx - 1) * model.Δtobs, next_t)
            generate_timestep!(u_quad, u_itp, u, prob, tspan, sys, ode_solver;
                adaptive=true,
                dense=false,
                save_everystep=false,
                callback=stepsize_callback,
                ode_kwargs...
            )
        else
            true_soln!(u, xgrid, next_t)
        end

        model.ϵx(u, next_t)

        # Collect observations
        tt[time_idx] = time_idx * model.Δtobs
        copy!(@view(ut[:, time_idx]), u)
        copy!(@view(bt[:, time_idx]), model.F.h(u, tt[time_idx]))
        if model.ϵy isa RelativeAdditiveInflation
            model.ϵy.times[time_idx] = tt[time_idx]
            copy!(@view(model.ϵy.offsets[:,time_idx]), @view(bt[:, time_idx]))
        end
        model.ϵy(@view(bt[:, time_idx]), tt[time_idx])
    end
    return SyntheticData(tt, model.Δtdyn, u0, ut, bt)
end