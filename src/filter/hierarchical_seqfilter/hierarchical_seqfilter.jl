function (enkf::HierarchicalSeqFilter)(
    X_forecast,
    ystar::Vector{Float64},
    t::Float64,
    verbose::Bool
)
    X_analysis = deepcopy(X_forecast)
    X_forecast_loop = enkf.useEnKIOpt ? X_analysis : X_forecast

    # workspace_sparsity = findall(isnan, enkf.sys.H' * fill(NaN, size(enkf.sys.H, 1)))
    verbose && @info "Getting Covariance..."
    ĈX_op = getĈX(enkf, X_forecast; with_matrix=!enkf.isiterative)

    if enkf.isθshared
        # Initial guess?
        fill!(enkf.θ, enkf.θinit)

        θold = zero(enkf.θ)
        for i = 1:enkf.Niter
            verbose && @info "IAS Optimization i = $i"
            copy!(θold, enkf.θ)
            verbose && @info "θ copied"

            # Update x
            update_x!(enkf, X_forecast_loop, ĈX_op, enkf.θ, ystar, t, X_analysis, verbose)

            verbose && @info "x updated"
            # Update theta
            update_θ!(enkf, X_analysis, enkf.θ, ystar, t, verbose)

            if norm(enkf.θ - θold) / norm(θold) < enkf.rtolθ
                break
            end
        end
    else
        enkf.θ .= rand(enkf.dist, enkf.sys.Ns, enkf.sys.Ne)
        θold = zero(enkf.θ)

        for _ = 1:enkf.Niter
            copy!(θold, enkf.θ)

            # Update theta
            update_θ!(enkf, X_analysis, enkf.θ, ystar, t; verbose)

            # Update x
            update_x!(enkf, X_forecast_loop, ĈX_op, enkf.θ, ystar, t, X_analysis, verbose)

            if norm(enkf.θ - θold) / norm(θold) < enkf.rtolθ
                break
            end
        end
    end
    verbose && @info "Finished optimization loop."
    update_x!(enkf, X_forecast, ĈX_op, enkf.θ, ystar, t, X_analysis, verbose)
    return X_analysis, enkf.θ
end