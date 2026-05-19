function getĈX_op(enkf::HierarchicalSeqFilter, X::AbstractMatrix; kwargs...)
    ĈX = getĈX(enkf, X; kwargs...)
    ĈX_mat = Matrix(ĈX)
    if isnothing(ĈX_mat)
        return ĈX
    else
        return LinearMap(ĈX_mat)
    end
end

isθshared(enkf::HierarchicalSeqFilter) = enkf.θ isa Vector{Float64}

function (enkf::HierarchicalSeqFilter)(
    X_forecast,
    ystar::Vector{Float64},
    t::Float64,
    verbose::Bool
)
    Ny = length(ystar)
    Ne = size(X_forecast, 2)

    X_analysis = deepcopy(X_forecast)
    X_forecast_loop = enkf.useEnKIOpt ? X_analysis : X_forecast

    perturbed_obs = enkf.obs_workspace
    @assert size(perturbed_obs) == (Ny, Ne)
    perturbed_obs .= ystar
    # Generate observational noise samples
    enkf.ϵy(perturbed_obs, t, true)

    # workspace_sparsity = findall(isnan, enkf.sys.H' * fill(NaN, size(enkf.sys.H, 1)))
    verbose && @info "Getting Covariance..."
    ĈX_op = getĈX(enkf, X_forecast; with_matrix=!enkf.isiterative)
    enkf.sys.Cϵ = LinearMap(get_cov(enkf.ϵy, t))

    # Initial guess?
    fill!(enkf.θ, enkf.θinit)

    θold = zero(enkf.θ)
    for i = 1:enkf.Niter
        verbose && @info "IAS Optimization i = $i"

        copy!(θold, enkf.θ)
        verbose && @info "θ copied"

        # Update x
        update_x!(enkf, X_forecast_loop, perturbed_obs, ĈX_op, enkf.θ, t, X_analysis, verbose)

        verbose && @info "x updated"
        # Update theta
        update_θ!(enkf, X_analysis, enkf.θ, verbose)

        if norm(enkf.θ - θold) / norm(θold) < enkf.rtolθ
            break
        end
    end
    verbose && @info "Finished optimization loop."
    update_x!(enkf, X_forecast, perturbed_obs, ĈX_op, enkf.θ, t, X_analysis, verbose)
    return X_analysis, enkf.θ
end