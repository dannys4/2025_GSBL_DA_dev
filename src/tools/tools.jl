export modfloat, interp_columns
modfloat(a, b) = abs(mod(a + 0.5 * b, b) - 0.5 * b)

function interp_columns(data::AbstractMatrix{Float64}, N_interp::Int)
    Nx, Tf = size(data)
    heatmap_data = zeros(N_interp * Tf, Nx)
    heatmap_data[1:N_interp:end, :] .= data'
    max_idx = -1
    for t in 1:N_interp-1
        rhs = (t * data[:, 2:end] + (N_interp - t) * data[:, 1:end-1]) / N_interp
        lhs_idx = (t+1:N_interp:size(heatmap_data, 1))[1:size(rhs, 2)]
        max_idx = max(max_idx, lhs_idx[end])
        heatmap_data[lhs_idx, :] .= rhs'
    end
    heatmap_data = heatmap_data[1:max_idx, :]
end

include(joinpath(@__DIR__, "PA.jl"))
include(joinpath(@__DIR__, "smooth_periodic.jl"))
include(joinpath(@__DIR__, "crps.jl"))
include(joinpath(@__DIR__, "selection_maps.jl"))
include(joinpath(@__DIR__, "localization.jl"))
include(joinpath(@__DIR__, "viz.jl"))
include(joinpath(@__DIR__, "shock_localization.jl"))
include(joinpath(@__DIR__, "differentiation.jl"))