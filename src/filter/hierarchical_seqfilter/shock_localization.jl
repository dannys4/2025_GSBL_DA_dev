struct ShockLocalization{T<:Union{AbstractMatrix{Float64},LinearMap{Float64}},U<:AbstractMatrix{Float64}}
    S::T
    ρX::U
    ρX_cache::U
    thresh::Float64
    is_periodic::Bool
    block_radius::Int
    Nvar::Int
end

function get_localization_mask(Loc::Localization)
    return Loc.ρX_cache
end

"""
    ShockLocalization(detector, grid, radius, metric, scale; symm_kernel, )
"""
function ShockLocalization(detector, grid::AbstractVector{Float64}, radius::Float64, metric::Function, scale::Real=true; Nvar::Int=1, symm_kernel=false, is_sparse=false, herm_matrix=false, thresh=2.0, is_periodic=false)
    ρX_dense = scale * Locgaspari(grid, grid, radius, metric, symm_kernel)
    ρX_sparse = is_sparse ? sparse(ρX_dense) : ρX_dense
    if Nvar > 1
        ρX_sparse = kron(ρX_sparse, ones(Nvar, Nvar))
    end
    ρX = herm_matrix ? Hermitian(ρX_sparse, :U) : ρX_sparse
    block_radius = (sum(@view(ρX_sparse[1, 1:div(end, 2)]) > 0)) - 1
    ShockLocalization(detector, ρX, zero(ρX), thresh, is_periodic, block_radius, Nvar)
end

ShockLocalization(detector, Nx::Int, args...; kwargs...) = ShockLocalization(detector, 1.0:Nx, args...; kwargs...)

function fill_mask!(
    mask::AbstractMatrix,
    shock_thresh::AbstractVector,
    Nvar::Int,
    block_radius::Int,
    is_periodic::Bool,
)
    Nx = size(mask, 1)
    function zero_out_upper_lower!(min_1, max_1, min_2, max_2)
        mask[min_1:max_1, min_2:max_2] .= 0.
        mask[min_2:max_2, min_1:max_1] .= 0.
    end
    for idx in 1:Nx
        shock_thresh[idx] || continue
        idx_right = idx + 1 + (Nvar - mod1(idx, Nvar))
        idx_left = idx - mod1(idx, Nvar)
        @info "" idx idx_right idx_left
        # Delete everything in this same col and row except
        # for covariance across states on this node.
        mask[idx_right:end, (idx_left + 1) : (idx_right - 1)] .= 0.
        mask[1:idx_left, (idx_left + 1) : (idx_right - 1)] .= 0.
        mask[(idx_left + 1) : (idx_right - 1), 1:idx_left] .= 0.
        mask[(idx_left + 1) : (idx_right - 1), idx_right:end] .= 0.

        if idx > 1 && idx < Nx
            # Indices for upper right block, swapped for lower left block
            min_upper_row = max(1, idx_left - block_radius)
            max_upper_row = idx_left
            min_upper_col = idx_right
            max_upper_col = min(idx_right + block_radius - 1, Nx)
            zero_out_upper_lower!(min_upper_row, max_upper_row, min_upper_col, max_upper_col)
        end

        is_periodic || continue

        if idx <= Nvar || idx > (Nx-Nvar)
            mask[(end+1).-(1:block_radius), 1:block_radius] .= 0.
            mask[1:block_radius, (end+1).-(1:block_radius)] .= 0.
        elseif idx < block_radius || idx >= Nx - block_radius
            idx_1, idx_2 = minmax(idx_left + 1, Nx - (idx_left) + 1)
            # Upper right block
            min_upper_row = idx_2 - block_radius
            max_upper_row = idx_2 - 1
            min_upper_col = 1
            max_upper_col = idx_1 + block_radius
            zero_out_upper_lower!(min_upper_row, max_upper_row, min_upper_col, max_upper_col)
        end
    end
    mask
end

function construct_mask!(Loc::ShockLocalization, center_X::Matrix{Float64})
    (;is_periodic, Nvar, block_radius) = Loc
    shock = Loc.detector * center_X
    shock_var = sum(abs2, shock, dims=2)
    shock_thresh = minimum(shock_var) * Loc.thresh
    mask = Loc.ρX_cache
    copy!(mask, Loc.ρX)
    any(shock_thresh) && fill_mask!(mask, shock_thresh, Nvar, block_radius, is_periodic)
end
