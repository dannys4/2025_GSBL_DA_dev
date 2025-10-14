export PolyAnnil2D, create_observation_operator2d, sample_initial_state2d

import TransportBasedInference2

function GridFromMesh(sys::TrixiSystem{<:Any,<:DGSEM,<:StructuredMesh{1}})
    mesh, basis = sys.mesh, sys.dg.basis
    L = mesh.cells_per_dimension[1]
    verts = range(0, 1, length=L + 1)[1:end-1]
    shift_nodes = (basis.nodes .+ 1) / 2
    nodes01 = repeat(verts', length(shift_nodes), 1) .+ (shift_nodes / L)
    nodes = vec(mesh.mapping.(nodes01 * 2 .- 1))
    return nodes
end

function GridFromMesh(sys::TrixiSystem{<:Any,<:DGMulti{1}})
    return vec(sys.mesh.md.xq)
end

# Because this uses intrinsic types from StartupDG, we keep this in the trixi subdir
function get_slice_elements(slice_idx, polydeg, N_cells, mode::Symbol)
    mode == :y || mode == :x || throw(ArgumentError("Unexpected mode: $mode"))
    # Which element in a given set of elements is this in
    elem_offset = (slice_idx - 1) ÷ (polydeg + 1) + 1
    if mode == :y
        # Which vertical slice within the element is this in
        (0:N_cells-1) * N_cells .+ elem_offset
    else
        (1:N_cells) .+ (elem_offset - 1) * N_cells
    end
end

function get_slice_nodes(slice_idx, polydeg, mode::Symbol)
    mode == :y || mode == :x || throw(ArgumentError("Unexpected mode: $mode"))
    shap_slice = mod1(slice_idx, polydeg + 1)
    shap = reshape(1:(polydeg+1)^2, polydeg + 1, :)
    mode == :y ? shap[shap_slice, :] : shap[:, shap_slice]
end

function get_square_mesh_N_cells(mesh::DGMultiMesh{2,Trixi.Affine})
    mesh.md.mesh_type isa Trixi.StartUpDG.VertexMappedMesh{Quad} || ArgumentError("Expected mesh to have quad elements. Got mesh type $(mesh.md.mesh_type)")
    N_cells = -1
    try
        N_cells = Int(sqrt(length(mesh.md.VX))) - 1
    catch e
        if e isa InexactError
            Nx, Ny = length(unique(mesh.md.VX)) - 1, length(unique(mesh.md.VY)) - 1
            ArgumentError("VerticalPolyAnnil2D only supports meshes with the same number of elements on each axis. Got ($Nx, $Ny)")
        else
            rethrow(e)
        end
    end
    return N_cells
end

# Only works with 2d non-curved meshes with quad elements
function __PolyAnnil2D(mesh::DGMultiMesh{2,Trixi.Affine}, PA_order::Int; isperiodic=false, kwargs...)
    N_cells = get_square_mesh_N_cells(mesh)
    (; xq, yq) = mesh.md
    num_nodes = length(yq)
    polydeg = Int(sqrt(size(yq, 1))) - 1
    init_slice_idx = 1
    # Initialize the local polynomial annihilator
    # X information
    vv_x = get_slice_elements(init_slice_idx, polydeg, N_cells, :x)
    nn_x = get_slice_nodes(init_slice_idx, polydeg, :x)
    all_x_quad = xq[nn_x, vv_x]
    vec_quad_x = vec(all_x_quad)
    PA_local_x = PolyAnnil_single(vec_quad_x, PA_order; isperiodic, kwargs...)
    PA_global_x = spzeros(num_nodes, num_nodes)

    # Y information
    vv_y = get_slice_elements(init_slice_idx, polydeg, N_cells, :y)
    nn_y = get_slice_nodes(init_slice_idx, polydeg, :y)
    all_y_quad = yq[nn_y, vv_y]
    vec_quad_y = vec(all_y_quad)
    PA_local_y = PolyAnnil_single(vec_quad_y, PA_order; isperiodic, kwargs...)
    PA_global_y = spzeros(num_nodes, num_nodes)

    PA_offset = (PA_order + 1) ÷ 2
    node_indices = LinearIndices(yq)
    # There are polydeg+1 slices per element and N_cells elements per side
    for slice_idx in 1:((polydeg+1)*N_cells)
        vv_x = get_slice_elements(slice_idx, polydeg, N_cells, :x)
        nn_x = get_slice_nodes(slice_idx, polydeg, :x)
        col_idxs_x = vec(node_indices[nn_x, vv_x])
        row_idxs_x = isperiodic ? col_idxs_x : col_idxs_x[PA_offset+1:end-PA_offset]
        PA_global_x[row_idxs_x, col_idxs_x] .+= PA_local_x

        vv_y = get_slice_elements(slice_idx, polydeg, N_cells, :y)
        nn_y = get_slice_nodes(slice_idx, polydeg, :y)
        col_idxs_y = vec(node_indices[nn_y, vv_y])
        row_idxs_y = isperiodic ? col_idxs_y : col_idxs_y[PA_offset+1:end-PA_offset]
        PA_global_y[row_idxs_y, col_idxs_y] .+= PA_local_y
    end
    vcat(PA_global_x, PA_global_y), vcat(vec_quad_x, vec_quad_y)
end

function PolyAnnil2D(mesh::DGMultiMesh, PA_order; Nvar=1, kwargs...)
    base_PA, vec_quad = __PolyAnnil2D(mesh, PA_order; kwargs...)
    nz_idx = vec(mapreduce(!iszero, |, base_PA, dims=2))
    base_PA = base_PA[nz_idx, :]
    select_kron = IdentityMap(Nvar)
    full_PA = Nvar == 1 ? base_PA : kron(base_PA, select_kron)
    PolyAnnil(vec_quad, PA_order, sparse(full_PA)), nz_idx
end

PolyAnnil2D(sys::TrixiSystem, PA_order; kwargs...) = PolyAnnil2D(sys.mesh, PA_order; kwargs...)

gaspari2D(offset_x, offset_y, radius) = gaspari(2 * sqrt(abs2(offset_x) + abs2(offset_y)) / radius)

function LocalizationMatrix2D(
    mesh::DGMultiMesh{2,Trixi.Affine},
    local_radius::Int,
    kernel::Function,
    isperiodic::Bool)

    N_cells = get_square_mesh_N_cells(mesh)
    yq = mesh.md.yq
    polydeg = Int(sqrt(size(yq, 1))) - 1
    # @assert local_radius <= polydeg + 1 "Currently only supports radius that is below polynomial degree. Got $local_radius > $(polydeg+1)"
    indices = reshape(LinearIndices(yq), polydeg + 1, polydeg + 1, N_cells, N_cells)

    # How many elements over the index is
    get_elem_offset(idx) = sign(idx - 1) * ((idx < 1) + (abs(idx) - (idx > polydeg)) ÷ (polydeg + 1))
    rows, cols, vals = Int[], Int[], Float64[]
    @showprogress for (node_matrix_row_idx, c_idx) in enumerate(CartesianIndices(indices))
        elem_row_idx, elem_col_idx, global_row_idx, global_col_idx = Tuple(c_idx)
        for location_offset in CartesianIndices((-local_radius:local_radius, -local_radius:local_radius))
            row_offset_rad, col_offset_rad = Tuple(location_offset)
            row_offset = elem_row_idx + row_offset_rad
            col_offset = elem_col_idx + col_offset_rad
            # Find where the neighbor is within the element
            row_elem_neigh = mod1(row_offset, polydeg + 1)
            col_elem_neigh = mod1(col_offset, polydeg + 1)
            # Find which element the neighbor belongs to
            row_global_neigh = global_row_idx + get_elem_offset(row_offset)
            col_global_neigh = global_col_idx + get_elem_offset(col_offset)

            if !isperiodic # If not periodic, check if we step over the bounds
                invalid_row = row_global_neigh > N_cells || row_global_neigh < 1
                invalid_col = col_global_neigh > N_cells || col_global_neigh < 1
                (invalid_row || invalid_col) && continue
            end
            # Wrap around for periodicity
            row_global_neigh = mod1(row_global_neigh, N_cells)
            col_global_neigh = mod1(col_global_neigh, N_cells)
            # Get the neighbor's node index in the global matrix
            node_idx_neigh = indices[row_elem_neigh, col_elem_neigh, row_global_neigh, col_global_neigh]
            # Calculate the value for the localization
            val = kernel(row_offset_rad, col_offset_rad)
            push!(rows, node_matrix_row_idx)
            push!(cols, node_idx_neigh)
            push!(vals, val)
        end
    end
    return rows, cols, vals
end

function block_toeplitz_tridiag(matrix, block_size)
    N = size(matrix, 1)
    num_blocks = N ÷ block_size
    main_block = matrix[1:block_size, 1:block_size]
    upper_block = matrix[1:block_size, block_size.+(1:block_size)]
    lower_block = matrix[block_size.+(1:block_size), 1:block_size]
    main_full = kron(IdentityMap(num_blocks), main_block)
    upper_full = kron(LinearMap(diagm(1 => ones(Bool, num_blocks - 1))), upper_block)
    lower_full = kron(LinearMap(diagm(-1 => ones(Bool, num_blocks - 1))), lower_block)
    return main_full + upper_full + lower_full
end

"""
    Localization(mesh::DGMultiMesh{2,Trixi.Affine}, local_radius::Int; kernel::Function, isperiodic=true, Nvar=1)

Determine the localization via a kernel(dx, dy). Defaults to a gaspari-cohn kernel determined by `local_radius`.

"""
function TransportBasedInference2.Localization(
    mesh::DGMultiMesh{2,Trixi.Affine},
    local_radius::Int;
    kernel::Function=(x, y) -> gaspari2D(x, y, local_radius),
    isperiodic=true,
    Nvar::Int=1
)
    rows, cols, vals = LocalizationMatrix2D(mesh, local_radius, kernel, isperiodic)
    loc = sparse(rows, cols, vals)
    dropzeros!(loc)
    if isperiodic
        loc_map = LinearMap(loc, issymmetric=true)
    else
        N_cells = get_square_mesh_N_cells(mesh)
        polydeg_p = Int(sqrt(size(loc, 1)) ÷ N_cells)^2
        loc_small = collect(loc[1:N_cells*N_cells, 1:N_cells*N_cells])
        main_block = block_toeplitz_tridiag(loc_small[1:end÷2, 1:end÷2], polydeg_p)
        upper_block = block_toeplitz_tridiag(loc_small[1:end÷2, (end÷2+1):end], polydeg_p)
        lower_block = block_toeplitz_tridiag(loc_small[(end÷2+1):end, 1:end÷2], polydeg_p)
        loc_small_map = [main_block upper_block; lower_block main_block]
        kron_I_size = size(loc, 1) ÷ (N_cells * N_cells)
        select_kron = IdentityMap(kron_I_size)
        loc_map = LinearMap(kron(select_kron, loc_small_map), issymmetric=true)
    end
    select_kron = IdentityMap(Nvar)
    loc_vars = Nvar == 1 ? loc_map : kron(loc_map, select_kron)
    return Localization(loc_vars)
end

TransportBasedInference2.Localization(
    sys::TrixiSystem,
    local_radius::Int;
    kwargs...
) = Localization(sys.mesh, local_radius; kwargs...)

# Metric should map (row_diff::Int, col_diff::Int) -> Float64
# If you are comparing integer coords (5, 8) and (7, 2), then output should assume input (-2, 6)

# function TransportBasedInference2.Localization(
#     mesh::DGMultiMesh{2,Trixi.Affine},
#     local_radius::Int;
#     kernel::Function=(x, y) -> gaspari2D(x, y, local_radius),
#     isperiodic=true,
#     Nvar::Int=1
# )
#     rows, cols, vals = LocalizationMatrix2D(mesh, local_radius, kernel, isperiodic)
#     loc = sparse(rows, cols, vals)
#     dropzeros!(loc)
#     if isperiodic
#         loc_map = LinearMap(loc, issymmetric=true)
#     else
#         N_cells = get_square_mesh_N_cells(mesh)
#         loc_small = collect(loc[1:N_cells*N_cells, 1:N_cells*N_cells])
#         loc_small_map = LinearMap(loc_small, issymmetric=true)
#         kron_I_size = size(loc, 1) ÷ (N_cells * N_cells)
#         select_kron = IdentityMap(kron_I_size)
#         loc_map = kron(select_kron, loc_small_map)
#     end
#     select_kron = IdentityMap(Nvar)
#     loc_vars = Nvar == 1 ? loc : kron(loc_map, select_kron)
#     return Localization(loc_vars)
# end

function create_observation_operator2d(mesh::DGMultiMesh{2}, spacing::Int, offset::Int)
    @assert offset < spacing
    N_cells = get_square_mesh_N_cells(mesh)
    (; xq, yq) = mesh.md
    polydeg = Int(sqrt(size(yq, 1))) - 1
    # @assert spacing % (polydeg + 1) == 0
    # @assert local_radius <= polydeg + 1 "Currently only supports radius that is below polynomial degree. Got $local_radius > $(polydeg+1)"
    xq_reshape = reshape(xq, polydeg + 1, polydeg + 1, N_cells, N_cells)
    yq_reshape = reshape(yq, polydeg + 1, polydeg + 1, N_cells, N_cells)

    # How many elements over the index is
    get_elem_offset(idx) = sign(idx - 1) * ((idx < 1) + (abs(idx) - (idx > polydeg)) ÷ (polydeg + 1))

    num_obs = ceil(Int, N_cells * (polydeg + 1) / spacing)^2
    obs_indices = zeros(Int, num_obs)
    obs_points = zeros(2, num_obs)
    obs_idx = 1
    for (node_matrix_idx, c_idx) in enumerate(CartesianIndices(yq_reshape))
        elem_row_idx, elem_col_idx, global_row_idx, global_col_idx = Tuple(c_idx)
        row_idx = (global_row_idx - 1) * (polydeg + 1) + elem_row_idx
        col_idx = (global_col_idx - 1) * (polydeg + 1) + elem_col_idx
        is_row_spaced = row_idx % spacing == offset # Make sure to get first column
        is_col_spaced = col_idx % spacing == offset
        if is_row_spaced && is_col_spaced
            obs_indices[obs_idx] = node_matrix_idx
            obs_points[:, obs_idx] .= (xq_reshape[c_idx], yq_reshape[c_idx])
            obs_idx += 1
        end
    end
    return SelectionMap(obs_indices, :out; in_size=length(yq_reshape)), obs_points
end

function create_observation_operator2d(mesh::DGMultiMesh{2}, spacing::Int; offset::Int=1, Nvar::Int=1, which_var::AbstractVector=1:Nvar)
    base_H = create_observation_operator2d(mesh, spacing, offset)
    Nvar == 1 && return base_H
    select_kron = IdentityMap(Nvar)
    which_var == 1:Nvar || (select_kron = select_kron[which_var, :])
    return kron(base_H, select_kron)
end

create_observation_operator2d(sys::TrixiSystem, spacing; kwargs...) = create_observation_operator2d(sys.mesh, spacing; kwargs...)

function sample_initial_state2d(mesh::DGMultiMesh{2}, f0_row, f0_col; unique_digits=3, Nvar=1, transform_fcn=ntuple(Returns(identity), Nvar), f0_scale=0.5)
    @assert length(transform_fcn) == Nvar
    Nx_var = length(mesh.md.xq)
    x0_ens = Matrix{Float64}(undef, Nx_var, Nvar)
    # Crude way of getting one dimensional grid, need to round due to numerical issues.
    Ncells_dim = Int(sqrt(mesh.md.num_elements))
    polydeg = Int(sqrt(size(mesh.md.xq, 1)) - 1)
    grid1d = unique(x -> round(x, digits=unique_digits), mesh.md.xq)
    @assert length(grid1d) == Ncells_dim * (polydeg + 1) "Unexpected grid length: Got $(length(grid1d)), expected $(Ncells_dim * (polydeg + 1))"
    f0_row_eval = f0_row(grid1d)
    f0_col_eval = f0_col(grid1d)
    for which_var in 1:Nvar
        regenerate!(f0_row)
        regenerate!(f0_col)
        fcn = transform_fcn[which_var]
        f0_row(f0_row_eval, grid1d)
        f0_col(f0_col_eval, grid1d)
        var_ens = @view x0_ens[:, which_var]
        for (lin_idx, c_idx) in enumerate(CartesianIndices((1:(polydeg+1), 1:(polydeg+1), 1:Ncells_dim, 1:Ncells_dim)))
            in_elem_row, in_elem_col, global_elem_row, global_elem_col = Tuple(c_idx)
            # These are each the indices for the f0_row_eval and f0_col_eval, respectively
            global_idx_row = (global_elem_row - 1) * (polydeg + 1) + in_elem_row
            global_idx_col = (global_elem_col - 1) * (polydeg + 1) + in_elem_col
            var_ens[lin_idx] = fcn(f0_scale * (f0_row_eval[global_idx_row] + f0_col_eval[global_idx_col]))
        end
    end
    return vec(x0_ens)
end

sample_initial_state2d(sys::TrixiSystem, f0_row, f0_col; kwargs...) = sample_initial_state2d(sys.mesh, f0_row, f0_col; kwargs...)