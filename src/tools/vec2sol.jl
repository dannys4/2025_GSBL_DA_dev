export sol2vec!, sol2vec, vec2sol!, vec2sol

# function sol2vec!(
#     x_vec::AbstractVector,
#     x_mat::AbstractMatrix,
#     equations::Trixi.AbstractEquations{1};
#     g::Function=cons2prim,
# )
#     Nvar = nvariables(equations)
#     Npoly, Nnodes = size(x_mat)

#     @assert length(x_vec) == Nvar * Npoly * Nnodes
#     for (i, xi) in enumerate(x_mat)
#         x̃i = g(xi, equations)
#         for k = 1:Nvar
#             x_vec[(k-1)*Npoly*Nnodes+i] = x̃i[k]
#         end
#     end
# end

# function sol2vec(x_mat::AbstractMatrix, equations::Trixi.AbstractEquations{1}; g::Function=cons2prim)
#     Nvar = nvariables(equations)
#     Npoly, Nnodes = size(x_mat)
#     x_vec = zeros(Nvar * Npoly * Nnodes)
#     sol2vec!(x_vec, x_mat, equations; g=g)
#     return x_vec
# end

# function sol2vec!(
#     x_vec::AbstractVector,
#     x_sol::AbstractVector,
#     equations::Trixi.AbstractEquations{1};
#     g::Function=cons2prim,
# )
#     Nvar = nvariables(equations)
#     N_nodes_total = length(x_sol) ÷ Nvar

#     for i in 1:N_nodes_total
#         xi = x_sol[i]
#         x̃i = g(xi, equations)
#         for k = 1:Nvar
#             x_vec[(k-1)*Nvar+i] = x̃i[k]
#         end
#     end
# end

function sol2vec!(
    x_vec::AbstractVector,
    x_sol::AbstractMatrix,
    equations::Trixi.AbstractEquations{__D,Nvar};
    g::Function=cons2prim,
) where {__D, Nvar}
    for node_idx in eachindex(x_sol)
        xi = x_sol[node_idx]
        node_vals = g(xi, equations)
        var_idxs = ((node_idx-1)*Nvar+1):(node_idx*Nvar)
        x_vec[var_idxs] .= node_vals
    end
    nothing
end

function sol2vec(x_sol::AbstractVector, equations::Trixi.AbstractEquations{1}; g::Function=cons2prim)
    x_vec = similar(x_sol)
    sol2vec!(x_vec, x_sol, equations; g=g)
    return x_vec
end

function sol2vec(x_sol::AbstractMatrix, equations::Trixi.AbstractEquations{2,Nvar}; g::Function=cons2prim) where {Nvar}
    x_vec = Vector{Float64}(undef, Nvar * length(x_sol))
    sol2vec!(x_vec, x_sol, equations; g=g)
    return x_vec
end

# function vec2sol!(
#     x_mat::AbstractMatrix,
#     x_vec::AbstractVector,
#     equations::Trixi.AbstractEquations{1};
#     g::Function=prim2cons,
# )
#     Nvar = nvariables(equations)
#     Npoly, Nnodes = size(x_mat)

#     tmp = zeros(Nvar)
#     for i in eachindex(x_mat)
#         for k = 1:Nvar
#             tmp[k] = x_vec[(k-1)*Npoly*Nnodes+i]
#         end
#         x_mat[i] = g(SVector{Nvar}(tmp), equations)
#     end
# end

# function vec2sol!(
#     x_sol::AbstractMatrix,
#     x_vec::AbstractVector,
#     equations::Trixi.AbstractEquations{1};
#     g::Function=prim2cons,
# )
#     Nvar = nvariables(equations)
#     N_nodes_total = length(x_sol) ÷ Nvar

#     tmp_vec = zeros(Nvar)
#     for elem_idx in 1:N_total_elem
#         node_idxs = ((elem_idx-1)*N_per_elem*Nvar+1):(elem_idx*N_per_elem*Nvar)
#         elem_nodes = @view x_vec[node_idxs]
#         for node_idx in 1:N_per_elem
#             vars_idx = ((node_idx-1)*Nvar+1):(node_idx*Nvar)
#             node = @view elem_nodes[vars_idx]
#             x_sol[node_idx, elem_idx] = g(SVector{Nvar}(node), equations)
#         end
#     end
# end

function vec2sol!(
    x_sol::AbstractMatrix,
    x_vec::AbstractVector,
    equations::Trixi.AbstractEquations{__D,Nvar};
    g::Function=prim2cons,
) where {__D, Nvar}
    N_per_elem, N_total_elem = size(x_sol)

    for elem_idx in 1:N_total_elem
        node_idxs = ((elem_idx-1)*N_per_elem*Nvar+1):(elem_idx*N_per_elem*Nvar)
        elem_nodes = @view x_vec[node_idxs]
        for node_idx in 1:N_per_elem
            vars_idx = ((node_idx-1)*Nvar+1):(node_idx*Nvar)
            node = @view elem_nodes[vars_idx]
            x_sol[node_idx, elem_idx] = g(SVector{Nvar}(node), equations)
        end
    end
    nothing
end

function vec2sol(x_vec::AbstractVector, equations::Trixi.AbstractEquations, semi::Trixi.AbstractSemidiscretization; g::Function=prim2cons)
    x_sol = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(semi)...)
    vec2sol!(x_sol, x_vec, equations; g=g)
    return x_sol
end
