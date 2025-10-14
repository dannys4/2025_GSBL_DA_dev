export sol2vec!, sol2vec, vec2sol!, vec2sol

function sol2vec!(
    x_vec::AbstractVector,
    x_sol::AbstractMatrix,
    equations::Trixi.AbstractEquations{__D,Nvar};
    g::Function=cons2prim,
) where {__D,Nvar}
    for node_idx in eachindex(x_sol)
        xi = x_sol[node_idx]
        node_vals = g(xi, equations)
        var_idxs = ((node_idx-1)*Nvar+1):(node_idx*Nvar)
        x_vec[var_idxs] .= node_vals
    end
    nothing
end

recurse_eltype(::Type{<:AbstractArray{T}}) where {T} = T
recurse_eltype(::Type{<:AbstractArray{T}}) where {T<:AbstractArray} = recurse_eltype(T)
recurse_eltype(::A) where {A<:AbstractArray} = recurse_eltype(A)

function sol2vec(x_sol::AbstractMatrix, equations::Trixi.AbstractEquations{1,__D}; g::Function=cons2prim) where {__D}
    x_vec = Vector{recurse_eltype(x_sol)}(undef, length(x_sol) * __D)
    sol2vec!(x_vec, x_sol, equations; g)
    return x_vec
end

function sol2vec(x_sol::AbstractMatrix, equations::Trixi.AbstractEquations{2,Nvar}; g::Function=cons2prim) where {Nvar}
    x_vec = Vector{Float64}(undef, Nvar * length(x_sol))
    sol2vec!(x_vec, x_sol, equations; g)
    return x_vec
end

function vec2sol!(
    x_sol::AbstractMatrix,
    x_vec::AbstractVector,
    equations::Trixi.AbstractEquations{__D,Nvar};
    g::Function=prim2cons,
) where {__D,Nvar}
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
