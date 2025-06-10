export sol2vec!, sol2vec, vec2sol!, vec2sol

function sol2vec!(
    x_vec::AbstractVector,
    x_mat::AbstractMatrix,
    equations;
    g::Function = cons2prim,
)
    Nvar = nvariables(equations)
    Npoly, Nnodes = size(x_mat)

    @assert length(x_vec) == Nvar * Npoly * Nnodes
    for (i, xi) in enumerate(x_mat)
        x̃i = g(xi, equations)
        for k = 1:Nvar
            x_vec[(k-1)*Npoly*Nnodes+i] = x̃i[k]
        end
    end
end

function sol2vec(x_mat::AbstractMatrix, equations; g::Function = cons2prim)
    Nvar = nvariables(equations)
    Npoly, Nnodes = size(x_mat)
    x_vec = zeros(Nvar * Npoly * Nnodes)
    sol2vec!(x_vec, x_mat, equations; g = g)
    return x_vec
end

function sol2vec!(
    x_vec::AbstractVector,
    x_sol::AbstractVector,
    equations;
    g::Function = cons2prim,
)
    Nvar = nvariables(equations)
    N_nodes_total = length(x_sol) ÷ Nvar

    for i in 1:N_nodes_total
        xi = @view x_sol[((i-1)*Nvar + 1):i*Nvar]
        x̃i = g(xi, equations)
        for k = 1:Nvar
            x_vec[(k-1)*Nvar + i] = x̃i[k]
        end
    end
end

function sol2vec(x_sol::AbstractVector, equations; g::Function = cons2prim)
    x_vec = similar(x_sol)
    sol2vec!(x_vec, x_sol, equations; g = g)
    return x_vec
end

function vec2sol!(
    x_mat::AbstractMatrix,
    x_vec::AbstractVector,
    equations;
    g::Function = prim2cons,
)
    Nvar = nvariables(equations)
    Npoly, Nnodes = size(x_mat)

    tmp = zeros(Nvar)
    for i in eachindex(x_mat)
        for k = 1:Nvar
            tmp[k] = x_vec[(k-1)*Npoly*Nnodes+i]
        end
        x_mat[i] = g(SVector{Nvar}(tmp), equations)
    end
end

function vec2sol!(
    x_sol::AbstractVector,
    x_vec::AbstractVector,
    equations;
    g::Function = prim2cons,
)
    Nvar = nvariables(equations)
    N_nodes_total = length(x_sol) ÷ Nvar

    tmp_vec = zeros(Nvar)
    for i in 1:N_nodes_total
        for k = 1:Nvar
            tmp_vec[k] = x_vec[(k-1)*N_nodes_total+i]
        end
        tmp_out = g(SVector{Nvar}(tmp_vec), equations)
        for k in 1:Nvar
            x_sol[(i-1)*Nvar+k] = tmp_out[k]
        end
    end
end

function vec2sol(x_vec::AbstractVector, equations, semi; g::Function = prim2cons)
    x_sol = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(semi)...)
    vec2sol!(x_sol, x_vec, equations; g = g)
    return x_sol
end
