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

function vec2sol(x_vec::AbstractVector, equations, semi; g::Function = prim2cons)
    Nvar = nvariables(equations)

    x_mat = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(semi)...)
    vec2sol!(x_mat, x_vec, equations; g = g)
    return x_mat
end
