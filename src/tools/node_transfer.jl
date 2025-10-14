export ensemble_to_itp

function node_transfer_matmul!(out, mat, in)
    Trixi.apply_to_each_field(Trixi.mul_by!(mat), out, in)
end

function get_interp_node_vals!(dg::DGMulti, cons_quad, cons_interp)
    # We need to move them to the Lobatto-Legendre nodes
    node_transfer_matmul!(cons_interp, dg.basis.Pq, cons_quad)
    nothing
end

function get_interp_node_vals!(::DGSEM, cons_quad, cons_interp)
    # We need to move them to the Lobatto-Legendre nodes
    copy!(cons_interp, cons_quad)
end

function get_quadrature_node_vals!(dg::DGMulti, cons_quad, cons_interp)
    node_transfer_matmul!(cons_quad, dg.basis.Vq, cons_interp)
    nothing
end

function get_quadrature_node_vals!(::DGSEM, cons_quad, cons_interp)
    copy!(cons_quad, cons_interp)
end

function ensemble_to_itp(ensemble, sys::TrixiSystem)
    x_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    x_ens_itp = similar(x_quad, (size(x_quad)..., size(ensemble, 2)))
    for t_idx in axes(ensemble, 2)
        vec2sol!(x_quad, @view(ensemble[:, t_idx]), sys.equations)
        HierarchicalDA.get_interp_node_vals!(sys.dg, x_quad, selectdim(x_ens_itp, ndims(x_ens_itp), t_idx))
    end
    x_ens_itp
end