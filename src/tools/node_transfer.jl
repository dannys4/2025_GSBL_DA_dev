function get_interp_node_vals!(dg::DGMulti, cons_quad, cons_interp)
    # We need to move them to the Lobatto-Legendre nodes
    mul!(cons_interp, dg.basis.Pq, cons_quad)
end

function get_interp_node_vals!(::DGSEM, cons_quad, cons_interp)
    # We need to move them to the Lobatto-Legendre nodes
    copy!(cons_interp, cons_quad)
end

function get_quadrature_node_vals!(dg::DGMulti, cons_quad, cons_interp)
    mul!(cons_quad, dg.basis.Vq, cons_interp)
end

function get_quadrature_node_vals!(::DGSEM, cons_quad, cons_interp)
    copy!(cons_quad, cons_interp)
end
