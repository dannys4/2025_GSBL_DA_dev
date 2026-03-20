using Base: size, Matrix

export second_order_finite_difference_op, DGMultiDiff1D

function second_order_finite_difference_op(xgrid, grid_bds, is_periodic::Bool, divide_by_length::Bool)
    xgrid_a, xgrid_b = grid_bds
    grid_len = xgrid_b - xgrid_a
    is_periodic || throw(ArgumentError("Only implemented for is_periodic"))
    xgrid_circ = [xgrid; (xgrid[1] + grid_len)]
    h_plus = xgrid_circ[2:end] - xgrid_circ[1:end-1]
    h_minus = [h_plus[end]; h_plus[1:end-1]]
    h_sum = h_plus + h_minus
    ratio_plus = h_plus ./ h_sum
    # Reorder
    ratio_plus = [ratio_plus[2:end]; ratio_plus[1]]
    ratio_minus = h_minus ./ h_sum
    # We look at variation, not worried about the step size
    if divide_by_length
        denom = @. 0.5 * h_plus * h_minus
    else
        denom = 0.5
    end
    d_sq_op = Tridiagonal(ratio_plus, -ones(length(xgrid_circ)), ratio_minus)
    d_wrap = sparse(d_sq_op[1:end-1, 1:end-1] ./ denom)
    d_wrap[1, end], d_wrap[end, 1] = d_sq_op[end, end-1] / denom[1], d_sq_op[end-1, end] / denom[end]
    d_wrap
end

# This is only a linear map if your DG discretization uses a linear stabilizer
# This is likely fine for the fully hyperbolic systems---consider the flux used for the volume integral
struct DGMultiDiff1D{S <: TrixiSystem, NT <: NamedTuple} <: LinearMaps.LinearMap{Float64}
    sys::S
    divide_by_J::Bool
    workspace::NT
    size::Dims{2}
    function DGMultiDiff1D(sys::_S, divide_by_J::Bool) where {_S}
        ndims(sys.equations) == 1 || throw(ArgumentError("Can only handle one-dimensional equations"))
        u_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
        u_itp = similar(u_quad)
        uf_cache1 = similar(u_quad, (2, size(u_quad, 2)))
        uf_cache2 = similar(uf_cache1)
        workspace = (; u_quad, u_itp, uf_cache1, uf_cache2)
        vec_size = length(u_quad) * nvariables(sys.equations)
        new{_S, typeof(workspace)}(sys, divide_by_J, workspace, (vec_size, vec_size))
    end
end

Base.size(A::DGMultiDiff1D) = A.size

function LinearMaps._unsafe_mul!(out_vec, trixi_diff::DGMultiDiff1D, u_quad_vec::AbstractVector)
    # Modified from: https://jlchan.github.io/StartUpDG.jl/v0.11/ex_dg_deriv/
    (; sys, divide_by_J, workspace) = trixi_diff
    (; u_quad, u_itp, uf_cache1, uf_cache2) = workspace
    for arr in (out_vec, u_quad, u_itp, uf_cache1, uf_cache2)
        fill!(arr, zero(eltype(arr)))
    end
    id_fcn = (x,_) -> x
    vec2sol!(u_quad, u_quad_vec, sys.equations; g=id_fcn)
    HierarchicalDA.get_interp_node_vals!(sys.dg, u_quad, u_itp)
    (;Vf,Dr,LIFT) = sys.dg.basis
    (;rxJ,J,nxJ,mapP) = sys.mesh.md

    # uf = Vf*u_itp
    uf = mul!(uf_cache1, Vf, u_itp)

    # ujump = uf[mapP]-uf
    ujump = uf_cache2
    for idx in eachindex(uf)
        ujump[idx] = uf[mapP[idx]] - uf[idx]
    end

    # derivatives using chain rule + lifted flux terms
    # ux = rxJ.*(Dr*u_itp) # + sxJ.*(Dr*u)
    ux = mul!(u_quad, Dr, u_itp)
    for idx in eachindex(ux)
        ux[idx] *= rxJ[idx]
    end

    # du_dx = ux + LIFT*(.5*ujump.*nxJ)
    for idx in eachindex(ujump)
        ujump[idx] = 0.5 * nxJ[idx] * ujump[idx]
    end
    du_dx = mul!(ux, LIFT, ujump, true, true)

    if divide_by_J
        # du_dx = du_dx ./ J
        for idx in eachindex(du_dx)
            du_dx[idx] /= J[idx]
        end
    end
    duq_dx = u_itp # alias
    HierarchicalDA.get_quadrature_node_vals!(sys.dg, duq_dx, du_dx)
    sol2vec!(out_vec, duq_dx, sys.equations; g=id_fcn)
    out_vec
end
