export get_plot_ensemble, ensemble_to_itp
include("tools/grid_from_mesh.jl")
include("tools/mesh2d.jl")
include("tools/node_transfer.jl")
include("tools/pos_preserving.jl")
include("tools/vec2sol.jl")

function ensemble_to_itp(ensemble, sys::TrixiSystem)
    x_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    x_ens_itp = similar(x_quad, (size(x_quad)..., size(ensemble, 2)))
    for t_idx in axes(ensemble, 2)
        vec2sol!(x_quad, @view(ensemble[:, t_idx]), sys.equations)
        HierarchicalDA.get_interp_node_vals!(sys.dg, x_quad, selectdim(x_ens_itp, ndims(x_ens_itp), t_idx))
    end
    x_ens_itp
end


function get_plot_ensemble(ensemble::Matrix{Float64}, sys::TrixiSystem)
    Ne = size(ensemble, 2)
    sys.equations isa Trixi.AbstractEquations{1} || throw(ArgumentError("Requires one-dimensional system"))
    Nvar = nvariables(sys.equations)
    ensemble_itp = ensemble_to_itp(ensemble, sys)
    N_elem = size(ensemble_itp, 2)
    x_plot = vec(sys.dg.basis.Vp * sys.mesh.md.x)
    ensemble_plot = Array{Float64,3}(undef, size(sys.dg.basis.Vp, 1) * N_elem, Nvar, Ne)
    for ens_idx in axes(ensemble_plot, 3)
        member_itp = @view ensemble_itp[:, :, ens_idx]
        data = map(x -> sys.dg.basis.Vp * x,
            StructArrays.components(cons2prim.(member_itp, sys.equations))
        )
        data_plot = reduce(hcat, vec.(data))
        copy!(@view(ensemble_plot[:, :, ens_idx]), data_plot)
    end
    x_plot, ensemble_plot
end