export get_plot_ensemble, ensemble_to_itp, ensemble_to_quad, get_filter_quad_pts
include("tools/grid_from_mesh.jl")
include("tools/mesh2d.jl")
include("tools/node_transfer.jl")
include("tools/pos_preserving.jl")
include("tools/vec2sol.jl")

function get_filter_quad_pts(ensemble, sys::TrixiSystem)
    ens_quad = ensemble_to_quad(ensemble, sys, g=(x, _) -> identity(x))
    if eltype(ens_quad) <: AbstractVector
        Nvar = nvariables(sys.equations)
        ens_quad_ret = Matrix{Float64}(undef, Nvar, length(ens_quad))
        for quad_idx in eachindex(ens_quad)
            quad_val = ens_quad[quad_idx]
            for var_idx in 1:Nvar
                ens_quad_ret[var_idx, quad_idx] = quad_val[var_idx]
            end
        end
        ens_quad = reshape(ens_quad_ret, Nvar, size(ens_quad)...)
    end
    ens_quad
end

function ensemble_to_quad(ensemble, sys::TrixiSystem; vec2sol_kwargs...)
    x_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    x_ens_quad = similar(x_quad, (size(x_quad)..., size(ensemble, 2)))
    for ens_idx in axes(ensemble, 2)
        vec2sol!(x_quad, @view(ensemble[:, ens_idx]), sys.equations; vec2sol_kwargs...)
        copy!(selectdim(x_ens_quad, ndims(x_ens_quad), ens_idx), x_quad)
    end
    x_ens_quad
end

function ensemble_to_itp(ensemble, sys::TrixiSystem; use_cons::Bool = true, ode_transforms::Union{Nothing, <:NamedTuple} = nothing)
    x_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys.semi)...)
    x_ens_itp = similar(x_quad, (size(x_quad)..., size(ensemble, 2)))

    to_solver_transform = isnothing(ode_transforms) ? identity : ode_transforms.to_solver_transform
    g = use_cons ? ((x, eqns) -> prim2cons(to_solver_transform(x), eqns)) : ((x, eqns) -> x)
    for t_idx in axes(ensemble, 2)
        vec2sol!(x_quad, @view(ensemble[:, t_idx]), sys.equations; g)
        HierarchicalDA.get_interp_node_vals!(sys.dg, x_quad, selectdim(x_ens_itp, ndims(x_ens_itp), t_idx))
    end
    x_ens_itp
end


function get_plot_ensemble(ensemble::Matrix{Float64}, sys::TrixiSystem; use_cons::Bool = true, ode_transforms::Union{Nothing, <:NamedTuple} = nothing)
    Ne = size(ensemble, 2)
    sys.equations isa Trixi.AbstractEquations{1} || throw(ArgumentError("Requires one-dimensional system"))

    from_solver_transform = isnothing(ode_transforms) ? identity : ode_transforms.from_solver_transform
    c_fcn = use_cons ? (x -> from_solver_transform(cons2prim(x, sys.equations))) : (x -> x)
    Nvar = nvariables(sys.equations)
    ensemble_itp = ensemble_to_itp(ensemble, sys; use_cons, ode_transforms)
    N_elem = size(ensemble_itp, 2)
    x_plot = vec(sys.dg.basis.Vp * sys.mesh.md.x)
    ensemble_plot = Array{Float64,3}(undef, size(sys.dg.basis.Vp, 1) * N_elem, Nvar, Ne)
    for ens_idx in axes(ensemble_plot, 3)
        member_itp = @view ensemble_itp[:, :, ens_idx]
        data = map(x -> sys.dg.basis.Vp * x, StructArrays.components(map(c_fcn, member_itp)))
        data_plot = reshape(reduce(hcat, vec.(data)), :, Nvar)
        copy!(@view(ensemble_plot[:, :, ens_idx]), data_plot)
    end
    x_plot, ensemble_plot
end