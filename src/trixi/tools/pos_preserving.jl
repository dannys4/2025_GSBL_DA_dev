export positivity_preserving_noise1d

function Trixi.limiter_zhang_shu!(u, threshold::Real, variable,
    mesh, equations, dg::DGMulti, cache)
    weights = dg.basis.wq

    for element in axes(u, 2)
        # determine minimum value
        value_min = typemax(eltype(eltype(u)))
        for i in eachnode(dg)
            u_node = u[i, element]
            tmp = variable(u_node, equations)
            value_min = min(value_min, tmp)
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        u_mean = zero(eltype(u))
        for i in eachnode(dg)
            u_node = u[i, element]
            u_mean += u_node * weights[i]
        end
        # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
        u_mean = u_mean / 2^ndims(mesh)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for i in eachnode(dg)
            u_node = u[i, element]
            u[i, element] = theta * u_node + (1 - theta) * u_mean
        end
    end

    return nothing
end


"""
    positivity_preserving_noise1d(f0::SmoothPeriodic, N_ens::Int, mesh::DGMultiMesh, pos_vars::AbstractVector{<:AbstractString}, noise_proportion::Float, pos_transform=(exp,log))

Add a proportion of smooth periodic noise to the states, preserving positivity for certain _cons_ variables (e.g., rho, rho_e).
"""
function positivity_preserving_noise1d(
    f0::AbstractFilterStateInitialization, initial_condition::Function, N_ens::Int,
    sys::TrixiSystem, pos_vars::AbstractVector{<:AbstractString},
    noise_sigma::Float64; is_dirichlet::NTuple{2,Bool}=(true, true), pos_transform=(exp, log)
)
    (; mesh, equations) = sys
    @assert mesh isa DGMultiMesh
    xq = mesh.md.xq
    xgrid = GridFromMesh(sys)
    Nvar = nvariables(equations)
    grid_shape = size(xq)

    pos_var_flags = in.(Trixi.varnames(cons2prim, equations), (pos_vars,))
    if !any(pos_var_flags)
        @warn "No positive variables found!"
    end
    x0_quad = map(x -> cons2prim(initial_condition(x, 0., equations), equations), xq) # Initial condition evaluated on the quad nodes
    transforms = [flag ? pos_transform : (identity, identity) for flag in pos_var_flags]
    noise_levels = ntuple(Nvar) do var_idx
        max_var = maximum(x[var_idx] for x in x0_quad)
        _, inverse_map = transforms[var_idx]
        inverse_map(max_var)
    end
    X0 = zeros(Nvar, grid_shape..., N_ens)
    noise_arr = similar(X0)
    for ens_idx in 1:N_ens
        regenerate!(f0)
        X0_slice = selectdim(X0, ndims(X0), ens_idx)
        out_f0 = f0(xgrid)
        out_f0 = permutedims(reshape(out_f0, grid_shape..., Nvar), (3, 1, 2))
        copy!(X0_slice, out_f0)
        copy!(selectdim(noise_arr, ndims(X0), ens_idx), out_f0)
        for c_idx in CartesianIndices(x0_quad)
            node_idx, elem_idx = Tuple(c_idx)
            x0_prim = x0_quad[c_idx]
            for var_idx in 1:Nvar
                forward_map, inverse_map = transforms[var_idx]
                x0 = X0_slice[var_idx, node_idx, elem_idx]
                x0_noise = noise_levels[var_idx] * x0
                new_ens_val = forward_map(
                    inverse_map(x0_prim[var_idx]) + noise_sigma * x0_noise
                )
                X0_slice[var_idx, node_idx, elem_idx] = new_ens_val
            end
        end
    end
    reshape(X0, :, N_ens), reshape(noise_arr, :, N_ens)
end
