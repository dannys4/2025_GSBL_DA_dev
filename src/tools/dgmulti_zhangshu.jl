function Trixi.limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh, equations, dg::DGMulti, cache)
    weights  = dg.basis.wq

    Trixi.@threaded for element in axes(u,2)
        # determine minimum value
        value_min = typemax(eltype(eltype(u)))
        for i in eachnode(dg)
            u_node = u[i,element]
            tmp = variable(u_node, equations)
            value_min = min(value_min, tmp)
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        u_mean = zero(eltype(u))
        for i in eachnode(dg)
            u_node = u[i,element]
            u_mean += u_node * weights[i]
        end
        # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
        u_mean = u_mean / 2^ndims(mesh)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for i in eachnode(dg)
            u_node = u[i,element]
            u[i,element] = theta * u_node + (1-theta) * u_mean
        end
    end

    return nothing
end