module HierarchicalDAMakieExt

using Makie
using HierarchicalDA
using Trixi




function HierarchicalDA.trixiheatmaps(itps, titles, sys::TrixiSystem; variable="u", plot_mesh=false)
    TrixiMakie = Base.get_extension(Trixi, :TrixiMakieExt)
    length(itps) == length(titles) || throw(ArgumentError())
    limits_x = extrema(sys.semi.mesh.md.VX)
    limits_y = extrema(sys.semi.mesh.md.VY)
    limits = (limits_x..., limits_y...)
    fig = Figure(size=(500 * length(titles) + 100, 500))
    pd_sols = map(itps) do itp
        PlotData2D(itp, sys.semi)[variable]
    end
    axs = map(enumerate(titles)) do (j, title)
        Axis(fig[1, j], aspect=DataAspect(); title, limits)
    end
    colorranges = map(pd_sols) do pds
        map(getindex, extrema(pds.plot_data.data))
    end
    colorrange = (minimum(first.(colorranges)), maximum(last.(colorranges)))
    for j in eachindex(titles)
        TrixiMakie.trixiheatmap!(axs[j], pd_sols[j]; plot_mesh, colorrange)
    end
    Colorbar(fig[1, end+1]; colorrange)
    fig, axs
end

end # module