export TrixiSystem, GridFromMesh

struct TrixiSystem{Eqns<:Trixi.AbstractEquations,Solver,MeshT,Semi<:Trixi.AbstractSemidiscretization}
    equations::Eqns

    dg::Solver

    mesh::MeshT

    semi::Semi
end

function GridFromMesh(sys::TrixiSystem{<:Any,<:DGSEM,<:StructuredMesh{1}})
    mesh, basis = sys.mesh, sys.dg.basis
    L = mesh.cells_per_dimension[1]
    verts = range(0, 1, length=L+1)[1:end-1]
    shift_nodes = (basis.nodes .+ 1)/2
    nodes01 = repeat(verts',length(shift_nodes),1) .+ (shift_nodes/L)
    nodes = vec(mesh.mapping.(nodes01*2 .- 1))
    return nodes
end

function GridFromMesh(sys::TrixiSystem{<:Any,<:DGMulti})
    return vec(sys.mesh.md.xq)
end