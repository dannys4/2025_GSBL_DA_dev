export TrixiSystem

struct TrixiSystem
    equations::Trixi.AbstractEquations

    dg::DGMulti

    mesh::DGMultiMesh

    semi::SemidiscretizationHyperbolic
end
