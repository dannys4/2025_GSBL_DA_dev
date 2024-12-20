export setup_linear_advection

import Trixi: prim2cons
@inline Trixi.prim2cons(u, equation::LinearScalarAdvectionEquation1D) = u

# Setup tools for time-stepper of linear advection equation
function setup_linear_advection(polydeg, cells_per_dimension)

    equations = LinearScalarAdvectionEquation1D(1.0)

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_ec

    # polydeg = 4
    basis = DGMultiBasis(Trixi.Line(), polydeg, approximation_type = GaussSBP())

    surface_flux = FluxLaxFriedrichs()

    dg = DGMulti(
        basis,
        surface_integral = SurfaceIntegralWeakForm(surface_flux),
        volume_integral = VolumeIntegralWeakForm(),
    )

    ###############################################################################
    #  setup the 1D mesh

    # cells_per_dimension = (64,)
    mesh = DGMultiMesh(
        dg,
        cells_per_dimension,
        coordinates_min = (-1.0,),
        coordinates_max = (1.0,),
        periodicity = true,
    )

    ###############################################################################
    #  setup the semidiscretization and ODE problem

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        initial_condition_convergence_test,
        dg,
    )
    return TrixiSystem(equations, dg, mesh, semi)
end
