export setup_burgers

import Trixi: prim2cons
@inline Trixi.prim2cons(u, equation::InviscidBurgersEquation1D) = u

# Setup tools for time-stepper of inviscid Burger's equation
function setup_burgers(polydeg, cells_per_dimension)

    equations = InviscidBurgersEquation1D()

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_ec

    # polydeg = 4
    basis = DGMultiBasis(Trixi.Line(), polydeg, approximation_type = GaussSBP())

    indicator_sc = IndicatorHennemannGassner(
        equations,
        basis,
        alpha_max = 0.5,
        alpha_min = 0.001,
        alpha_smooth = true,
        variable = first,
    )

    volume_integral = VolumeIntegralShockCapturingHG(
        indicator_sc;
        volume_flux_dg = volume_flux,
        volume_flux_fv = surface_flux,
    )

    dg = DGMulti(
        basis,
        surface_integral = SurfaceIntegralWeakForm(surface_flux),
        volume_integral = volume_integral,
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
