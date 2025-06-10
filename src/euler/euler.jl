export initial_condition_shu_osher, setup_euler


# Shu-Osher initial condition for 1D compressible Euler equations
# Example 8 from Shu, Osher (1989).
# [https://doi.org/10.1016/0021-9991(89)90222-2](https://doi.org/10.1016/0021-9991(89)90222-2)
function initial_condition_shu_osher(
    x::AbstractVector,
    t,
    equations::CompressibleEulerEquations1D,
)
    x0 = -4

    rho_left = 27 / 7
    v_left = 4 * sqrt(35) / 9
    p_left = 31 / 3

    # Replaced v_right = 0 to v_right = 0.5 to avoid positivity issues.
    v_right = 0.5
    p_right = 1.0

    rho = ifelse(x[1] > x0, 1 + 1 / 5 * sin(5 * x[1]), rho_left)
    v = ifelse(x[1] > x0, v_right, v_left)
    p = ifelse(x[1] > x0, p_right, p_left)

    return prim2cons(SVector(rho, v, p), equations)
end

function initial_condition_shu_osher(x::Real, t, equations::CompressibleEulerEquations1D)
    x0 = -4

    rho_left = 27 / 7
    v_left = 4 * sqrt(35) / 9
    p_left = 31 / 3

    # Replaced v_right = 0 to v_right = 0.5 to avoid positivity issues.
    v_right = 0.5
    p_right = 1.0

    rho = ifelse(x > x0, 1 + 1 / 5 * sin(5 * x[1]), rho_left)
    v = ifelse(x > x0, v_right, v_left)
    p = ifelse(x > x0, p_right, p_left)

    return Vector(prim2cons(SVector(rho, v, p), equations))
end


# Setup tools for time-stepper of compressible Euler's equation
function setup_euler(
    polydeg,
    cells_per_dimension;
    initial_condition = initial_condition_shu_osher,
)

    gamma_gas = 1.4
    equations = CompressibleEulerEquations1D(gamma_gas)

    ###############################################################################
    # setup the GSBP DG discretization that uses the Gauss operators from 
    # Chan, Del Rey Fernandez, Carpenter (2019). 
    # [https://doi.org/10.1137/18M1209234](https://doi.org/10.1137/18M1209234)

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_ranocha

    basis = DGMultiBasis(Trixi.Line(), polydeg, approximation_type = GaussSBP())

    indicator_sc = IndicatorHennemannGassner(
        equations,
        basis,
        alpha_max = 0.5,
        alpha_min = 0.001,
        alpha_smooth = true,
        variable = density_pressure,
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

    boundary_condition = BoundaryConditionDirichlet(initial_condition)
    boundary_conditions = (; :entire_boundary => boundary_condition)

    ###############################################################################
    #  setup the 1D mesh

    mesh = DGMultiMesh(
        dg,
        (cells_per_dimension,),
        coordinates_min = (-5.0,),
        coordinates_max = (5.0,),
        periodicity = false,
    )

    ###############################################################################
    #  setup the semidiscretization

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        initial_condition,
        dg,
        boundary_conditions = boundary_conditions,
    )

    return TrixiSystem(equations, dg, mesh, semi)
end

function setup_euler_SEM(
    polydeg,
    cells_per_dimension;
    initial_condition = initial_condition_shu_osher,
)

    gamma_gas = 1.4
    equations = CompressibleEulerEquations1D(gamma_gas)

    ###############################################################################
    # setup the GSBP DG discretization that uses the Gauss operators from 
    # Chan, Del Rey Fernandez, Carpenter (2019). 
    # [https://doi.org/10.1137/18M1209234](https://doi.org/10.1137/18M1209234)

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_ranocha

    basis = LobattoLegendreBasis(polydeg)

    indicator_sc = IndicatorHennemannGassner(equations, basis,
                             alpha_max = 0.5,
                             alpha_min = 0.001,
                             alpha_smooth = false,
                             variable = density_pressure)
    
    volume_integral = VolumeIntegralShockCapturingHG(
        indicator_sc;
        volume_flux_dg = volume_flux,
        volume_flux_fv = surface_flux,
    )

    dg = DGSEM(basis, SurfaceIntegralWeakForm(surface_flux), volume_integral)

    
    boundary_condition = BoundaryConditionDirichlet(initial_condition)
    boundary_conditions = (x_neg = boundary_condition, x_pos = boundary_condition)

    ###############################################################################
    #  setup the 1D mesh

    mesh = StructuredMesh((cells_per_dimension,),(Returns(-5.),Returns(5.)),periodicity=false)

    ###############################################################################
    #  setup the semidiscretization

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        initial_condition,
        dg;
        boundary_conditions
    )
    return TrixiSystem(equations, dg, mesh, semi)
end