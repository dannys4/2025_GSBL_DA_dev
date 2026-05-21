export initial_condition_shu_osher, initial_condition_sod, setup_euler


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

    rho = ifelse(x[] > x0, 1 + 1 / 5 * sin(5 * x[1]), rho_left)
    v = ifelse(x[] > x0, v_right, v_left)
    p = ifelse(x[] > x0, p_right, p_left)

    return prim2cons(SVector(rho, v, p), equations)
end

function initial_condition_sod(
    x,
    t,
    equations::CompressibleEulerEquations1D,
    u_L=(rho=1., v=0., p=1.),
    u_R=(rho=0.125, v=0., p=0.1),
    x0=0.5
)
    GAMMA = 1.4
    @assert equations.gamma == GAMMA
    rho = ifelse(x[] < x0, u_L.rho, u_R.rho)
    v = ifelse(x[] < x0, u_L.v, u_R.v)
    p = ifelse(x[] < x0, u_L.p, u_R.p)
    return prim2cons(SVector(rho, v, p), equations)
end


# Setup tools for time-stepper of compressible Euler's equation
function setup_euler(
    polydeg,
    cells_per_dimension;
    initial_condition=:shu_osher,
    bcs = :dirichlet
)
    if !(initial_condition in [:shu_osher, :sod])
        throw(ArgumentError("Unexpected initial condition $initial_condition"))
    end
    if !(bcs in [:zero_neumann, :dirichlet, nothing])
        throw(ArgumentError("Unexpected boundary conditions $bcs"))
    end
    gamma_gas = 1.4
    equations = CompressibleEulerEquations1D(gamma_gas)

    ###############################################################################
    # setup the GSBP DG discretization that uses the Gauss operators from
    # Chan, Del Rey Fernandez, Carpenter (2019).
    # [https://doi.org/10.1137/18M1209234](https://doi.org/10.1137/18M1209234)

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_ranocha

    basis = DGMultiBasis(Trixi.Line(), polydeg, approximation_type=GaussSBP())

    indicator_sc = IndicatorHennemannGassner(
        equations,
        basis,
        alpha_max=0.5,
        alpha_min=0.001,
        alpha_smooth=true,
        variable=density_pressure,
    )
    volume_integral = VolumeIntegralShockCapturingHG(
        indicator_sc;
        volume_flux_dg=volume_flux,
        volume_flux_fv=surface_flux,
    )

    dg = DGMulti(
        basis,
        surface_integral=SurfaceIntegralWeakForm(surface_flux),
        volume_integral=volume_integral,
    )

    initial_condition_fcn = nothing
    if initial_condition == :shu_osher
        initial_condition_fcn = initial_condition_shu_osher
    elseif initial_condition == :sod
        initial_condition_fcn = initial_condition_sod
    else
        throw(ArgumentError("Unknown initial condition $(initial_condition)"))
    end
    boundary_condition = if bcs == :dirichlet
        BoundaryConditionDirichlet(initial_condition_fcn)
    elseif bcs == :zero_neumann
        BoundaryConditionNeumann(Returns(@SVector[0., 0., 0.]))
    elseif isnothing(bcs)
        boundary_condition_do_nothing
    else
        throw(ArgumentError("Unknown boundary condition $bcs"))
    end

    boundary_conditions = (; :entire_boundary => boundary_condition)

    ###############################################################################
    #  setup the 1D mesh
    coordinates_min = coordinates_max = nothing
    if initial_condition == :shu_osher
        coordinates_min, coordinates_max = -5., 5.
    elseif initial_condition == :sod
        coordinates_min, coordinates_max = 0., 1.
    end
    mesh = DGMultiMesh(
        dg,
        (cells_per_dimension,),
        coordinates_min=(coordinates_min,),
        coordinates_max=(coordinates_max,),
        periodicity=false,
    )

    ###############################################################################
    #  setup the semidiscretization

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        initial_condition_fcn,
        dg;
        boundary_conditions
    )

    return TrixiSystem(equations, dg, mesh, semi)
end

function setup_euler_SEM(
    polydeg,
    cells_per_dimension;
    initial_condition=initial_condition_shu_osher,
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
        alpha_max=0.5,
        alpha_min=0.001,
        alpha_smooth=false,
        variable=density_pressure)

    volume_integral = VolumeIntegralShockCapturingHG(
        indicator_sc;
        volume_flux_dg=volume_flux,
        volume_flux_fv=surface_flux,
    )

    dg = DGSEM(basis, SurfaceIntegralWeakForm(surface_flux), volume_integral)


    boundary_condition = BoundaryConditionDirichlet(initial_condition)
    boundary_conditions = (x_neg=boundary_condition, x_pos=boundary_condition)

    ###############################################################################
    #  setup the 1D mesh

    mesh = StructuredMesh((cells_per_dimension,), (Returns(-5.), Returns(5.)), periodicity=false)

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