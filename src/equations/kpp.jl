export initial_condition_kpp, setup_kpp


###############################################################################
# Definition of the 2D scalar "KPP equations"
#
# See: Kurganov, A., Petrova, G., and Popov, B. (2007).
# Adaptive Semidiscrete Central-Upwind Schemes for Nonconvex Hyperbolic Conservation Laws,
# SIAM Journal on Scientific Computing, 29(6), 2381--2401
# DOI: https://doi.org/10.1137/040614189

struct KPPEquation2D <: Trixi.AbstractEquations{2,1} end


# The KPP flux is F(u) = (sin(u), cos(u))
@inline function Trixi.flux(u, orientation::Integer, ::KPPEquation2D)
    if orientation == 1
        return SVector(sin(u[1]))
    else
        return SVector(cos(u[1]))
    end
end

# Since the KPP problem is a scalar equations, the entropy-conservative flux is uniquely determined
@inline function Trixi.flux_ec(u_ll::StaticVector{1,RealT}, u_rr::StaticVector{1,RealT}, orientation::Integer, ::KPPEquation2D) where {RealT}
    # The tolerance of 1e-12 is based on experience and somewhat arbitrarily chosen
    u_ll, u_rr = u_ll[], u_rr[]
    if abs(u_ll - u_rr) < RealT(1e-12)
        return 0.5f0 * (flux(u_ll, orientation, KPPEquation2D()) +
                        flux(u_rr, orientation, KPPEquation2D()))
    else
        factor = 1 / (u_rr - u_ll)
        if orientation == 1
            return SVector(factor * (-cos(u_rr) + cos(u_ll)))
        else
            return SVector(factor * (sin(u_rr) - sin(u_ll)))
        end
    end
end

@inline function normal_to_orientation(normal_vec::SVector{2})
    # @assert abs2(normal_vec[1]) < 1e-10 || abs2(normal_vec[2]) < 1e-10
    which_coord = 1 + Int(abs2(normal_vec[1]) < abs2(normal_vec[2]))
    which_coord, normal_vec[which_coord]
end

@inline function Trixi.flux(u, normal_vec::SVector{2}, eq::KPPEquation2D)
    orientation, scaling = normal_to_orientation(normal_vec)
    flux(u, orientation, eq) * scaling
end

@inline function Trixi.flux_ec(u_ll, u_rr, normal_vec::SVector{2}, eq::KPPEquation2D)
    orientation, scaling = normal_to_orientation(normal_vec)
    flux_ec(u_ll, u_rr, orientation, eq) * scaling
end

# Wavespeeds
# Please note that the return type should be modified if other
# floating point types should be used.
@inline wavespeed(::KPPEquation2D) = 1.0
@inline Trixi.max_abs_speeds(u, equations::KPPEquation2D) = (wavespeed(equations),
    wavespeed(equations))
@inline Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::KPPEquation2D) = wavespeed(equations)
@inline Trixi.max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::KPPEquation2D) = wavespeed(equations) *
                                                                                                            norm(normal_direction)

# Compute entropy: we use the square entropy
@inline Trixi.entropy(u::Real, ::KPPEquation2D) = 0.5f0 * u^2
@inline Trixi.entropy(u, ::KPPEquation2D) = entropy(u[1], equations)

# Convert between conservative, primitive, and entropy variables. The conserved quantity "u" is also
# considered the "primitive variable". Since we use the square entropy, "u" is also the entropy
# variable.
@inline Trixi.prim2cons(u, ::KPPEquation2D) = u
@inline Trixi.cons2prim(u, ::KPPEquation2D) = u
@inline Trixi.cons2entropy(u, ::KPPEquation2D) = u
@inline Trixi.entropy2cons(u, ::KPPEquation2D) = u

Trixi.varnames(::Any, ::KPPEquation2D) = ("u",)

# Standard KPP test problem with discontinuous initial condition
function initial_condition_kpp(x, t, ::KPPEquation2D)
    RealT = eltype(x)
    if x[1]^2 + x[2]^2 < 1
        return SVector(0.25f0 * 14 * convert(RealT, pi))
    else
        return SVector(0.25f0 * convert(RealT, pi))
    end
end

# Setup tools for time-stepper of KPP equation
function setup_kpp(
    polydeg,
    Ncells_dim;
    initial_condition=initial_condition_kpp,
    coordinates_min=(-2., -2.),
    coordinates_max=(2., 2.)
)

    # semidiscretization of the KPP problem
    equations = KPPEquation2D()
    surface_flux = flux_lax_friedrichs
    volume_flux = flux_ec
    cells_per_dimension = (Ncells_dim, Ncells_dim)

    basis = DGMultiBasis(Trixi.Quad(), polydeg, approximation_type=GaussSBP())

    indicator_sc = IndicatorHennemannGassner(
        equations,
        basis,
        alpha_max=0.5,
        alpha_min=0.001,
        alpha_smooth=true,
        variable=first,
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

    dg_mesh = DGMultiMesh(dg, cells_per_dimension; periodicity=true, coordinates_min, coordinates_max)
    semi = SemidiscretizationHyperbolic(dg_mesh, equations, initial_condition, dg)
    return TrixiSystem(equations, dg, dg_mesh, semi)
end