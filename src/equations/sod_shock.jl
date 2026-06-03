export SodShock, sod_solution!
using LinearAlgebra
using Base: vec

Base.@kwdef struct SodState
    rho::Float64
    u::Float64
    P::Float64
end

function Base.vec(state::SodState)
    return [state.rho, state.u, state.P]
end

function P_3_identity(gamma::Float64, P_3::Float64, rho_L::Float64, P_L::Float64, rho_R::Float64, P_R::Float64)
    m2 = ((gamma - 1) / (gamma + 1))
    m4 = m2^2
    lhs = sqrt(
        (( 1 - m4) * (P_L^(1/gamma)) ) / (m4 * rho_L)
    ) * (
        ( P_L^( (gamma - 1)/(2gamma) ) ) - ( P_3^( (gamma - 1)/(2gamma) ) )
    )
    rhs = sqrt(
        (1 - m2) / (rho_R * (P_3 + m2 * P_R))
    ) * (P_3 - P_R)
    return lhs, rhs
end

struct SodShock
    gamma::Float64
    x0::Float64
    state_L::SodState
    state_R::SodState
    P_3::Float64
    function SodShock(;gamma = 1.4, x0 = 0.5, state_L = SodState(1., 0., 1.), state_R = SodState(0.125, 0., 0.1), P_3 = 0.30313)
        p3_lhs, p3_rhs = P_3_identity(gamma, P_3, state_L.rho, state_L.P, state_R.rho, state_R.P)
        if abs2(p3_lhs - p3_rhs)/abs2(p3_rhs) > 1e-8
            throw(ArgumentError("Invalid P3 $(P_3):\nLHS: $(p3_lhs), RHS: $(p3_rhs)"))
        end
        new(gamma, x0, state_L, state_R, P_3)
    end
end

# %%
function constant_C(state::SodState, s::SodShock)
    sqrt(s.gamma * state.P / state.rho)
end

function velocity_4(s::SodShock)
    gamma = s.gamma
    m2 = (gamma - 1) / (gamma + 1)
    P_4 = s.P_3 # identity p_4 = p_3
    rho_R, P_R = s.state_R.rho, s.state_R.P
    return (P_4 - P_R) * sqrt(
        (1 - m2) / (rho_R * (P_4 + m2 * P_R))
    )
end

function density_4(s::SodShock)
    rho_R, P_R = s.state_R.rho, s.state_R.P
    gamma, P_4 = s.gamma, s.P_3 # identity p_4 = p_3
    m2 = (gamma - 1) / (gamma + 1)
    return rho_R * (P_4 + m2 * P_R) / (P_R + m2 * P_4)
end

function region_boundaries(t::Float64, s::SodShock)
    (;x0, gamma) = s
    C_L = constant_C(s.state_L, s)
    u3 = velocity_4(s) # u_3 = u_4 by construction
    x1 = x0 - C_L * t
    x2 = x0 + (((gamma + 1) / 2)*u3 - C_L) * t
    x3 = x0 + u3 * t
    rho_4 = density_4(s)
    u_s = velocity_4(s) * rho_4 / (rho_4 - s.state_R.rho)
    x4 = x0 + u_s * t
    return x1, x2, x3, x4
end

function region_one(_::Float64, _::Float64, s::SodShock)
    return s.state_L
end

function region_two(x::Float64, t::Float64, s::SodShock)
    wave_loc = (x - s.x0) / t
    gamma = s.gamma
    u_2 = 2 * (wave_loc + constant_C(s.state_L, s)) / (gamma + 1)
    rho_L, P_L = s.state_L.rho, s.state_L.P
    rho_2 = (((rho_L^gamma) / (gamma * P_L)) * ((u_2 - wave_loc)^2))^(1 / (gamma - 1))
    P_2 = (rho_2^gamma) * P_L / (rho_L^gamma)
    return SodState(rho = rho_2, u = u_2, P = P_2)
end

function region_three(x::Float64, t::Float64, s::SodShock)
    u_3 = velocity_4(s) # u_3 = u_4 by construction
    P_3 = s.P_3
    rho_3 = s.state_L.rho * ((P_3 / s.state_L.P) ^ (1 / s.gamma))
    return SodState(rho = rho_3, u = u_3, P = P_3)
end

function region_four(x::Float64, t::Float64, s::SodShock)
    u_4 = velocity_4(s)
    P_4 = s.P_3 # p_4 = p_3 by construction
    rho_4 = density_4(s)
    return SodState(rho = rho_4, u = u_4, P = P_4)
end

function region_five(x::Float64, t::Float64, s::SodShock)
    return s.state_R
end

# %%
function sod_solution!(out_vec::Vector{Float64}, xgrid::AbstractVector{Float64}, t::Float64, s::SodShock)
    @assert(issorted(xgrid))
    out = reshape(out_vec, 3, length(xgrid))#Matrix{Float64}(undef, 3, length(xgrid))
    bc_L, bc_R = vec(s.state_L), vec(s.state_R)
    if t <= eps()
        x0_bdry = findfirst(xgrid .> s.x0)
        out[:,1:(x0_bdry-1)] .= bc_L
        out[:,x0_bdry:end] .= bc_R
        return out
    end
    bdry = region_boundaries(t, s)
    if 0. > bdry[1] || bdry[1] > bdry[2] || bdry[2] > bdry[3] || bdry[3] > bdry[4] || bdry[4] > 1.0
        throw(ArgumentError("Invalid region boundaries $bdry at time t=$(t)!"))
    end
    start_2 = findfirst(xgrid .> bdry[1])
    start_3 = findfirst(xgrid .> bdry[2])
    start_4 = findfirst(xgrid .> bdry[3])
    start_5 = findfirst(xgrid .> bdry[4])
    region_1, region_2, region_3, region_4, region_5 = 1:(start_2-1), start_2:(start_3-1), start_3:(start_4-1), start_4:(start_5-1), start_5:length(xgrid)
    out[:, region_1] .= bc_L
    out[:, region_2] .= reduce(hcat, vec(region_two(x, t, s)) for x in xgrid[region_2])
    out[:, region_3] .= reduce(hcat, vec(region_three(x, t, s)) for x in xgrid[region_3])
    out[:, region_4] .= reduce(hcat, vec(region_four(x, t, s)) for x in xgrid[region_4])
    out[:, region_5] .= bc_R
    return out
end


function example_sod_solution_eval()
    gamma, x0 = 1.4, 0.5
    x_L, x_R = 0., 1.
    N_X = 500
    xgrid = range(x_L, x_R, N_X + 1)
    s = SodShock(gamma, x0, SodState(1., 0., 1.), SodState(0.125, 0., 0.1), 0.30310)

    t_0 = 0.1
    out_sod = Matrix{Float64}(undef, 3, length(xgrid))
    sod_solution!(vec(out_sod), xgrid, t_0, s)
end