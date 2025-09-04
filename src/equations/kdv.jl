export KDV_Fourier, spectral_kdv!

struct KDV_Fourier{Q,R,S,U}
    L::Float64
    Nx::Int64
    Δx::Float64
    grid::Vector{Float64}
    Space::Q
    T::R
    T_mul!::Function
    T_mul::Function
    Tinv::S
    Tinv_mul!::Function
    Tinv_mul::Function
    D::BandedMatrices.BandedMatrix
    # D2::BandedMatrices.BandedMatrix
    D3::BandedMatrices.BandedMatrix
    δ::Float64
    # Store the diagonal operator -D^3 for the integrating factor
    mD3::U
    cache::Vector{Float64}
    u::Vector{Float64}
end

function KDV_Fourier(L::Float64, Nx::Int64)
    Δx = 1 / Nx
    Space = Fourier(Interval(0, L))
    grid = points(Space, Nx)
    # D2 = Derivative(Space,2)[1:Nx,1:Nx]
    D = (Derivative(Space)→Space)[1:Nx, 1:Nx]
    T = ApproxFun.plan_transform(Space, Nx)
    Tinv = ApproxFun.plan_itransform(Space, Nx)

    # Define multiplication routines 
    # (needed to bypass distribution of FFTW plans in Distributed computing)
    T_mul!(y, x) = mul!(y, T, x)
    T_mul(x) = T_mul!(zero(x), x)
    Tinv_mul!(y, x) = mul!(y, Tinv, x)
    Tinv_mul(x) = Tinv_mul!(zero(x), x)

    D3 = Derivative(Space, 3)[1:Nx, 1:Nx]

    mD3 = DiffEqArrayOperator(-Diagonal(D3))

    cache = zeros(Nx)
    u = zeros(Nx)

    return KDV_Fourier{typeof(Space),typeof(T),typeof(Tinv),typeof(mD3)}(
        L,
        Nx,
        Δx,
        grid,
        Space,
        T,
        T_mul!,
        T_mul,
        Tinv,
        Tinv_mul!,
        Tinv_mul,
        D,
        D3,
        mD3,
        cache,
        u,
    )
end


function Base.show(io::IO, p::KDV_Fourier)
    println(io, "Parameters for Korteweg-de Vries model on [0, $(p.L)] with $(p.Nx) points")
end

# from https://docs.sciml.ai/SciMLBenchmarksOutput/stable/SimpleHandwrittenPDE/kdv_spectral_wpd/

function spectral_kdv!(dû, û, p::KDV_Fourier, t)
    T_mul! = p.T_mul!
    Tinv_mul! = p.Tinv_mul!
    D = p.D
    cache = p.cache
    u = p.u
    mul!(u, D, û)
    # mul!(tmp,Ti,u)
    Tinv_mul!(cache, u)
    # mul!(u,Ti,û)
    Tinv_mul!(u, û)
    @. cache = u * cache
    # mul!(u,T,cache)
    T_mul!(u, cache)
    @.dû = 6 * u
end
