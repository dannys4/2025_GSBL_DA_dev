export update_θ!

function update_θ!(enkf, X, θ::Vector{Float64}, ystar, t)

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)
    Ns = size(θ, 1)
    θ0 = copy(θ)

    # Make sure that the flow is computed correctly

    s = zeros(Ns)
    # We need to compute the sum of the square
    #@assert max
    for i = 1:Ne
        s .+= (enkf.sys.S * X[Ny+1:Ny+Nx, i]) .^ 2
    end

    # @show "Need to change the value for the initial condition"

    for j = 1:Ns
        θ[j] = enkf.flow.ϑ * enkf.flow(√(s[j] / enkf.flow.ϑ))
    end
end

function update_θ!(enkf, X, θ::Matrix{Float64}, ystar, t)

    Ny = size(ystar, 1)
    Nx = size(X, 1) - Ny
    Ne = size(X, 2)
    Ns = size(θ, 1)
    θ0 = copy(θ)

    # Make sure that the flow is computed correctly

    s = zeros(Ns)

    # We need to compute the square of each component of S x
    for i = 1:Ne
        s .= (enkf.sys.S * X[Ny+1:Ny+Nx, i]) .^ 2
        for j = 1:Ns
            θ[j, i] = enkf.flow.ϑ * enkf.flow(√(s[j] / enkf.flow.ϑ))
        end
    end
end
