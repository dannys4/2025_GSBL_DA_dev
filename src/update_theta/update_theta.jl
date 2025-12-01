export update_θ!

function update_θ!(enkf, X, θ::Vector{Float64}, verbose::Bool)
    s = zero(θ)
    # We need to compute the sum of the square
    for i = axes(X, 2)
        X_i = @view X[:, i]
        s .+= (enkf.sys.S * X_i) .^ 2
    end
    # s = vec(sum(abs2, enkf.sys.S * X, dims=1))

    # @show "Need to change the value for the initial condition"

    for j = eachindex(θ, s)
        θ[j] = enkf.flow.ϑ * enkf.flow(√(s[j] / enkf.flow.ϑ))
    end
end

function update_θ!(enkf, X, θ::Matrix{Float64}, verbose::Bool)
    Ne = size(X, 2)
    Ns = size(θ, 1)

    # Make sure that the flow is computed correctly
    sqrt_s = zeros(Ns)

    # We need to compute the square of each component of S x
    for i = 1:Ne
        X_i = @view X[:, i]
        mul!(sqrt_s, enkf.sys.S, X_i)
        for j = 1:Ns
            θ[j, i] = enkf.flow.ϑ * enkf.flow(abs(sqrt_s[j]) / sqrt(enkf.flow.ϑ))
        end
    end
end
