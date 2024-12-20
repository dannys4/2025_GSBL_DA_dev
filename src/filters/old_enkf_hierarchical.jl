# import TransportBasedInference: SeqFilter

# export EnKF

# """
# $(TYPEDEF)

# A structure for the likelihood-based ensemble Kalman filter (EnKF)

# References:

# $(TYPEDFIELDS)
# """

# struct EnKF<:SeqFilter
#     "Filter function"
#     G::Function

#     "Standard deviations of the measurement noise distribution"
#     ϵy::InflationType

#     "Boolean: is state vector filtered"
#     isfiltered::Bool
# end

# function EnKF(G::Function, ϵy::InflationType; isfiltered = false)
#     return EnKF(G, ϵy, isfiltered)
# end

# # If no filtering function is provided, use the identity in the constructor.
# function EnKF(ϵy::InflationType)
#     return EnKF(x -> x, ϵy, false)
# end



# function Base.show(io::IO, enkf::EnKF)
# 	println(io,"Hierarchical likelihood-based EnKF  with filtered = $(enkf.isfiltered)")
# end

# function (enkf::EnKF)(X, ystar::Array{Float64,1}, t::Float64)

#     # We add a Tikkonov regularization to ensure that CY is positive definite
#     Ny = size(ystar,1)
#     Nx = size(X,1)-Ny
#     Ne = size(X, 2)

#     # Generate observational noise samples
#     E = zeros(Ny, Ne)
#     if typeof(enkf.ϵy) <: AdditiveInflation
#         E .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
#     end

#     # Estimate the mean and covariance of the joint distribution
#     X[1:Ny,:] .+= E

#     ΣYX = cov(X')
                                                         
#     # Regularized covarariance of the observations Y
#     ΣY = factorize(Symmetric(ΣYX[1:Ny, 1:Ny]))

#     # Cross scale matrix between X and Y
#     ΣXcrossY = ΣYX[Ny+1:Ny+Nx,1:Ny]

#     for i=1:Ne
#         xi = X[Ny+1:Ny+Nx,i]
#         yi = X[1:Ny,i]
        
#         bi = ΣY \ (yi - ystar)
#         X[Ny+1:Ny+Nx,i] =  xi - ΣXcrossY*bi
#     end

# 	return X
# end 
