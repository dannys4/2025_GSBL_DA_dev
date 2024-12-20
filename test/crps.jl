


@testset "Test CRPS implementation" begin
    
    μX = 0.1
    σX = 0.2
    πX = Normal(μX, σX)

    xstar = 0.3

    X = rand(πX, 10000)

    @test isapprox(CRPS_gaussian(πX, xstar), CRPS_quadrature(X, xstar), atol =1e-2)
    @test isapprox(CRPS_gaussian(πX, xstar), CRPS(X, xstar), atol =1e-2)
end