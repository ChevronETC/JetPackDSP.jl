using Jets, JetPack, JetPackDSP, Statistics, Test, Random

@testset "PEF - dot product" begin
    Random.seed!(1234)
    nt, nx, ny = 21, 4, 1
    x = rand(Float64, nt, nx, ny)
    A = JopStreamingPEF1D(x)

    m = rand(domain(A))
    d = rand(range(A))

    @show extrema(A * m)
    @show extrema(A' * d)

    lhs, rhs = dot_product_test(A, m, d)
    @show lhs, rhs
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)

    x = rand(Float64, nt)
    A = JopStreamingPEF1D(x)

    m = rand(domain(A))
    d = rand(range(A))

    @show extrema(A * m)
    @show extrema(A' * d)

    lhs, rhs = dot_product_test(A, m, d)
    @show lhs, rhs
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)
end