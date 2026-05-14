using Jets, JetPack, JetPackDSP, Statistics, Test

@testset "Focusing - dot product" begin

    nt, nx, ny = 21, 4, 1
    spc = JetSpace(Float64, nt, nx, ny)
    A = JopFocusing1D(spc)

    m = rand(domain(A))
    d = rand(range(A))

    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)
end