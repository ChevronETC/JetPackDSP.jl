using Jets, JetPack, JetPackDSP, Statistics, Test

@testset "Windowed Correlation - dot product" begin

    nt, nx, ny = 21, 4, 1
    nwin = 4
    x = rand(nt, nx, ny)
    A = JopWindowedCorrelation1D(x; nwin=nwin)

    m = rand(domain(A))
    d = rand(range(A))

    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)
end

@testset "Sliding Correlation - dot product" begin

    nt, nx, ny = 21, 4, 1
    winlen = 4
    skip = 2
    taper = 3
    maxlag = 7
    x = rand(nt, nx, ny)
    A = JopSlidingCorrelation1D(x; winlen=winlen, skip=skip, maxlag=maxlag, taper=taper)

    m = rand(domain(A))
    d = rand(range(A))

    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)
end
