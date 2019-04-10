using Jets, JetPack, JetPackDSP, Statistics, Test

@testset "convolve, 1D" begin
    m = sin.([0:0.004:511*0.004;] * 2 * pi * 2.0)

    A = JopConvolve(JetSpace(Float64, 512), JetSpace(Float64, 512), [0.5], x0=0.0, dx=0.004)
    d = A * m
    @test (0.5*m) ≈ (d)

    A = JopConvolve(JetSpace(Float64, 512), JetSpace(Float64, 512), [2.0], x0=-10*0.004, dx=0.004)
    d = A * m
    for i=1:502
        @test isapprox(2.0*m[10+i], d[i], atol=1e-12)
    end
    for i=1:9
        @test isapprox(2.0*m[i], d[502+i], atol=1e-12)
    end

    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-10)
end

@testset "convolve, 2D" begin
    A = JopConvolve(JetSpace(Float64,128,128), rand(Float64,32,32), x0=(-16.,-16.), dx=(1.,1.))
    lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
    @test lhs ≈ rhs
end

@testset "convolve smoothing, 2D-2D" begin
    # even
    P = JopPad(JetSpace(Float64,128,128), -10:128+11, -10:128+11, extend=true)
    S = JopConvolve(range(P), smoother=:rect, n=(1,1))
    R = JopPad(domain(P), -10:128+11, -10:128+11, extend=false)
    @test iseven(size(range(P),1))
    @test iseven(size(range(P),2))

    m = rand(domain(P))
    d = (R' ∘ S ∘ P)*m
    for i = 2:127, j = 2:127
        @test d[i,j] ≈ mean(m[i-1:i+1,j-1:j+1])
    end
    @test sum(d) ≈ sum(m)

    # odd
    P = JopPad(JetSpace(Float64,128,128), -7:128+9, -7:128+9, extend=true)
    S = JopConvolve(range(P), smoother=:rect, n=(1,1))
    R = JopPad(domain(P), -7:128+9, -7:128+9, extend=false)
    @test isodd(size(range(P),1))
    @test isodd(size(range(P),2))

    m = rand(domain(P))
    d = (R' ∘ S ∘ P)*m
    for i = 2:127, j = 2:127
        @test d[i,j] ≈ mean(m[i-1:i+1,j-1:j+1])
    end
    @test sum(d) ≈ sum(m)
end

@testset "smoothing, 2D-1D" begin
    # even
    P = JopPad(JetSpace(Float64,128,128), -10:128+11, -10:128+11, extend=true)
    S = JopConvolve(range(P), smoother=:rect, n=(1,0))
    R = JopPad(domain(P), -10:128+11, -10:128+11, extend=false)
    @test iseven(size(range(P),1))
    @test iseven(size(range(P),2))

    m = rand(domain(P))
    d = (R' ∘ S ∘ P)*m
    for i = 2:127, j = 2:127
        @test d[i,j] ≈ mean(m[i-1:i+1,j])
    end
    @test sum(d) ≈ sum(m)
    @test sum(d,dims=1) ≈ sum(m,dims=1)

    # odd
    P = JopPad(JetSpace(Float64,128,128), -7:128+9, -7:128+9, extend=true)
    S = JopConvolve(range(P), smoother=:rect, n=(1,0))
    R = JopPad(domain(P), -7:128+9, -7:128+9, extend=false)
    @test isodd(size(range(P),1))
    @test isodd(size(range(P),2))

    m = rand(domain(P))
    d = (R' ∘ S ∘ P)*m
    for i = 2:127, j = 2:127
        @test d[i,j] ≈ mean(m[i-1:i+1,j])
    end
    @test sum(d) ≈ sum(m)
    @test sum(d,dims=1) ≈ sum(m,dims=1)

    # even
    P = JopPad(JetSpace(Float64,128,128), -10:128+11, -10:128+11, extend=true)
    S = JopConvolve(range(P), smoother=:rect, n=(0,1))
    R = JopPad(domain(P), -10:128+11, -10:128+11, extend=false)
    @test iseven(size(range(P),1))
    @test iseven(size(range(P),2))

    m = rand(domain(P))
    d = (R' ∘ S ∘ P)*m
    for i = 2:127, j = 2:127
        @test d[i,j] ≈ mean(m[i,j-1:j+1])
    end
    @test sum(d) ≈ sum(m)
    @test sum(d,dims=2) ≈ sum(m,dims=2)

    # odd
    P = JopPad(JetSpace(Float64,128,128), -7:128+9, -7:128+9, extend=true)
    S = JopConvolve(range(P), smoother=:rect, n=(0,1))
    R = JopPad(domain(P), -7:128+9, -7:128+9, extend=false)
    @test isodd(size(range(P),1))
    @test isodd(size(range(P),2))

    m = rand(domain(P))
    d = (R' ∘ S ∘ P)*m
    for i = 2:127, j = 2:127
        @test d[i,j] ≈ mean(m[i,j-1:j+1])
    end
    @test sum(d) ≈ sum(m)
    @test sum(d,dims=2) ≈ sum(m,dims=2)
end
