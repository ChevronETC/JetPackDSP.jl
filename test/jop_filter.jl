using DSP, JetPackDSP, Jets, Test

n1,n2,n3 = 80,70,10

@testset "JopFilter linearity test, Lowpass, dimension=$(length(N)), T=$(T)" for T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    A = JopFilter(JetSpace(T,N), Lowpass(T(0.25)), Butterworth(T,5))
    lhs, rhs = linearity_test(A)
    @test lhs ≈ rhs
    lhs, rhs = linearity_test(A')
    @test lhs ≈ rhs
end

@testset "JotOpFilter linearity test, Highpass, dimension=$(length(N)), T=$(T)" for T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    A = JopFilter(JetSpace(T,N), Highpass(T(0.25)), Butterworth(T,5))
    lhs, rhs = linearity_test(A)
    @test lhs ≈ rhs
    lhs, rhs = linearity_test(A')
    @test lhs ≈ rhs
end

@testset "JotOpFilter linearity test, Bandpass, dimension=$(length(N)), T=$(T)" for T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    A = JopFilter(JetSpace(T,N), Bandpass(T(0.2),T(0.3)), Butterworth(T,5))
    lhs, rhs = linearity_test(A)
    @test lhs ≈ rhs
    lhs, rhs = linearity_test(A')
    @test lhs ≈ rhs
end

@testset "JotOpFilter dot product test, Lowpass, dimension=$(length(N)), T=$(T)" for T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    A = JopFilter(JetSpace(T,N), Lowpass(T(0.1)), Butterworth(5))
    lhs,rhs = dot_product_test(A, -1 .+ 2 .* rand(domain(A)), -1 .+ 2 .* rand(range(A)))
    @test lhs ≈ rhs
end

@testset "JotOpFilter dot product test, Highpass, dimension=$(length(N)), T=$(T)" for T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    A = JopFilter(JetSpace(T,N), Highpass(T(0.4)), Butterworth(5))
    lhs,rhs = dot_product_test(A, -1 .+ 2 .* rand(domain(A)), -1 .+ 2 .* rand(range(A)))
    @test lhs ≈ rhs
end

@testset "JotOpFilter dot product test, Bandpass, dimension=$(length(N)), T=$(T)" for T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    A = JopFilter(JetSpace(T,N), Bandpass(T(0.1),T(0.4)), Butterworth(5))
    lhs,rhs = dot_product_test(A, -1 .+ 2 .* rand(domain(A)), -1 .+ 2 .* rand(range(A)))
    @test lhs ≈ rhs
end

@testset "JotOpFilter dot product test, explicit time filter, nh=$nh" for nh in (n1,2*n1,div(n1,2), div(n1,2)+1)
    N = (n1,n2,n3)
    T = Float64
    A = JopFilter(JetSpace(T,N), randn(T, nh))
    lhs,rhs = dot_product_test(A, -1 .+ 2 .* rand(domain(A)), -1 .+ 2 .* rand(range(A)))
    @test lhs ≈ rhs
end

@testset "JotOpFilter parity test, analytical vs explicit time filter" begin
    n = 81
    T = Float64
    responsetype = Lowpass(T(0.25))
    designmethod = Butterworth(T,5)
    fil = zeros(T,n)
	fil[div(n,2)+1] = 1
	fil .= filtfilt(digitalfilter(responsetype, designmethod), fil)
    A = JopFilter(JetSpace(T,n), responsetype, designmethod)
    B = JopFilter(JetSpace(T,n), fil)
    m = -1 .+ 2 .* rand(domain(A))
    dA = A*m
    dB = B*m
    error = sum((dA-dB).^2) / sum(dA.^2)
    @show error
    @test error < 1e-8   
end

@testset "JotOpFilter dot product test, 1D domain with ND filtering" begin
    N = (n1,n2,n3)
    T = Float64
    v = randn(T, N)
    A = JopFilter(v)
    lhs,rhs = dot_product_test(A, -1 .+ 2 .* rand(domain(A)), -1 .+ 2 .* rand(range(A)))
    @test lhs ≈ rhs
end

@testset "JotOpFilter parity test, 1D domain with ND filtering vs ND domain with 1D filtering" begin
    N = (n1,n2,n3)
    T = Float64
    fil = randn(T, n1)
    v = randn(T, N)
    A = JopFilter(JetSpace(T,N), fil)
    B = JopFilter(v)
    dA = A*v
    dB = B*fil
    error = sum((dA-dB).^2) / sum(dA.^2)
    @show error
    @test error < 1e-10
end
