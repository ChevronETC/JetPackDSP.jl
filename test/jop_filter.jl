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
