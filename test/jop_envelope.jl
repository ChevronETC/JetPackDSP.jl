using DSP, JetPackDSP, Jets, Test

n1,n2,n3 = 500,5,7

@testset "Envelope^($(power)) calculation, dimension=$(dim), T=$(T)" for power in (0.25, 0.5, 1.0), T in (Float64,Float32), dim in (2,3)
    space = dim == 2 ? JetSpace(T, n3, n1) : JetSpace(T, n3, n2, n1)
    F = JopEnvelope(space, power)
    x1 = -1 .+ 2 .* rand(domain(F))
    y1 = F*x1
    y2 = (x1.^2 + imag.(hilbert(x1)).^2).^(power * 0.5)
    @test y1 ≈ y2
end

@testset "Envelope^($(power)) linearity test, dimension=$(length(N)), T=$(T)" for power in (0.25, 0.5, 1.0), T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    N=(n1,n2)
    T=Float64
    pwr = 0.25
    F = JopEnvelope(JetSpace(T,N), pwr)
    J  = jacobian!(F, -1 .+ 2 .* rand(domain(F)))

    #  check: J m1 + J m2 - J m3 = J (m1 + m2 - m3)
    m1 = -1 .+ 2 .* rand(domain(F))
    m2 = -1 .+ 2 .* rand(domain(F))
    m3 = -1 .+ 2 .* rand(domain(F))
    @test J*m1 .+ J*m2 .- J*m3 ≈ J*(m1 .+ m2 .- m3)

    #  check: J' d1 + J' d2 -J' d3 = J' (d1 + d2 - d3)
    d1 = -1 .+ 2 .* rand(range(F))
    d2 = -1 .+ 2 .* rand(range(F))
    d3 = -1 .+ 2 .* rand(range(F))
    @test J'*d1 .+ J'*d2 .- J'*d3 ≈ J'*(d1 .+ d2 .- d3)
end

@testset "Envelope^($(power)) dot product test, dimension=$(length(N)), T=$(T)" for power in (0.25, 0.5, 1.0), T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    F = JopEnvelope(JetSpace(T,N))
    J  = jacobian!(F, -1 .+ 2 .* rand(domain(F)))
    lhs, rhs = dot_product_test(J, -1 .+ 2 .* rand(domain(F)), -1 .+ 2 .* rand(range(F)))
    @test lhs ≈ rhs
end

@testset "Envelope^($(power)) dot product test with zeros, dimension=$(length(N)), T=$(T)" for power in (0.25, 0.5, 1.0), T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
        op = JopEnvelope(JetSpace(T,N),power,damping=eps(T))
        m0 = -1 .+ 2 .* rand(domain(op))
        m0[1:n1] .= T(0.0)
        J  = jacobian!(op, m0)
        lhs, rhs = dot_product_test(J, -1 .+ 2 .* rand(domain(op)), -1 .+ 2 .* rand(range(op)))
        @test isapprox(lhs, rhs, rtol=1e-4) # TODO... why the bigger rtol?
end

@test_skip @testset "Envelope^($(power)) linearization test, dimension=$(length(N)), T=$(T)" for power in (0.25, 0.5, 1.0), T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
    T = Float32
    N =  (n1,n2,n3)
    F = JopEnvelope(JetSpace(T,N))
    m0 = -1 .+ 2 .* rand(domain(F))

    mu = .1 * sqrt.([1.0,1.0/2.0,1.0/4.0,1.0/8.0,1.0/16.0,1.0/32.0,1.0/64.0,1.0/128.0,1.0/256.0,1.0/512.0])
    observed, expected = linearization_test(F, m0, μ=mu)
        @show observed - expected
    
    δ = minimum(abs, observed - expected)
    @test δ < .2
end

@test_skip @testset "Envelope^($(power)) linearization test with zeros, dimension=$(length(N)), T=$(T)" for power in (0.25, 0.5, 1.0), T in (Float64,Float32), N in ( (n1,n2), (n1,n2,n3) )
        op = JopEnvelope(JetSpace(T,N),power)
        m0 = -1 .+ 2 .* rand(domain(op))
        m0[1:n1] .= T(0.0)
        mu = .1 * sqrt.([1.0,1.0/2.0,1.0/4.0,1.0/8.0,1.0/16.0,1.0/32.0,1.0/64.0,1.0/128.0,1.0/256.0,1.0/512.0])
        
        observed, expected = linearization_test(op, m0, μ=mu)
        @show observed - expected
        δ = minimum(abs, observed - expected)
        @test δ < .2
    end

nothing
