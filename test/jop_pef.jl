using Jets, JetPack, JetPackDSP, Statistics, Test, Random, LinearAlgebra

@test_skip @testset "PEF - dot product" begin
    Random.seed!(1234)
    nt, nx, ny = 21, 4, 1
    dom = JetSpace(Float64, (nt, nx, ny))
    A = JopNlStreamingPEF1D(; dom)

    m0 = rand(domain(A))
    m = rand(domain(A))
    d = rand(range(A))

    J = jacobian!(A, m0)

    @show extrema(J * m)
    @show extrema(J' * d)

    lhs, rhs = dot_product_test(J, m, d)
    @show lhs, rhs
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)

    dom = JetSpace(Float64, (nt))
    A = JopNlStreamingPEF1D(; dom)

    m0 = rand(domain(A))
    m = rand(domain(A))
    d = rand(range(A))

    J = jacobian!(A, m0)

    @show extrema(J * m)
    @show extrema(J' * d)

    lhs, rhs = dot_product_test(J, m, d)
    @show lhs, rhs
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)
end

@test_skip @testset "PEF - linearization - λ = $λ " for λ in (1e-3, 1)
    Random.seed!(1234)
    nt = 51
    dom = JetSpace(Float64, (nt,))
    A = JopNlStreamingPEF1D(; dom, λ=λ)
    m0 = rand(domain(A))
    δm = rand(domain(A))

    Fm0 = A * m0
    J = jacobian!(A, m0)
    Jδm = J * δm

    μs = [1.0, 0.1, 0.01, 0.001, 0.0001]
    e0 = [norm(A*(m0 .+ μ .* δm) -  Fm0)                for μ in μs]   # O(μ)
    e1 = [norm(A*(m0 .+ μ .* δm) .- Fm0 .- μ .* Jδm)    for μ in μs]   # O(μ²)

    @show e0
    @show e1

    # Check that e0 decays linearly
    rate = log2(e0[1] / e0[end]) / log2(μs[1] / μs[end])
    @show rate
    @test abs(rate - 1) < 0.1

    # Check that e1 decays quadratically
    rate = log2(e1[1] / e1[end]) / log2(μs[1] / μs[end])
    @show rate
    @test abs(rate - 2) < 0.1
end

@testset "PEF - gradient" begin
    Random.seed!(1234)
    nt = 51
    dom = JetSpace(Float64, (nt,))

    λs = [1.0, 0.1, 0.01, 0.001, 0.0001]
    e = []

    for λ in λs
        A = JopNlStreamingPEF1D(; dom, λ=λ)
        m = rand(domain(A))

        J = jacobian!(A, m)
        g = J' * (A * m)

        function misfit(x)
            0.5 * norm((A * x))^2
        end

        μs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        g_fd = zeros(nt, length(μs))
        mi = zeros(nt)
        phi0 = misfit(m)
        for ie = 1:length(μs)
            for i = 1:nt
                mi .= m
                mi[i] += μs[ie]
                phi = misfit(mi)
                g_fd[i,ie] = (phi - phi0) / μs[ie]
            end
        end

        error = [norm(g .- g_fd[:,ie]) ./ (norm(g) + norm(g_fd[:,ie])) for ie in 1:length(μs)]
        push!(e, minimum(error))
    end
    @show e
    @test all(e .< 1e-4)
end