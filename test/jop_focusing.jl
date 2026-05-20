using Jets, JetPack, JetPackDSP, Statistics, Test

PLOTS = parse(Int32, get(ENV, "JPDSP_PLOTS", "0"))

function _wavelet(n, T, tc, s)
    t = range(-T, T; length=n)
    window = zeros(length(t))
    inside = abs.(t .- tc) .<= T
    window[inside] .= 0.5 .* (1 .+ cos.(π .* (t[inside] .- tc) ./ T))
    sinc.((t .- tc) ./ s) .* window
end

@testset "Focusing - correctness" begin
    n = 101
    T  = 1.0
    dt = 2 * T / (n - 1)
    t = range(-T, T; length=n)
    s  = 0.6
    tc = [0.0, 0.2,-0.3]
    w1 = _wavelet(n, T, tc[1], s)
    w2 = -_wavelet(n, T, tc[2], s)
    w3 = _wavelet(n, T, tc[3], s)
    
    m = stack([w1,w2,w3]; dims=2)

    spc = JetSpace(Float64, n, 3)
    A = JopFocusing1D(spc)

    d = A * m

    for i = 1:3
        @test abs((argmax(abs.(d[:,i])) - 1) * dt - T/2)  <= abs(t[i] - T/2)
    end

    @show size(t)
    @show size(m)
    @show size(d)

    if PLOTS > 0
        using PyPlot
        close("all")
        figure(figsize=(8,4), dpi=100)
        colors = ["blue", "green", "red"]
        for i = 1:3
            plot(t,m[:,i], label="input", color=colors[i], linewidth=i, linestyle="--")
            plot(t,d[:,i], label="output", color=colors[i], linewidth=i, linestyle=":")
        end
        legend()
        PyPlot.grid()
        tight_layout()
        savefig("jop_focusing_verification.png")
    end
end

@testset "Focusing - dot product - conserve_energy = $(conserve_energy)" for conserve_energy in (false, true)
    nt, nx, ny = 21, 4, 1
    spc = JetSpace(Float64, nt, nx, ny)
    A = JopFocusing1D(spc; conserve_energy=conserve_energy)

    m = rand(domain(A))
    d = rand(range(A))

    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox((lhs - rhs)/(lhs + rhs), 0.0, atol=1e-7)
end