using Jets, JetPack, JetPackDSP, Statistics, Test, LinearAlgebra

PLOTS = parse(Int32, get(ENV, "JPDSP_PLOTS", "0"))

function _wavelet(n, T, tc, s)
    t = range(-T, T; length=n)
    window = zeros(length(t))
    inside = abs.(t .- tc) .<= T
    window[inside] .= 0.5 .* (1 .+ cos.(π .* (t[inside] .- tc) ./ T))
    sinc.((t .- tc) ./ s) .* window
end

@testset "Windowed Correlation - correctness" begin
    n = 101
    T  = 1.0
    dt = 2 * T / (n - 1)
    t = range(-T, T; length=n)
    s  = 0.6
    to = [0.0, 0.2, 0.0]
    tm = [0.0, 0.0, 0.1]
    wo1 = _wavelet(n, T, to[1], s)
    wo2 = -_wavelet(n, T, to[2], s)
    wo3 = _wavelet(n, T, to[3], s)
    wm1 = _wavelet(n, T, tm[1], s)
    wm2 = -_wavelet(n, T, tm[2], s)
    wm3 = _wavelet(n, T, tm[3], s)

    wo = stack([wo1,wo2,wo3]; dims=2)
    wm = stack([wm1,wm2,wm3]; dims=2)

    F = JopWindowedCorrelation1D(wo)
    cc = F * wm

    @show size(cc)
    for i = 1:3
        @test argmax(abs.(cc[:,1,i])) - div(n,2) - 1 == round(Int, (tm[i] - to[i]) / dt)
    end

    if PLOTS > 0
        using PyPlot
        close("all")
        figure(figsize=(8,4), dpi=100)
        colors = ["blue", "green", "red"]
        for i = 1:3
            plot(t,wo[:,i], label="reference", color=colors[i], linestyle=":", linewidth=i)
            plot(t,wm[:,i], label="input", color=colors[i], linestyle="--", linewidth=i)
            plot(dt .* state(F,:lags)[:,1],cc[:,1,i] ./ maximum(abs.(cc[:,1,i])), label="correlation", color=colors[i], linestyle="-", linewidth=i)
        end
        legend()
        PyPlot.grid()
        tight_layout()
        savefig("jop_correlation_windowed_verification.png")
    end
end

@testset "Sliding Correlation - correctness" begin
    n = 101
    T  = 1.0
    dt = 2 * T / (n - 1)
    t = range(-T, T; length=n)
    s  = 0.6
    to = [0.0, 0.2, 0.0]
    tm = [0.0, 0.0, 0.1]
    wo1 = _wavelet(n, T, to[1], s)
    wo2 = -_wavelet(n, T, to[2], s)
    wo3 = _wavelet(n, T, to[3], s)
    wm1 = _wavelet(n, T, tm[1], s)
    wm2 = -_wavelet(n, T, tm[2], s)
    wm3 = _wavelet(n, T, tm[3], s)

    wo = stack([wo1,wo2,wo3]; dims=2)
    wm = stack([wm1,wm2,wm3]; dims=2)

    F = JopSlidingCorrelation1D(wo; winlen = n, maxlag=0, skip = n)
    cc = F * wm

    for i = 1:3
        @test cc[1,1,i] ≈ dot(wo[:,i], wm[:,i])
    end

    F = JopSlidingCorrelation1D(wo; winlen = n, skip = n)
    cc = F * wm

    @show size(cc)
    for i = 1:3
        @test argmax(abs.(cc[:,1,i])) - div(n,2) - 1 == round(Int, (tm[i] - to[i]) / dt)
    end

    if PLOTS > 0
        using PyPlot
        close("all")
        figure(figsize=(8,4), dpi=100)
        colors = ["blue", "green", "red"]
        lags = collect(-state(F,:maxlag):state(F,:maxlag))
        for i = 1:3
            plot(t,wo[:,i], label="reference", color=colors[i], linestyle=":", linewidth=i)
            plot(t,wm[:,i], label="input", color=colors[i], linestyle="--", linewidth=i)
            plot(dt .* lags, cc[:,1,i] ./ maximum(abs.(cc[:,1,i])), label="correlation", color=colors[i], linestyle="-", linewidth=i)
        end
        legend()
        PyPlot.grid()
        tight_layout()
        savefig("jop_correlation_sliding_verification.png")
    end
end

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
