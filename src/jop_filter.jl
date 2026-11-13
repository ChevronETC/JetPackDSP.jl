"""
    A = JopFilter(spc, responsetype, designmethod)

where `A` is a filter applied to a signal in `spc::JotSpace`, and built
using `responsetype` and `designmethod`.  The `responsetype` and
`designmethod` are described in `https://github.com/JuliaDSP/DSP.jl`.
The filter is applied along the fast dimension of the space.

# Examples

## 1D
```
using JetPackDSP, Jets
A = JopFilter(JetSpace(Float64,512), Highpass(10.0, fs=125), Butterworth(4))
d = A*rand(domain(A))
```

## 2D
```
using JetPackDSP, Jets
A = JopFilter(JetSpace(Float64,512,10), Highpass(10.0, fs=125), Butterworth(4))
d = A*rand(domain(A))
```

# Alternative option
* The filter coefficients can be provided explicitly as a vector of samples. It is assumed zero phase and centered in time. To maximize performance, the domain and range are restricted to real values.

    A = JopFilter(spc, h; domain = "frequency")

where `h` is the filter. When domain = "frequency", if the filter length does not match that of the domain/range fast dimension, it will be padded or cut equally on both sides.

# Example
```
using JetPackDSP, Jets
A = JopFilter(JetSpace(Float64,512, 256, 128), Float64[0.0, -1.0, 1.0, 1.0, 0.0])
d = A*rand(domain(A))
```

# Convenience operator
* User might want to swap the role of the filter and the model `m` so that the filter itself becomes the model (e.g., during matching filter estimation). In this case, we provide another convenience operator constructor

    A = JopFilter(v; domain = "frequency", flen = size(v,1))

where `v` is the multidimensional data to which the filter (model) is to be applied. Note that in this case, the domain is 1D and will match the fast dimension of `v` when mode = "frequency", or will be of size `flen` when mode = "time". The range will match the size of `v`.

# Example
```
using JetPackDSP, Jets
A = JopFilter(randn(Float64, 512, 256, 128))
d = A*rand(size(domain(A),1))
```
"""
function JopFilter(spc::JetSpace{T}, responsetype::FilterType, designmethod::FilterCoefficients) where {T}
	n = size(spc,1)
	tmp1 = zeros(T,n)
	tmp2 = zeros(T,n)
	tmp1[div(n,2)] = 1;
	tmp2 .= filtfilt(digitalfilter(responsetype, designmethod), tmp1)
	ztmp2 = fft(tmp2)
	filter = real.(abs.(ztmp2))
	JopLn(dom = spc, rng = spc, df! = JopFilter_df!, df′! = JopFilter_df′!, s = (filter=filter,))
end

function JopFilter(spc::JetSpace{T}, f::AbstractVector{T}; domain::String="frequency") where {T<:Real}
	n = size(spc,1)
    nf = length(f)

    if domain == "frequency"
        
        # trim the filter if it is too long but keep its original parity
        if nf > n
            istart = div(nf-n,2)+1
            iend = istart + n - 1
            f = f[istart:iend]
        # pad the filter if it is too short but keep its original parity
        elseif nf < n
            istart = div(n-nf,2)
            iend = n - nf - istart
            f = vcat(zeros(T, istart), f, zeros(T, iend))
        end

        # deduce the center of the filter after padding or trimming to determine the proper phase shift
        iseven(n) && iseven(nf) && (center = n/2-1/2)
        isodd(n) && isodd(nf) && (center = n/2-1/2)
        iseven(n) && isodd(nf) && (center = n/2-1)
        isodd(n) && iseven(nf) && (center = n/2-1)

        f_fft = rfft(f)
        nfreq = length(f_fft)
        dw = T(2 * π / n * center)
        phase = exp.(im .* LinRange(0,nfreq-1,nfreq) .* dw )
        f_fft_phase = f_fft .* phase
        JopLn(dom = spc, rng = spc, df! = JopFilterReal_df!, df′! = JopFilterReal_df′!, s = (; filter=f_fft_phase,))
    elseif domain == "time"
        iseven(nf) && @warn "The filter length is even, which will cause a half-sample shift in the time domain."
        JopLn(dom = spc, rng = spc, df! = JopFilterRealTD_df!, df′! = JopFilterRealTD_df′!, s = (; filter=f,))
    else
        error("Unknown domain: $domain. Possible options are \"frequency\" and \"time\".")
    end
end

function JopFilter(v::AbstractArray{T}; domain::String="frequency", flen = size(v,1)) where {T<:Real}
    n = size(v, 1)
    rng = JetSpace(T, size(v)...)

    v = reshape(v, n, :)

    if domain == "frequency"
        dom = JetSpace(T, n)
        v_fft = rfft(v, 1)
        nfreq = size(v_fft, 1)
        iseven(n) && (center = n/2-1/2)
        isodd(n) && (center = n/2-1/2)
        dw = T(2 * π / n * center)
        phase = exp.(im .* LinRange(0,nfreq-1,nfreq) .* dw )
    
        JopLn(; dom = dom, rng = rng, df! = JopFilterNd1d_df!, df′! = JopFilterNd1d_df′!, 
            s = (; v_fft=v_fft, phase=phase))
    elseif domain == "time"
        iseven(flen) && @warn "The filter length is even, which will cause a half-sample shift in the time domain."
        dom = JetSpace(T, flen)
        JopLn(dom = dom, rng = rng, df! = JopFilterNd1dTD_df!, df′! = JopFilterNd1dTD_df′!, s = (; v=v,))
    else
        error("Unknown domain: $domain. Possible options are \"frequency\" and \"time\".")
    end
end

export JopFilter

function JopFilter_df!(d::AbstractArray{T}, m::AbstractArray{T}; filter, kwargs...) where {T}
	n = size(m, 1)
    _d = reshape(d, n, :)
    _m = reshape(m, n, :)

    ztmp = zeros(Complex{T}, n)
    for k = 1:size(_d, 2)
        @inbounds begin
            ztmp .= fft(_m[:,k])
            ztmp .*= filter
            _d[:,k] .= real.(ifft(ztmp))
        end
    end
    d
end

function JopFilter_df′!(m::AbstractArray{T}, d::AbstractArray{T}; filter, kwargs...) where {T}
    n = size(m, 1)
    _d = reshape(d, n, :)
    _m = reshape(m, n, :)

    ztmp = zeros(Complex{T}, n)
    for k = 1:size(_d,2)
        @inbounds begin
            ztmp .= fft(_d[:,k])
            ztmp .*= filter
            _m[:,k] .= real.(ifft(ztmp))
        end
    end
    m
end

function JopFilterReal_df!(d::AbstractArray{T}, m::AbstractArray{T}; filter, kwargs...) where {T<:Real}
	n = size(m, 1)
    _d = reshape(d, n, :)
    _m = reshape(m, n, :)

    m_fft = rfft(_m, 1)
    @inbounds @threads for k = 1:size(_d, 2)
        @inbounds begin
            _d[:,k] .= irfft( m_fft[:,k] .* filter, n)
        end
    end
    d
end

function JopFilterReal_df′!(m::AbstractArray{T}, d::AbstractArray{T}; filter, kwargs...) where {T<:Real}
    n = size(m, 1)
    _d = reshape(d, n, :)
    _m = reshape(m, n, :)

    d_fft = rfft(_d, 1)
    @inbounds @threads for k = 1:size(_d, 2)
        @inbounds begin
            _m[:,k] = irfft(conj(filter) .* d_fft[:,k], n)
        end
    end
    m
end

function JopFilterRealTD_df!(d::AbstractArray{T}, m::AbstractArray{T}; filter, kwargs...) where {T<:Real}
	n = size(m, 1)
    _d = reshape(d, n, :)
    _m = reshape(m, n, :)
    L = length(filter)
    ic = div(L+1, 2)   # 1-based center tap of the filter (exact for odd L)

    @inbounds @threads for k = 1:size(_d, 2)
        @inbounds for i = 1:n
            s = zero(T)
            for j = 1:L
                idx = i + ic - j
                if 1 <= idx <= n
                    s += _m[idx, k] * filter[j]
                end
            end
            _d[i, k] = s
        end
    end
    d
end

function JopFilterRealTD_df′!(m::AbstractArray{T}, d::AbstractArray{T}; filter, kwargs...) where {T<:Real}
    n = size(m, 1)
    _d = reshape(d, n, :)
    _m = reshape(m, n, :)
    L = length(filter)
    ic = div(L+1, 2)   # 1-based center tap of the filter (exact for odd L)

    @inbounds @threads for k = 1:size(_d, 2)
        @inbounds for p = 1:n
            s = zero(T)
            for j = 1:L
                i = p - ic + j
                if 1 <= i <= n
                    s += _d[i, k] * filter[j]
                end
            end
            _m[p, k] = s
        end
    end
    m
end

function JopFilterNd1d_df!(d::AbstractArray{T}, m::AbstractVector{T}; v_fft, phase, kwargs...) where {T<:Real}
    n = length(m)
    _d = reshape(d, n, :)
    ntr = size(_d, 2)

    m_fft = rfft(m)
    m_fft_phase = m_fft .* phase
    @inbounds @threads for k = 1:ntr
        @inbounds begin
            _d[:,k] .= irfft( v_fft[:,k] .* m_fft_phase, n)
        end
    end
    d
end

function JopFilterNd1d_df′!(m::AbstractVector{T}, d::AbstractArray{T}; v_fft, phase, kwargs...) where {T<:Real}
    n = length(m)
    _d = reshape(d, n, :)
    ntr = size(_d, 2)

    d_fft = rfft(_d, 1)
    nf = size(d_fft, 1)
    m_fft = zeros(Complex{T}, nf)
    @inbounds for k = 1:ntr
        @inbounds begin
            m_fft .+= conj(v_fft[:,k] .* phase) .* d_fft[:,k]
        end
    end
    m = irfft(m_fft, n)
    m
end

function JopFilterNd1dTD_df!(d::AbstractArray{T}, m::AbstractVector{T}; v, kwargs...) where {T<:Real}
    n = size(v, 1)
    _d = reshape(d, n, :)
    v = reshape(v, n, :)
    L = length(m)
    ic = div(L+1, 2)   # 1-based center tap of the filter (exact for odd L)

    @inbounds @threads for k = 1:size(_d, 2)
        @inbounds for i = 1:n
            s = zero(T)
            for j = 1:L
                idx = i + ic - j
                if 1 <= idx <= n
                    s += v[idx, k] * m[j]
                end
            end
            _d[i, k] = s
        end
    end
    d
end

function JopFilterNd1dTD_df′!(m::AbstractVector{T}, d::AbstractArray{T}; v, kwargs...) where {T<:Real}
    n = size(v, 1)
    _d = reshape(d, n, :)
    ntr = size(_d, 2)
    L = length(m)
    ic = div(L+1, 2)   # 1-based center tap of the filter (exact for odd L)
    nt = Threads.nthreads()
    m_local = zeros(T, L, nt+1)

    @inbounds @threads for k = 1:ntr
        tid = Threads.threadid()
        for i = 1:n
            di = _d[i, k]
            for j = 1:L
                idx = i + ic - j
                if 1 <= idx <= n
                    m_local[j, tid] += v[idx, k] * di
                end
            end
        end
    end
    
    m = dropdims(sum(m_local, dims=2), dims=2)
    m
end