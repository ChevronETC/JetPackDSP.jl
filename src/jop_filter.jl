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
