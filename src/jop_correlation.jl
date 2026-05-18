"""
    A = JopWindowedCorrelation1D(x; nwin=1, taper=nothing, maxlag=nothing)

1D Correlation operator along the first dimension with a given reference array x using discrete windows and cosine taper.\\
The operator works with any multi-dimensional array, performing the correlation along the first dimension while preserving trailing dimensions.

Arguments:
- 'x :: AbstractArray{T}': reference array with the same size the operator domain
- 'nwin :: Int': number of windows along the first dimension (default: 1)
- 'taper :: Int': taper length in samples (default: nothing -> winlen/2 where winlen = div(nt, nwin))
- 'maxlag :: Int': maximum lag in samples (default: nothing -> winlen/2)

Forward mode returns:
- 'd :: AbstractArray{T}` with shape `(2*maxlag+1, nwin, size(x)[2:end]...)` containing raw cross correlations
"""
function JopWindowedCorrelation1D(x::AbstractArray{T}; nwin::Int = 1, taper::Union{Int, Nothing} = nothing, maxlag::Union{Int, Nothing} = nothing) where {T<:AbstractFloat}
    nt = size(x, 1)
    winlen = div(nt, nwin)
    tap = (taper === nothing ? div(winlen,2) : taper)
    maxlag0 = (maxlag === nothing ? div(winlen,2) : maxlag)
    lag = clamp(maxlag0, 0, nt-1)
    lags = collect(-lag:lag)
    
    dom = JetSpace(T, size(x)...)
    rng = JetSpace(T, length(lags), nwin, size(x)[2:end]...)

    JopLn(; dom, rng, df! = JopWindowedCorrelation1D_df!, df′! = JopWindowedCorrelation1D_df′!, s = (; x=x, lags=lags, taper=tap, dom, rng) )
end

export JopWindowedCorrelation1D

@inline function _taper_weights_vec(full_taper::Vector{T}, n::Int) where {T<:AbstractFloat}
    if n <= 0
        return Vector{T}(undef, 0)           # always a Vector{T}
    else
        return full_taper[end-n+1:end]       # slicing a Vector -> Vector
    end
end

function JopWindowedCorrelation1D_df!(d::AbstractArray{T}, m::AbstractArray{T}; x::AbstractArray{T}, lags::Vector{Int}, taper::Int, dom, rng, kwargs...) where {T<:AbstractFloat}
    if size(dom) != size(m)
        error("Model vector size does not match domain size")
    end
    d .= 0

    nt = size(dom, 1)
    nwin = size(rng, 2)

    # Window partition (without taper)
    winlen  = div(nt, nwin)
    starts  = [1 + (i-1)*winlen for i in 1:nwin]
    ends    = [min(i*winlen, nt) for i in 1:nwin]

    # Precompute full cosine taper of length `taper`
    full_taper = taper == 0 ? T[] :
        T.(0.5 .* (1 .- cos.(range(0, π, length=taper))))

    # Trailing dimensions iteration
    trailing_shape = size(x)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    @inbounds begin
        # loop over trailing dims
        @threads for I in trailing_inds
            idx = I.I
            for i in 1:nwin
                lo = starts[i]; hi = ends[i]
                lo_aug = max(lo - taper, 1)
                hi_aug = min(hi + taper, nt)
                L      = hi_aug - lo_aug + 1

                # Taper weights for this window
                left_len, right_len, mid_len = lo - lo_aug, hi_aug - hi, hi - lo + 1
                weights_1d = ones(T, left_len + mid_len + right_len)
                
                # left taper
                if left_len > 0
                    tl = _taper_weights_vec(full_taper, left_len)
                    @inbounds @simd for j in 1:left_len
                        weights_1d[j] = tl[j]
                    end
                end
                
                # right taper (reversed)
                if right_len > 0
                    tr = _taper_weights_vec(full_taper, right_len) # Vector{T}
                    # copy reversed into tail
                    @inbounds @simd for j in 1:right_len
                        weights_1d[left_len + mid_len + j] = tr[right_len - j + 1]
                    end
                end

                # Compute correlations for all lags
                for (k, τ) in enumerate(lags)
                    acc = zero(T)
                    if τ >= 0
                        lm = L - τ
                        if lm >= 1
                            @fastmath @simd for t in 1:lm
                                xi = x[lo_aug + t - 1 + τ, idx...] * weights_1d[t + τ]
                                mi = m[lo_aug + t - 1,     idx...] * weights_1d[t]
                                acc += xi * mi
                            end
                        end
                    else
                        lm = -τ
                        s = L - lm
                        if s >= 1
                            @fastmath @simd for t in 1:s
                                xi = x[lo_aug + t - 1,     idx...] * weights_1d[t]
                                mi = m[lo_aug + t - 1 + lm, idx...] * weights_1d[t + lm]
                                acc += xi * mi
                            end
                        end
                    end
                    d[k, i, idx...] = acc
                end
            end
        end
    end
    d
end

function JopWindowedCorrelation1D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; x::AbstractArray{T}, lags::Vector{Int}, taper::Int, dom, rng, kwargs...) where {T<:AbstractFloat}
    if size(rng) != size(d)
        error("Data vector size does not match range size")
    end
    m .= 0

    nt = size(dom, 1)
    nwin = size(rng, 2)

    # Window partition (without taper)
    winlen  = div(nt, nwin)
    starts  = [1 + (i-1)*winlen for i in 1:nwin]
    ends    = [min(i*winlen, nt) for i in 1:nwin]

    # Precompute full cosine taper of length `taper`
    full_taper = taper == 0 ? T[] :
        T.(0.5 .* (1 .- cos.(range(0, π, length=taper))))

    # Trailing dimensions iteration
    trailing_shape = size(x)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    
    @inbounds begin
        # loop over trailing dims
        @threads for I in trailing_inds
            idx = I.I
            for i in 1:nwin
                lo = starts[i]; hi = ends[i]
                lo_aug = max(lo - taper, 1)
                hi_aug = min(hi + taper, nt)
                L      = hi_aug - lo_aug + 1

                # Taper weights for this window
                left_len, right_len, mid_len = lo - lo_aug, hi_aug - hi, hi - lo + 1
                weights_1d = ones(T, left_len + mid_len + right_len)
                
                # left taper
                if left_len > 0
                    tl = _taper_weights_vec(full_taper, left_len)
                    @inbounds @simd for j in 1:left_len
                        weights_1d[j] = tl[j]
                    end
                end
                
                # right taper (reversed)
                if right_len > 0
                    tr = _taper_weights_vec(full_taper, right_len) # Vector{T}
                    # copy reversed into tail
                    @inbounds @simd for j in 1:right_len
                        weights_1d[left_len + mid_len + j] = tr[right_len - j + 1]
                    end
                end

                # Adjoint accumulation (convolution) for all lags
                for (k, τ) in enumerate(lags)
                    dk = d[k, i, idx...] 
                    if dk == zero(T)
                        continue
                    end

                    if τ >= 0
                        lm = L - τ
                        if lm >= 1
                            @fastmath @simd for t in 1:lm
                                ta_m  = lo_aug + t - 1
                                ta_x  = ta_m + τ
                                wm_t  = weights_1d[t]
                                wx_t  = weights_1d[t + τ]
                                m[ta_m, idx...] += dk * x[ta_x, idx...] * wx_t * wm_t
                            end
                        end
                    else
                        lm = -τ
                        s  = L - lm
                        if s >= 1
                            @fastmath @simd for t in 1:s
                                ta_x  = lo_aug + t - 1
                                ta_m  = ta_x + lm
                                wx_t  = weights_1d[t]
                                wm_t  = weights_1d[t + lm]
                                m[ta_m, idx...] += dk * x[ta_x, idx...] * wx_t * wm_t
                            end
                        end
                    end
                end
            end
        end
    end
    m
end

"""
    A = JopSlidingCorrelation1D(x; nwin=1, taper=nothing, maxlag=nothing)

1D Correlation operator along the first dimension with a given reference array x using a sliding window and cosine taper.\\
The operator works with any multi-dimensional array, performing the correlation along the first dimension while preserving trailing dimensions.

Arguments:
- 'x :: AbstractArray{T}': reference array with the same size the operator domain
- 'winlen :: Int': half-width of the window used in correlation (default: 0)
- 'skip :: Int': step size between windows (default: 1)
- 'taper :: Int': taper length in samples (default: 0)
- 'maxlag :: Int': maximum lag in samples (default: 0)

Forward mode returns:
- 'd :: AbstractArray{T}` with shape `(2*maxlag+1, nt/skip, size(x)[2:end]...)` containing raw cross correlations
"""
function JopSlidingCorrelation1D(x::AbstractArray{T}; winlen::Int = 0, skip::Int = 1, taper::Int = 0, maxlag::Int = 0) where {T<:AbstractFloat}
    if winlen < 0 || taper < 0 || maxlag < 0
        error("winlen, taper, and maxlag must be non-negative")
    end
    if skip < 1
        error("skip must be at least 1")
    end
    nlags = 2*maxlag + 1
    nt = size(x,1)
    nwin = length(1:skip:nt)
    
    dom = JetSpace(T, size(x)...)
    rng = JetSpace(T, nlags, nwin, size(x)[2:end]...)

    JopLn(; dom, rng, df! = JopSlidingCorrelation1D_df!, df′! = JopSlidingCorrelation1D_df′!, s = (; x=x, winlen=winlen, skip=skip, maxlag=maxlag, taper=taper, dom, rng) )
end

export JopSlidingCorrelation1D

function JopSlidingCorrelation1D_df!(d::AbstractArray{T}, m::AbstractArray{T}; x::AbstractArray{T}, winlen::Int, skip::Int, maxlag::Int, taper::Int, dom, rng, kwargs...) where {T<:AbstractFloat}
    if size(dom) != size(m)
        error("Model vector size does not match domain size")
    end
    d .= 0

    nt = size(dom, 1)
    npad = winlen + taper + maxlag

    # Window partition with taper
    fullwin = 2 * winlen + 1 + 2 * taper

    # Precompute full cosine taper of length `taper`
    kernel = ones(T, fullwin)
    if taper > 0
        kernel[1:taper] .= 0.5 .* (1 .- cos.(range(0, π, length=taper)))
        kernel[end-taper+1:end] .= reverse(kernel[1:taper])
    end
    kernel .= kernel .^ 2
    
    # Trailing dimensions iteration
    trailing_shape = size(x)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    @inbounds begin
        # loop over trailing dims
        @threads for I in trailing_inds
            idx = I.I
            # padded traces
            x_padded = zeros(T, nt + 2*npad)
            m_padded = zeros(T, nt + 2*npad)
            x_padded[npad+1:npad+nt] .= x[:, idx...]
            m_padded[npad+1:npad+nt] .= m[:, idx...]
            j = 0
            for i in 1:skip:nt
                j += 1
                @views xi = x_padded[maxlag + i : maxlag + i + fullwin - 1]
                @views mi = m_padded[i : i + fullwin - 1 + 2*maxlag]
                xiw = xi .* kernel
                @fastmath @simd for lag in -maxlag:maxlag
                    @views mil = mi[lag + maxlag + 1 : lag + maxlag + fullwin]
                    d[lag + maxlag + 1, j, idx...] = sum(xiw .* mil)
                end
            end
        end
    end
    d
end

function JopSlidingCorrelation1D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; x::AbstractArray{T}, winlen::Int, skip::Int, maxlag::Int, taper::Int, dom, rng, kwargs...) where {T<:AbstractFloat}
    if size(rng) != size(d)
        error("Data vector size does not match range size")
    end
    m .= 0

    fill!(m, zero(T))

    nt = size(dom, 1)
    npad = winlen + taper + maxlag

    # Window partition with taper
    fullwin = 2 * winlen + 1 + 2 * taper

    # Precompute full cosine taper of length `taper`
    kernel = ones(T, fullwin)
    if taper > 0
        kernel[1:taper] .= 0.5 .* (1 .- cos.(range(0, π, length=taper)))
        kernel[end-taper+1:end] .= reverse(kernel[1:taper])
    end
    kernel .= kernel .^ 2
    
    # Trailing dimensions iteration
    trailing_shape = size(x)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    @inbounds begin
        # loop over trailing dims
        @threads for I in trailing_inds
            idx = I.I
            # padded traces
            x_padded = zeros(T, nt + 2*npad)
            m_padded = zeros(T, nt + 2*npad)
            x_padded[npad+1:npad+nt] .= x[:, idx...]
            j = 0
            for i in 1:skip:nt
                j += 1
                @views xi = x_padded[maxlag + i : maxlag + i + fullwin - 1]
                @views mi = m_padded[i : i + fullwin - 1 + 2*maxlag]
                xiw = xi .* kernel
                @fastmath @simd for lag in -maxlag:maxlag
                    val = d[lag + maxlag + 1, j, idx...]
                    mi[lag + maxlag + 1 : lag + maxlag + fullwin] .+= xiw .* val
                end
            end
            @views m[:, idx...] .= m_padded[npad+1:npad+nt]
        end
    end
    m
end