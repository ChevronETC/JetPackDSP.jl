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

    # Precompute taper weights per window (independent of trailing dims)
    win_lo_aug  = Vector{Int}(undef, nwin)
    win_L       = Vector{Int}(undef, nwin)
    win_weights = Vector{Vector{T}}(undef, nwin)
    for i in 1:nwin
        lo = starts[i]; hi = ends[i]
        lo_aug = max(lo - taper, 1)
        hi_aug = min(hi + taper, nt)
        win_lo_aug[i] = lo_aug
        win_L[i]      = hi_aug - lo_aug + 1
        left_len  = lo     - lo_aug
        right_len = hi_aug - hi
        mid_len   = hi     - lo + 1
        w = ones(T, left_len + mid_len + right_len)
        if left_len > 0
            tl = _taper_weights_vec(full_taper, left_len)
            @inbounds @simd for j in 1:left_len
                w[j] = tl[j]
            end
        end
        if right_len > 0
            tr = _taper_weights_vec(full_taper, right_len)
            @inbounds @simd for j in 1:right_len
                w[left_len + mid_len + j] = tr[right_len - j + 1]
            end
        end
        win_weights[i] = w
    end

    # Trailing dimensions iteration
    trailing_shape = size(x)[2:end]
    trailing_inds  = CartesianIndices(trailing_shape)
    max_L = maximum(win_L)

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        # Scratch buffers: allocated once per trailing element, reused across windows
        wx = Vector{T}(undef, max_L)
        wm = Vector{T}(undef, max_L)
        for i in 1:nwin
            lo_aug     = win_lo_aug[i]
            L          = win_L[i]
            weights_1d = win_weights[i]

            # Gather weighted copies into contiguous buffers (removes strided access from lag loop)
            @fastmath @simd for t in 1:L
                wx[t] = x[lo_aug + t - 1, idx...] * weights_1d[t]
                wm[t] = m[lo_aug + t - 1, idx...] * weights_1d[t]
            end

            # Compute correlations for all lags on contiguous buffers
            for (k, τ) in enumerate(lags)
                acc = zero(T)
                if τ >= 0
                    lm = L - τ
                    if lm >= 1
                        @fastmath @simd for t in 1:lm
                            acc += wm[t + τ] * wx[t]
                        end
                    end
                else
                    lm = -τ
                    s  = L - lm
                    if s >= 1
                        @fastmath @simd for t in 1:s
                            acc += wm[t] * wx[t + lm]
                        end
                    end
                end
                d[k, i, idx...] = acc
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

    # Precompute taper weights per window (independent of trailing dims)
    win_lo_aug  = Vector{Int}(undef, nwin)
    win_L       = Vector{Int}(undef, nwin)
    win_weights = Vector{Vector{T}}(undef, nwin)
    for i in 1:nwin
        lo = starts[i]; hi = ends[i]
        lo_aug = max(lo - taper, 1)
        hi_aug = min(hi + taper, nt)
        win_lo_aug[i] = lo_aug
        win_L[i]      = hi_aug - lo_aug + 1
        left_len  = lo     - lo_aug
        right_len = hi_aug - hi
        mid_len   = hi     - lo + 1
        w = ones(T, left_len + mid_len + right_len)
        if left_len > 0
            tl = _taper_weights_vec(full_taper, left_len)
            @inbounds @simd for j in 1:left_len
                w[j] = tl[j]
            end
        end
        if right_len > 0
            tr = _taper_weights_vec(full_taper, right_len)
            @inbounds @simd for j in 1:right_len
                w[left_len + mid_len + j] = tr[right_len - j + 1]
            end
        end
        win_weights[i] = w
    end

    # Trailing dimensions iteration
    trailing_shape = size(x)[2:end]
    trailing_inds  = CartesianIndices(trailing_shape)
    max_L = maximum(win_L)

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx     = I.I
        # Scratch buffers: allocated once per trailing element, reused across windows
        wx      = Vector{T}(undef, max_L)
        m_local = Vector{T}(undef, max_L)
        for i in 1:nwin
            lo_aug     = win_lo_aug[i]
            L          = win_L[i]
            weights_1d = win_weights[i]

            # Gather weighted x into contiguous buffer
            @fastmath @simd for t in 1:L
                wx[t] = x[lo_aug + t - 1, idx...] * weights_1d[t]
            end

            @simd for t in 1:L
                m_local[t] = zero(T)
            end

            # Accumulate adjoint into local contiguous buffer (no strided writes in lag loop)
            for (k, τ) in enumerate(lags)
                dk = d[k, i, idx...]
                dk == zero(T) && continue

                if τ >= 0
                    lm = L - τ
                    if lm >= 1
                        @fastmath @simd for t in 1:lm
                            m_local[t + τ] += dk * wx[t] * weights_1d[t + τ]
                        end
                    end
                else
                    lm = -τ
                    s  = L - lm
                    if s >= 1
                        @fastmath @simd for t in 1:s
                            m_local[t] += dk * wx[t + lm] * weights_1d[t]
                        end
                    end
                end
            end

            # Scatter result back (one strided write per element)
            @fastmath @simd for t in 1:L
                m[lo_aug + t - 1, idx...] += m_local[t]
            end
        end
    end
    m
end

"""
    A = JopSlidingCorrelation1D(x; winlen=1, skip=1, taper=nothing, maxlag=nothing)

1D Correlation operator along the first dimension with a given reference array x using a sliding window and cosine taper.\\
The operator works with any multi-dimensional array, performing the correlation along the first dimension while preserving trailing dimensions.

Arguments:
- 'x :: AbstractArray{T}': reference array with the same size the operator domain
- 'winlen :: Int': half-width of the window used in correlation (default: 0)
- 'skip :: Int': step size between windows (default: 1)
- 'taper :: Int': taper length in samples (default: nothing -> winlen/2)
- 'maxlag :: Int': maximum lag in samples (default: nothing -> winlen/2)

Forward mode returns:
- 'd :: AbstractArray{T}` with shape `(2*maxlag+1, nt/skip, size(x)[2:end]...)` containing raw cross correlations
"""
function JopSlidingCorrelation1D(x::AbstractArray{T}; winlen::Int = 0, skip::Int = 1, taper::Union{Int, Nothing} = nothing, maxlag::Union{Int, Nothing} = nothing) where {T<:AbstractFloat}
    if skip < 1
        error("skip must be at least 1")
    end
    taper = (taper === nothing ? div(winlen,2) : taper)
    maxlag = (maxlag === nothing ? div(winlen,2) : maxlag)
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
    nlags = 2*maxlag + 1

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

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        # padded traces
        x_padded = zeros(T, nt + 2*npad)
        m_padded = zeros(T, nt + 2*npad)
        @fastmath @simd for t in 1:nt
            x_padded[npad + t] = x[t, idx...]
            m_padded[npad + t] = m[t, idx...]
        end
        xiw     = Vector{T}(undef, fullwin)
        d_local = Vector{T}(undef, nlags)
        j = 0
        for i in 1:skip:nt
            j += 1
            # Build weighted x slice
            @fastmath @simd for t in 1:fullwin
                xiw[t] = x_padded[maxlag + i + t - 1] * kernel[t]
            end
            # d_local[k] = sum_t xiw[t] * m_padded[i+k+t-2]  (outer t, inner k = SIMD-friendly)
            @simd for k in 1:nlags; d_local[k] = zero(T); end
            @fastmath for t in 1:fullwin
                xit  = xiw[t]
                base = i + t - 2
                @simd for k in 1:nlags
                    d_local[k] += xit * m_padded[base + k]
                end
            end
            @simd for k in 1:nlags
                d[k, j, idx...] = d_local[k]
            end
        end
    end
    d
end

function JopSlidingCorrelation1D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; x::AbstractArray{T}, winlen::Int, skip::Int, maxlag::Int, taper::Int, dom, rng, kwargs...) where {T<:AbstractFloat}
    if size(rng) != size(d)
        error("Data vector size does not match range size")
    end
    fill!(m, zero(T))

    nt = size(dom, 1)
    npad = winlen + taper + maxlag
    nlags = 2*maxlag + 1

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

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        # padded traces
        x_padded = zeros(T, nt + 2*npad)
        m_padded = zeros(T, nt + 2*npad)
        @fastmath @simd for t in 1:nt
            x_padded[npad + t] = x[t, idx...]
        end
        xiw = Vector{T}(undef, fullwin)
        j = 0
        for i in 1:skip:nt
            j += 1
            # Build weighted x slice
            @fastmath @simd for t in 1:fullwin
                xiw[t] = x_padded[maxlag + i + t - 1] * kernel[t]
            end
            # Accumulate adjoint: m_padded[i+k+t-2] += d[k,j]*xiw[t] (AXPY per lag)
            @fastmath for k in 1:nlags
                dk = d[k, j, idx...]
                dk == zero(T) && continue
                base = i + k - 2
                @simd for t in 1:fullwin
                    m_padded[base + t] += dk * xiw[t]
                end
            end
        end
        @fastmath @simd for t in 1:nt
            m[t, idx...] = m_padded[npad + t]
        end
    end
    m
end