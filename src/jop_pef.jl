"""
    A = JopStreamingPEF1D(x; n=7, λ=1e-3)

Apply a 1D streaming prediction error filter (PEF) along the first dimension. \\
The filter is estimated on the fly from the reference array x.
Reference: https://doi.org/10.1190/geo2023-0646.1

Arguments:
- 'x :: AbstractArray{T}': reference array (used to estimate the PEF) with the same size the operator domain/range
- 'n :: Int': number of samples in the PEF (default: 8). The leading sample of the PEF is always 1, so there are `n-1` active coefficients.
- 'λ :: Real': regularization parameter for the PEF estimation (default: 1e-1)

Forward mode returns:
- 'd :: AbstractArray{T}` containing filtered input with same shape as `x`
"""
function JopStreamingPEF1D(x::AbstractArray{T}; n::Int=8, λ::Real=1e-1) where {T<:AbstractFloat}
    @assert n > 1 "n must be strictly greater than 1"    
    @assert λ > 0 "λ must be strictly greater than 0"    
    dom = JetSpace(T, size(x)...)
    rng = JetSpace(T, size(x)...)
    trailing_shape = size(dom)[2:end]
    trailing_dims = length(trailing_shape)
    f = zeros(T, n, trailing_shape...) # filter coefficients
    f[1:1, ntuple(_->Colon(), trailing_dims)...] .= 1 # the leading coefficient of PEF is always 1

    JopLn(; dom, rng, df! = JopStreamingPEF1D_df!, df′! = JopStreamingPEF1D_df′!, s = (; x=x, f=f, λ=λ, chksum_x=Ref(zero(UInt32)), chksum_f=Ref(zero(UInt32))) )
end

export JopStreamingPEF1D

function JopStreamingPEF1D_df!(d::AbstractArray{T}, m::AbstractArray{T}; x, f, λ, kwargs...) where {T<:AbstractFloat}
    trailing_shape = size(x)[2:end]
    trailing_dims = length(trailing_shape)
    f[2:end, ntuple(_->Colon(), trailing_dims)...] .= 0 # reset the filter
    d = streaming_pef_forward!(m, x, f, λ)
    kwargs[:chksum_x][] = crc32c(x)
    kwargs[:chksum_f][] = crc32c(f)
    d
end

function JopStreamingPEF1D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; x, f, λ, kwargs...) where {T<:AbstractFloat}
    m = streaming_pef_adjoint!(d, x, f, λ)
    # m = streaming_pef_adjoint_2!(d, x, f, λ, kwargs[:chksum_x], kwargs[:chksum_f])
    m
end

function streaming_pef_forward!(m::AbstractArray{T},
                  x::AbstractArray{T},
                  f::AbstractArray{T},
                  λ::Real) where {T<:AbstractFloat}
    nt = size(m, 1)
    nf = size(f, 1) - 1 # number of active filter coefficients
    λ2T = T(λ^2)

    trailing_shape = size(m)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)
    trailing_dims = length(trailing_shape)

    # holders for padded arrays with filter length
    xpad = zeros(T, nf + nt, Threads.maxthreadid())
    mpad = zeros(T, nf + nt, Threads.maxthreadid())

    # Output array
    d = similar(m)
    d[1:1, ntuple(_->Colon(), trailing_dims)...] .= m[1:1, ntuple(_->Colon(), trailing_dims)...] # first sample is always copied

    @inbounds begin
        # loop over trailing dims
         @threads for I in trailing_inds
            idx = I.I
            tid = Threads.threadid()
            f1 = reverse(f[2:end, idx...])
            xpad[nf+1:end, tid] .= x[:, idx...]
            mpad[nf+1:end, tid] .= m[:, idx...]

            xTx = zero(T)
            xTf = zero(T)

            @fastmath @simd for i = 2:nt
                # Update PEF coefficients from x
                xTx += xpad[nf+i-1, tid]^2 - xpad[i-1, tid]^2
                xTf = sum(xpad[i:nf+i-1, tid] .* f1)
                f1 .-= (xpad[nf+i, tid] + xTf) / (λ2T + xTx) .* xpad[i:nf+i-1, tid]
                
                # Apply PEF to m
                d[i, idx...] = mpad[nf+i, tid] + sum(mpad[i:nf+i-1, tid] .* f1)
            end
            # copy back the final filter coefficients
            f[2:end, idx...] .= reverse(f1)
        end
    end
    d
end

function streaming_pef_adjoint!(d::AbstractArray{T},
                  x::AbstractArray{T},
                  f::AbstractArray{T},
                  λ::Real) where {T<:AbstractFloat}
    nt = size(d, 1)
    nf = size(f, 1) - 1 # number of active filter coefficients
    λ2T = T(λ^2)

    trailing_shape = size(d)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    # holders for padded arrays with filter length
    xpad = zeros(T, nf + nt, Threads.maxthreadid())
    mpad = zeros(T, nf + nt, Threads.maxthreadid())

    # Output array
    m = similar(d)

    @inbounds begin
        # loop over trailing dims
        @threads for I in trailing_inds
            idx = I.I
            tid = Threads.threadid()
            xpad[nf+1:end, tid] .= x[:, idx...]

            # ------------------------------------------------------------
            # Reverse-time adjoint sweep
            # ------------------------------------------------------------
            fill!(@view(mpad[:, tid]), zero(T))
            mpad[nf+1, tid] = d[1, idx...]   # adjoint of: d[1] = m[1]

            for i = nt:-1:2
                # Recompute filter at time i from cold start (active taps = 0)
                f1 = zeros(T, nf)
                xTx = zero(T)
                @fastmath @simd for j = 2:i
                    xTx += xpad[nf+j-1, tid]^2 - xpad[j-1, tid]^2
                    xTf = sum(xpad[j:nf+j-1, tid] .* f1)
                    f1 .-= (xpad[nf+j, tid] + xTf) / (λ2T + xTx) .* xpad[j:nf+j-1, tid]
                end

                # Adjoint of: d[i] = m[i] + dot(m[i-nf:i-1], f_i)
                di = d[i, idx...]
                mpad[nf+i, tid]     += di
                mpad[i:nf+i-1, tid] .+= f1 .* di
            end

            m[:, idx...] .= mpad[nf+1:end, tid]
        end
    end
    m
end

function streaming_pef_adjoint_2!(d::AbstractArray{T},
                      x::AbstractArray{T},
                      f::AbstractArray{T},
                      λ::Real,
                      chksum_x,
                      chksum_f) where {T<:AbstractFloat}
    # check if x and f are compatible with forward pass (i.e. not modified since last forward)
    valid_x = chksum_x[] == crc32c(x)
    valid_f = chksum_f[] == crc32c(f)
    isvalid = (valid_x && valid_f)

    if !isvalid
        @warn "Forward will be run to compute terminal filter state for adjoint."
        dummy = streaming_pef_forward!(zero(d), x, f, λ)
        chksum_x[] = crc32c(x)
        chksum_f[] = crc32c(f)
    end

    nt = size(d, 1)
    nf = size(f, 1) - 1
    λ2T = T(λ^2)

    trailing_shape = size(d)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    xpad = zeros(T, nf + nt, Threads.maxthreadid())
    mpad = zeros(T, nf + nt, Threads.maxthreadid())

    m = similar(d)
    fill!(m, zero(T))

    @inbounds begin
        # loop over trailing dims
        @threads for I in trailing_inds
            idx = I.I
            tid = Threads.threadid()
            f1 = reverse(f[2:end, idx...]) # filter at terminal time index from forward pass
            xpad[nf+1:end, tid] .= x[:, idx...]
            # xTx = sum(xpad[nt-2:nf+nt-1, tid].^2) # compute xTx at terminal time index

            # Rebuild xTx at terminal index using the same recurrence as in forward.
            xTx = zero(T)
            for i = 2:nt
                xTx += xpad[nf+i-1, tid]^2 - xpad[i-1, tid]^2
            end

            # ------------------------------------------------------------
            # Reverse-time adjoint + reversible filter rollback
            # ------------------------------------------------------------
            fill!(@view(mpad[:, tid]), zero(T))
            mpad[nf+1, tid] = d[1, idx...]   # forward copied first sample directly

            for i = nt:-1:2
                di = d[i, idx...]
                a = @view xpad[i:nf+i-1, tid]

                # Adjoint of: d[i] = m[i] + dot(m[i-nf:i-1], f_i)
                mpad[nf+i, tid] += di
                mpad[i:nf+i-1, tid] .+= f1 .* di

                # Invert one forward filter update: f_i -> f_{i-1}
                # Forward was: f_i = f_{i-1} - ((b + dot(a,f_{i-1})) / c) * a
                #  with b = x[nf+i], c = λ^2 + xTx.
                c = λ2T + xTx
                b = xpad[nf+i, tid]
                r = sum(a .* f1)        # r = dot(a, f_i)
                a2 = sum(a .* a)
                den = c - a2
                s = (c * r + a2 * b) / den  # s = dot(a, f_{i-1})
                β = (b + s) / c
                f1 .+= β .* a

                # Roll xTx back to previous time index
                xTx -= xpad[nf+i-1, tid]^2 - xpad[i-1, tid]^2
            end

            m[:, idx...] .= mpad[nf+1:end, tid]
        end
    end
    m
end