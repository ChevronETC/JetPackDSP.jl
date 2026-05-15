"""
    A = JopStreamingPEF1D(x; n=7, λ=1e-3)

Apply a 1D streaming prediction error filter (PEF) along the first dimension. \\
The filter is estimated on the fly from the reference array x.
Reference: https://doi.org/10.1190/geo2023-0646.1

Arguments:
- 'x :: AbstractArray{T}': reference array (used to estimate the PEF) with the same size the operator domain/range
- 'n :: Int': number of active samples in the PEF (default: 7). The leading sample of the PEF is always 1 (implicitly).
- 'λ :: Real': regularization parameter for the PEF estimation (default: 1e-1)

Forward mode returns:
- 'd :: AbstractArray{T}` containing filtered input with same shape as `x`

Effective filter shape:
|1|f1|f2|f3|...|fn|
"""
function JopStreamingPEF1D(x::AbstractArray{T}; n::Int=7, λ::Real=1e-1) where {T<:AbstractFloat}
    @assert n > 0 "n must be strictly greater than 0"
    @assert λ > 0 "λ must be strictly greater than 0"
    dom = JetSpace(T, size(x)...)
    rng = JetSpace(T, size(x)...)

    JopLn(; dom, rng, df! = JopStreamingPEF1D_df!, df′! = JopStreamingPEF1D_df′!, s = (; x=x, n=n, λ=T(λ)))
end

export JopStreamingPEF1D

function JopStreamingPEF1D_df!(d::AbstractArray{T}, m::AbstractArray{T}; x, n, λ, kwargs...) where {T<:AbstractFloat}
    d = streaming_pef_forward!(m, x, n, λ)
    d
end

function JopStreamingPEF1D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; x, n, λ, kwargs...) where {T<:AbstractFloat}
    m = streaming_pef_adjoint!(d, x, n, λ)
    m
end

"""
    A = JopStreamingPEF2D(x; n1=7, n2=3, λ1=1e-3, λ2=1e-3)

Apply a 2D streaming prediction error filter (PEF) along the first and second dimensions. \\
The filter is estimated on the fly from the reference array x.

Arguments:
- 'x :: AbstractArray{T}': reference array (used to estimate the PEF) with the same size the operator domain/range
- 'n1,n2 :: Int': number of active samples in the first,second dimensions of the PEF (default: 7,3). The leading sample of the PEF is always 1 (implicitly) and is centered at the first column.
- 'λ1,λ2 :: Real': regularization parameters for the first and second dimensions of the PEF estimation (default: 1e-3,1e-3)



Forward mode returns:
- 'd :: AbstractArray{T}` containing filtered input with same shape as `x`

Effective filter shape for n1 = 5 and n2 = 3: 
|f53|f52|f21|
|f43|f42|f11|
|f33|f32| 1 |
|f23|f22| 0 |
|f13|f12| 0 |

Effective filter shape for n1 = 4 and n2 = 3: 
|f43|f42|f21|
|f33|f32|f11|
|f23|f22| 1 |
|f13|f12| 0 |
"""
function JopStreamingPEF2D(x::AbstractArray{T}; n1::Int=7, n2::Int=3, λ1::Real=1e-3, λ2::Real=1e-3) where {T<:AbstractFloat}
    @assert n1 > 0 "n1 must be strictly greater than 0"
    @assert n2 > 0 "n2 must be strictly greater than 0"
    @assert λ1 > 0 "λ1 must be strictly greater than 0"
    @assert λ2 > 0 "λ2 must be strictly greater than 0"
    @assert ndims(x) > 1 "number of dimensions of x must be greater than 1"
    dom = JetSpace(T, size(x)...)
    rng = JetSpace(T, size(x)...)

    JopLn(; dom, rng, df! = JopStreamingPEF2D_df!, df′! = JopStreamingPEF2D_df′!, s = (; x=x, n1=n1, n2=n2, λ1=T(λ1), λ2=T(λ2)))
end

export JopStreamingPEF2D

function JopStreamingPEF2D_df!(d::AbstractArray{T}, m::AbstractArray{T}; x, n1, n2, λ1, λ2, kwargs...) where {T<:AbstractFloat}
    d = streaming_pef_forward!(m, x, n1, n2, λ1, λ2)
    d
end

function JopStreamingPEF2D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; x, n1, n2, λ1, λ2, kwargs...) where {T<:AbstractFloat}
    m = streaming_pef_adjoint!(d, x, n1, n2, λ1, λ2)
    m
end

function streaming_pef_forward!(m::AbstractArray{T},
                  x::AbstractArray{T},
                  n::Int,
                  λ::T) where {T<:AbstractFloat}
    nt = size(m, 1)
    λ2 = λ^2

    trailing_shape = size(m)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)
    trailing_dims = length(trailing_shape)

    # Initialize filter coefficients to zero (leading coeff is always 1)
    f = zeros(T, n, Threads.maxthreadid())

    # holders for padded arrays with filter length
    xpad = zeros(T, n + nt, Threads.maxthreadid())
    mpad = zeros(T, n + nt, Threads.maxthreadid())

    # Output array
    d = similar(m)
    d[1:1, ntuple(_->Colon(), trailing_dims)...] .= m[1:1, ntuple(_->Colon(), trailing_dims)...] # first sample is always copied

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()
        
        xpad[n+1:end, tid] .= x[:, idx...]
        mpad[n+1:end, tid] .= m[:, idx...]

        fill!(@view(f[:, tid]), zero(T))
        xTx = zero(T)

        @fastmath @simd for i = 2:nt
            # Update PEF coefficients from x
            xv = @view(xpad[i:n+i-1, tid])
            fv = @view(f[:, tid])
            xTx += xpad[n+i-1, tid]^2 - xpad[i-1, tid]^2
            xTf = sum(xv .* fv)
            fv .-= (xpad[n+i, tid] + xTf) / (λ2 + xTx) .* xv
            
            # Apply PEF to m
            d[i, idx...] = mpad[n+i, tid] + sum(@view(mpad[i:n+i-1, tid]) .* fv)
        end
    end
    d
end

function streaming_pef_forward!(m::AbstractArray{T},
                  x::AbstractArray{T},
                  n1::Int, n2::Int,
                  λ1::T, λ2::T) where {T<:AbstractFloat}
    nt = size(m, 1)
    nx = size(m, 2)
    λ12 = λ1^2
    λ22 = λ2^2
    λ = λ12 + λ22

    trailing_shape = size(m)[3:end]
    trailing_inds = CartesianIndices(trailing_shape)
    trailing_dims = length(trailing_shape)

    # Initialize filter coefficients to zero (leading coeff is always 1)
    f1 = zeros(T, n1, n2, Threads.maxthreadid())
    f2 = zeros(T, n1, n2, nt, Threads.maxthreadid()) # store filter from previous trace

    # holders for padded arrays with filter length
    ntpad = 2*n1 + nt
    nxpad = n2 + nx
    xpad = zeros(T, ntpad, nxpad, Threads.maxthreadid())
    mpad = zeros(T, ntpad, nxpad, Threads.maxthreadid())

    # Output array
    d = similar(m)

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()
        
        xpad[n1+1:n1+nt, n2+1:end, tid] .= x[:,:,idx...]
        mpad[n1+1:n1+nt, n2+1:end, tid] .= m[:,:,idx...]

        fill!(@view(f1[:, :, tid]), zero(T))
        fill!(@view(f2[:, :, :, tid]), zero(T))

        for i2 = 1:nx
            for i1 = 1:nt
                # Update PEF coefficients from x
                xv = @view(xpad[n1+i1-div(n1,2):2*n1+i1-1-div(n1,2),i2+1:n2+i2,tid])
                f1v = @view(f1[:,:,tid])
                f2v = @view(f2[:,:,i1,tid])
                xTx = sum(xv.^2) - sum(xv[div(n1,2)+1:end,end].^2)
                xTf = sum(xv .* f1v)
                fbar = (λ12 .* f1v .+ λ22 .* f2v) ./ λ
                f1v .= fbar .- (xpad[n1+i1,n2+i2,tid] + xTf) / (λ + xTx) .* xv
                f1v[div(n1,2)+1:end,end] .= 0 # zero out inactive coeffs
                f2[:,:,i1,tid] .= f1v # store current filter for next trace
                
                # Apply PEF to m
                d[i1,i2,idx...] = mpad[n1+i1,n2+i2,tid] + sum(@view(mpad[n1+i1-div(n1,2):2*n1+i1-1-div(n1,2),i2+1:n2+i2, tid]) .* f1v)
            end
        end
    end
    d
end

function streaming_pef_adjoint!(d::AbstractArray{T},
                  x::AbstractArray{T},
                  n::Int,
                  λ::T) where {T<:AbstractFloat}
    nt = size(d, 1)
    λ2 = λ^2

    trailing_shape = size(d)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    # Initialize filter coefficients to zero (leading coeff is always 1)
    f = zeros(T, n, nt, Threads.maxthreadid())

    # holders for padded arrays with filter length
    xpad = zeros(T, n + nt, Threads.maxthreadid())
    mpad = zeros(T, n + nt, Threads.maxthreadid())

    # Output array
    m = similar(d)

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()

        xpad[n+1:end, tid] .= x[:, idx...]

        # ------------------------------------------------------------
        # Reconstruct filter coefficients at each time index
        # Mathematically, the filter can be inverted from the last filter in the forward operator,
        # however, this is numerically unstable. Instead, we reconstruct the filter coefficients at each time index by re-running the forward loop.
        # ------------------------------------------------------------
        fill!(@view(f[:,:,tid]), zero(T))
        xTx = zero(T)

        @fastmath @simd for i = 2:nt
            xv = @view(xpad[i:n+i-1, tid])
            fp = @view f[:, i-1, tid]
            fn = @view f[:, i,   tid]
            xTx += xpad[n+i-1, tid]^2 - xpad[i-1, tid]^2
            xTf  = sum(xv .* fp)
            fn  .= fp .- (xpad[n+i, tid] + xTf) / (λ2 + xTx) .* xv
        end

        # ------------------------------------------------------------
        # Reverse-time adjoint sweep
        # ------------------------------------------------------------
        fill!(@view(mpad[:, tid]), zero(T))
        mpad[n+1, tid] = d[1, idx...]   # adjoint of: d[1] = m[1]

        for i = nt:-1:2
            # Adjoint of: d[i] = m[i] + dot(m[i-n:i-1], f_i)
            di = d[i, idx...]
            mpad[n+i, tid]     += di
            mpad[i:n+i-1, tid] .+= @view(f[:,i,tid]) .* di
        end
        m[:, idx...] .= mpad[n+1:end, tid]
    end
    m
end