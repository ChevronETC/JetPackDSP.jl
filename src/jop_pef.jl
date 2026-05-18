"""
    A = JopStreamingPEF1D(; dom, n=7, λ=1e-1)

Apply a 1D streaming prediction error filter (PEF) along the first dimension. \\
The filter is estimated and applied on the fly.
Reference: https://doi.org/10.1190/geo2023-0646.1

Arguments:
- 'dom :: JetSpace{T}': domain (and range) of the operator
- 'n :: Int': number of active samples in the PEF (default: 7). The leading sample of the PEF is always 1 (implicitly).
- 'λ :: Real': regularization parameter for the PEF estimation (default: 1e-1)

Forward mode returns:
- 'd :: AbstractArray{T}` containing filtered input with same shape as `dom`

Effective filter shape:
|1|f1|f2|f3|...|fn|

Note: the operator is non-linear, so the gradient of `1/2||A(m)||^2` given in Jon Claerbout's short book "Data Fitting with Nonstationary Statistics" Chapter 3 (effectively A|ₘ'*A(m)) is incorrect.\\
It is only exact when `λ` is zero (no regularization). In the more general case, it will be missing a correction term that accounts for the variation of the filter coefficients with respect to the input `m`.
Warning: the linearization implementation below is only approximately correct. The adjoint will still pass the dot product test, but it will be the adjoint of an approximate linearization.
"""
function JopStreamingPEF1D(; dom::JetSpace{T}, n::Int=7, λ::Real=1e-1) where {T<:AbstractFloat}
    @assert n > 0 "n must be strictly greater than 0"
    @assert λ > 0 "λ must be strictly greater than 0"
    rng = dom

    Jet(; dom, rng, f! = JopStreamingPEF1D_f!, df! = JopStreamingPEF1D_df!, df′! = JopStreamingPEF1D_df′!, s = (; n=n, λ=T(λ)))
end

JopNlStreamingPEF1D(;kwargs...) = JopNl(JopStreamingPEF1D(;kwargs...))
JopLnStreamingPEF1D(; v, kwargs...) = JopLn(JopStreamingPEF1D(;kwargs...), v)

export JopNlStreamingPEF1D
export JopLnStreamingPEF1D

function JopStreamingPEF1D_f!(d::AbstractArray{T}, m::AbstractArray{T}; kwargs...) where {T<:AbstractFloat}
    n = kwargs[:n]
    λ = kwargs[:λ]              
    nt = size(m, 1)
    λ2 = λ^2

    trailing_shape = size(m)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)
    trailing_dims = length(trailing_shape)

    # Initialize filter coefficients to zero (leading coeff is always 1)
    f = zeros(T, n, Threads.maxthreadid())

    # holder for padded array with filter length
    xpad = zeros(T, n + nt, Threads.maxthreadid())

    # Output array
    d[1:1, ntuple(_->Colon(), trailing_dims)...] .= m[1:1, ntuple(_->Colon(), trailing_dims)...] # first sample is always copied

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()
        
        xpad[n+1:end, tid] .= m[:, idx...]

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
            d[i, idx...] = xpad[n+i, tid] + sum(@view(xpad[i:n+i-1, tid]) .* fv)
        end
    end
    d
end

function JopStreamingPEF1D_df!(δd::AbstractArray{T}, δm::AbstractArray{T}; kwargs...) where {T<:AbstractFloat}
    mₒ = kwargs[:mₒ]
    n  = kwargs[:n]
    λ  = kwargs[:λ]
    nt = size(mₒ, 1)
    λ2 = λ^2

    trailing_shape = size(δm)[2:end]
    trailing_inds  = CartesianIndices(trailing_shape)
    trailing_dims  = length(trailing_shape)

    # filter at step i-1 (rolling), linearized filter perturbation (rolling)
    f   = zeros(T, n, Threads.maxthreadid())
    δf  = zeros(T, n, Threads.maxthreadid())

    # holders for padded arrays with filter length
    xpad  = zeros(T, n + nt, Threads.maxthreadid())
    δxpad = zeros(T, n + nt, Threads.maxthreadid())

    δd[1:1, ntuple(_->Colon(), trailing_dims)...] .= δm[1:1, ntuple(_->Colon(), trailing_dims)...] # first sample is always copied

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()

        xpad[n+1:end, tid]  .= mₒ[:, idx...]
        δxpad[n+1:end, tid] .= δm[:, idx...]

        fill!(@view(f[:, tid]),  zero(T))
        fill!(@view(δf[:, tid]), zero(T))
        xTx = zero(T)

        @fastmath @simd for i = 2:nt
            xv  = @view xpad[i:n+i-1, tid]     # window of mₒ (= x in f!)
            δmv = @view δxpad[i:n+i-1, tid]    # window of δm
            fv  = @view f[:, tid]               # f_{i-1}
            δfv = @view δf[:, tid]              # δf_{i-1}

            xTx += xpad[n+i-1, tid]^2 - xpad[i-1, tid]^2
            ci   = λ2 + xTx
            ei   = xpad[n+i, tid] + sum(xv .* fv)
            αi   = ei / ci

            # Linearize filter update: δf_i = δf_{i-1} - δα_i*xv - α_i*δmv
            # (must use fv = f_{i-1} and δfv = δf_{i-1} before either is updated)
            δei  = δxpad[n+i, tid] + sum(xv .* δfv) + sum(δmv .* fv)
            δci  = 2 * sum(xv .* δmv)
            δαi  = (δei - αi * δci) / ci
            δfv .-= δαi .* xv .+ αi .* δmv    # δf_i  (updated before fv)
            fv  .-= αi .* xv                   # f_i   (now fv = f_i)

            # Apply linearized PEF to δm: δd[i] = δm_i + dot(δmv, f_i) + dot(xv, δf_i)
            δd[i, idx...] = δxpad[n+i, tid] + sum(δmv .* fv) + sum(xv .* δfv)
        end
    end
    δd
end

function JopStreamingPEF1D_df′!(δm::AbstractArray{T}, δd::AbstractArray{T}; kwargs...) where {T<:AbstractFloat}
    mₒ = kwargs[:mₒ]
    n  = kwargs[:n]
    λ  = kwargs[:λ]
    nt = size(mₒ, 1)
    λ2 = λ^2

    trailing_shape = size(δm)[2:end]
    trailing_inds  = CartesianIndices(trailing_shape)
    trailing_dims  = length(trailing_shape)

    # filter history, RLS scalars (needed for exact adjoint of df!)
    f    = zeros(T, n, nt, Threads.maxthreadid())
    α    = zeros(T, nt,   Threads.maxthreadid())
    c    = zeros(T, nt,   Threads.maxthreadid())

    # holders for padded arrays with filter length
    xpad = zeros(T, n + nt, Threads.maxthreadid())
    mpad = zeros(T, n + nt, Threads.maxthreadid())

    # adjoint of δf chain (propagated backward) and scratch for δmv adjoint
    g    = zeros(T, n, Threads.maxthreadid())
    dav  = zeros(T, n, Threads.maxthreadid())

    δm[1:1, ntuple(_->Colon(), trailing_dims)...] .= δd[1:1, ntuple(_->Colon(), trailing_dims)...] # first sample is always copied

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()

        xpad[n+1:end, tid] .= mₒ[:, idx...]

        # Pass 1: reconstruct filter history + α_i, c_i from mₒ (same as f! with x=mₒ)
        fill!(@view(f[:, :, tid]), zero(T))
        xTx = zero(T)
        @fastmath @simd for i = 2:nt
            xv  = @view xpad[i:n+i-1, tid]
            fp  = @view f[:, i-1, tid]
            fn  = @view f[:, i,   tid]
            xTx += xpad[n+i-1, tid]^2 - xpad[i-1, tid]^2
            ci   = λ2 + xTx
            ei   = xpad[n+i, tid] + sum(xv .* fp)
            αi   = ei / ci
            fn  .= fp .- αi .* xv
            α[i, tid] = αi
            c[i, tid] = ci
        end

        # Pass 2: reverse-time adjoint sweep
        fill!(@view(mpad[:, tid]), zero(T))
        mpad[n+1, tid] = δd[1, idx...]  # adjoint of: δd[1] = δm[1]
        fill!(@view(g[:, tid]), zero(T))

        for i = nt:-1:2
            xv  = @view xpad[i:n+i-1, tid]
            fp  = @view f[:, i-1, tid]
            fi  = @view f[:, i,   tid]
            gv  = @view g[:, tid]
            dv  = @view dav[:, tid]
            αi  = α[i, tid]
            ci  = c[i, tid]
            di  = δd[i, idx...]

            # Adjoint of application: δd[i] = δm_i + dot(δmv, fi) + dot(xv, δf_i)
            mpad[n+i, tid] += di
            dv             .= fi .* di       # adjoint from dot(δmv, fi)
            gv            .+= xv .* di       # adjoint from dot(xv, δf_i) → accumulate into g

            # Adjoint of δf update: δf_i = δf_{i-1} - δα_i*xv - α_i*δmv
            dα_bar = -sum(xv .* gv)          # adjoint from -δα_i * xv term
            dv    .-= αi .* gv               # adjoint from -α_i * δmv term

            # Adjoint of δα_i = (δe_i - α_i*δc_i) / c_i
            de_bar = dα_bar / ci
            dc_bar = -αi * dα_bar / ci

            # Adjoint of δe_i = δm_i + dot(xv, δf_{i-1}) + dot(δmv, fp)
            mpad[n+i, tid] += de_bar
            gv             .+= de_bar .* xv  # propagates g to step i-1
            dv             .+= de_bar .* fp  # adjoint from dot(δmv, fp)

            # Adjoint of δc_i = 2*dot(xv, δmv)
            dv .+= 2 .* xv .* dc_bar

            # Scatter all δmv contributions into mpad
            mpad[i:n+i-1, tid] .+= dv
        end

        δm[:, idx...] .= mpad[n+1:end, tid]
    end
    δm
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

        fbar = zeros(T, n1, n2)

        for i2 = 1:nx
            for i1 = 1:nt
                # Update PEF coefficients from x
                xv = @view(xpad[n1+i1-div(n1,2):2*n1+i1-1-div(n1,2),i2+1:n2+i2,tid])
                f1v = @view(f1[:,:,tid])
                f2v = @view(f2[:,:,i1,tid])
                @views @. fbar = (λ12 * f1v + λ22 .* f2v) / λ
                xTx = sum(xv.^2) - sum(xv[div(n1,2)+1:end,end].^2)
                xTf = sum(xv .* fbar)
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

