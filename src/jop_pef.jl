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
The linearization and adjoint implementation below account for missing term correctly.
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

        @fastmath for i = 2:nt
            # Update PEF coefficients from x
            xv = @view xpad[i:n+i-1, tid]
            fv = @view f[:, tid]
            xTx += xpad[n+i-1, tid]^2 - xpad[i-1, tid]^2
            xTf = @inbounds sum(xv[k]*fv[k] for k in eachindex(xv))
            fv .-= (xpad[n+i, tid] + xTf) / (λ2 + xTx) .* xv
            
            # Apply PEF to m
            d[i, idx...] = xpad[n+i, tid] + @inbounds sum(xv[k]*fv[k] for k in eachindex(xv))
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

        @fastmath for i = 2:nt
            xv  = @view xpad[i:n+i-1, tid]     # window of mₒ (= x in f!)
            δmv = @view δxpad[i:n+i-1, tid]    # window of δm
            fv  = @view f[:, tid]               # f_{i-1}
            δfv = @view δf[:, tid]              # δf_{i-1}

            xTx += xpad[n+i-1, tid]^2 - xpad[i-1, tid]^2
            ci   = λ2 + xTx
            ei   = xpad[n+i, tid] + @inbounds sum(xv[k]*fv[k] for k in eachindex(xv))
            αi   = ei / ci

            # Linearize filter update: δf_i = δf_{i-1} - δα_i*xv - α_i*δmv
            # (must use fv = f_{i-1} and δfv = δf_{i-1} before either is updated)
            δei  = δxpad[n+i, tid] + @inbounds(sum(xv[k]*δfv[k] for k in eachindex(xv))) + @inbounds(sum(δmv[k]*fv[k] for k in eachindex(δmv)))
            δci  = 2 * @inbounds sum(xv[k]*δmv[k] for k in eachindex(xv))
            δαi  = (δei - αi * δci) / ci
            δfv .-= δαi .* xv .+ αi .* δmv    # δf_i  (updated before fv)
            fv  .-= αi .* xv                   # f_i   (now fv = f_i)

            # Apply linearized PEF to δm: δd[i] = δm_i + dot(δmv, f_i) + dot(xv, δf_i)
            δd[i, idx...] = δxpad[n+i, tid] + @inbounds(sum(δmv[k]*fv[k] for k in eachindex(δmv))) + @inbounds(sum(xv[k]*δfv[k] for k in eachindex(xv)))
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
        @fastmath for i = 2:nt
            xv  = @view xpad[i:n+i-1, tid]
            fp  = @view f[:, i-1, tid]
            fn  = @view f[:, i,   tid]
            xTx += xpad[n+i-1, tid]^2 - xpad[i-1, tid]^2
            ci   = λ2 + xTx
            ei   = xpad[n+i, tid] + @inbounds sum(xv[k]*fp[k] for k in eachindex(xv))
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
            dα_bar = -@inbounds sum(xv[k]*gv[k] for k in eachindex(xv))  # adjoint from -δα_i * xv term
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
    A = JopStreamingPEF2D(; dom, n1=7, n2=3, λ1=1e-1, λ2=1e-1)

Apply a 2D streaming prediction error filter (PEF) along the first and second dimensions. \\
The filter is estimated and applied on the fly.

Arguments:
- 'dom :: JetSpace{T}': domain (and range) of the operator
- 'n1,n2 :: Int': number of active samples in the first,second dimensions of the PEF (default: 7,3). The leading sample of the PEF is always 1 (implicitly) and is centered at the first column.
- 'λ1,λ2 :: Real': regularization parameters for the first and second dimensions of the PEF estimation (default: 1e-1,1e-1)

Forward mode returns:
- 'd :: AbstractArray{T}` containing filtered input with same shape as `dom`

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
function JopStreamingPEF2D(; dom::JetSpace{T}, n1::Int=7, n2::Int=3, λ1::Real=1e-1, λ2::Real=1e-1) where {T<:AbstractFloat}
    @assert n1 > 0 "n1 must be strictly greater than 0"
    @assert n2 > 0 "n2 must be strictly greater than 0"
    @assert λ1^2 + λ2^2 > 0 "λ1^2 + λ2^2 must be strictly greater than 0"
    @assert ndims(dom) > 1 "number of dimensions of `dom` must be greater than 1"
    rng = dom

    Jet(; dom, rng, f! = JopStreamingPEF2D_f!, df! = JopStreamingPEF2D_df!, df′! = JopStreamingPEF2D_df′!, s = (; n1=n1, n2=n2, λ1=T(λ1), λ2=T(λ2)))
end

JopNlStreamingPEF2D(;kwargs...) = JopNl(JopStreamingPEF2D(;kwargs...))
JopLnStreamingPEF2D(; v, kwargs...) = JopLn(JopStreamingPEF2D(;kwargs...), v)

export JopNlStreamingPEF2D
export JopLnStreamingPEF2D

function JopStreamingPEF2D_f!(d::AbstractArray{T}, m::AbstractArray{T}; kwargs...) where {T<:AbstractFloat}
    nt = size(m, 1)
    nx = size(m, 2)
    n1 = kwargs[:n1]
    n2 = kwargs[:n2]
    λ12 = kwargs[:λ1]^2
    λ22 = kwargs[:λ2]^2
    λ = λ12 + λ22

    trailing_shape = size(m)[3:end]
    trailing_inds = CartesianIndices(trailing_shape)

    # Initialize filter coefficients to zero (leading coeff is always 1)
    f1   = zeros(T, n1, n2, Threads.maxthreadid())
    f2   = zeros(T, n1, n2, nt, Threads.maxthreadid()) # store filter from previous trace
    fbar = zeros(T, n1, n2, Threads.maxthreadid())     # blended filter (thread-local)

    # holder for padded array with filter length
    ntpad = 2*n1 + nt
    nxpad = n2 + nx
    xpad = zeros(T, ntpad, nxpad, Threads.maxthreadid())

    hn = div(n1, 2)  # half-width (inactive tap offset)

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()
        
        xpad[n1+1:n1+nt, n2+1:end, tid] .= m[:,:,idx...]

        fill!(@view(f2[:, :, :, tid]), zero(T))

        fbv = @view fbar[:, :, tid]

        for i2 = 1:nx
            fill!(@view(f1[:, :, tid]), zero(T))
            for i1 = 1:nt
                # Update PEF coefficients from x
                xv  = @view xpad[n1+i1-hn:2*n1+i1-1-hn, i2+1:n2+i2, tid]
                f1v = @view f1[:, :, tid]
                f2v = @view f2[:, :, i1, tid]
                @. fbv = (λ12 * f1v + λ22 * f2v) / λ
                sv  = @view xv[hn+1:end, end]
                xTx = dot(xv, xv) - dot(sv, sv)
                xTf = dot(xv, fbv)
                f1v .= fbv .- (xpad[n1+i1,n2+i2,tid] + xTf) / (λ + xTx) .* xv
                f1v[hn+1:end, end] .= 0  # zero out inactive coeffs
                @view(f2[:,:,i1,tid]) .= f1v  # store current filter for next trace
                
                # Apply PEF to m
                d[i1,i2,idx...] = xpad[n1+i1,n2+i2,tid] + dot(xv, f1v)
            end
        end
    end
    d
end

function JopStreamingPEF2D_df!(δd::AbstractArray{T}, δm::AbstractArray{T}; kwargs...) where {T<:AbstractFloat}
    mₒ = kwargs[:mₒ]
    nt  = size(mₒ, 1)
    nx  = size(mₒ, 2)
    n1  = kwargs[:n1]
    n2  = kwargs[:n2]
    λ12 = kwargs[:λ1]^2
    λ22 = kwargs[:λ2]^2
    λ   = λ12 + λ22

    trailing_shape = size(δm)[3:end]
    trailing_inds  = CartesianIndices(trailing_shape)

    # filter at step (i1-1, i2) rolling, and linearized filter perturbations
    f1   = zeros(T, n1, n2, Threads.maxthreadid())
    f2   = zeros(T, n1, n2, nt, Threads.maxthreadid())  # previous-trace filter history
    δf1  = zeros(T, n1, n2, Threads.maxthreadid())
    δf2  = zeros(T, n1, n2, nt, Threads.maxthreadid())  # previous-trace δf history

    # holders for padded arrays with filter length
    ntpad  = 2*n1 + nt
    nxpad  = n2 + nx
    xpad   = zeros(T, ntpad, nxpad, Threads.maxthreadid())
    δxpad  = zeros(T, ntpad, nxpad, Threads.maxthreadid())

    fbar  = zeros(T, n1, n2, Threads.maxthreadid())
    δfbar = zeros(T, n1, n2, Threads.maxthreadid())

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()

        xpad[n1+1:n1+nt,  n2+1:end, tid] .= mₒ[:,:,idx...]
        δxpad[n1+1:n1+nt, n2+1:end, tid] .= δm[:,:,idx...]

        fill!(@view(f2[:, :, :,  tid]), zero(T))
        fill!(@view(δf2[:, :, :, tid]), zero(T))

        fbv  = @view fbar[:, :, tid]
        δfbv = @view δfbar[:, :, tid]
        hn = div(n1, 2)

        for i2 = 1:nx
            fill!(@view(f1[:, :,  tid]), zero(T))
            fill!(@view(δf1[:, :, tid]), zero(T))
            for i1 = 1:nt
                xv   = @view xpad[n1+i1-hn:2*n1+i1-1-hn, i2+1:n2+i2, tid]
                δmv  = @view δxpad[n1+i1-hn:2*n1+i1-1-hn, i2+1:n2+i2, tid]
                f1v  = @view f1[:, :, tid]
                f2v  = @view f2[:, :, i1, tid]
                δf1v = @view δf1[:, :, tid]
                δf2v = @view δf2[:, :, i1, tid]

                sv   = @view xv[hn+1:end, end]
                xTx  = dot(xv, xv) - dot(sv, sv)
                @. fbv = (λ12 * f1v + λ22 * f2v) / λ
                xTf  = dot(xv, fbv)
                ei   = xpad[n1+i1, n2+i2, tid] + xTf
                ci   = λ + xTx
                αi   = ei / ci

                # Linearized fbar: δfbar = (λ12 * δf1v + λ22 * δf2v) / λ
                @. δfbv = (λ12 * δf1v + λ22 * δf2v) / λ

                # Linearized filter update
                δsmv = @view δmv[hn+1:end, end]
                δsxv = @view xv[hn+1:end, end]
                δei  = δxpad[n1+i1, n2+i2, tid] + dot(xv, δfbv) + dot(δmv, fbv)
                δci  = 2 * dot(xv, δmv) - 2 * dot(δsmv, δsxv)
                δαi  = (δei - αi * δci) / ci
                δf1v .= δfbv .- δαi .* xv .- αi .* δmv
                δf1v[hn+1:end, end] .= 0
                f1v  .= fbv .- αi .* xv
                f1v[hn+1:end, end] .= 0

                @view(f2[:, :, i1, tid])  .= f1v
                @view(δf2[:, :, i1, tid]) .= δf1v

                # Apply linearized PEF
                δd[i1,i2,idx...] = δxpad[n1+i1,n2+i2,tid] + dot(δmv, f1v) + dot(xv, δf1v)
            end
        end
    end
    δd
end

function JopStreamingPEF2D_df′!(δm::AbstractArray{T}, δd::AbstractArray{T}; kwargs...) where {T<:AbstractFloat}
    mₒ  = kwargs[:mₒ]
    nt  = size(mₒ, 1)
    nx  = size(mₒ, 2)
    n1  = kwargs[:n1]
    n2  = kwargs[:n2]
    λ12 = kwargs[:λ1]^2
    λ22 = kwargs[:λ2]^2
    λ   = λ12 + λ22

    trailing_shape = size(δm)[3:end]
    trailing_inds  = CartesianIndices(trailing_shape)

    # filter history + RLS scalars (needed for exact adjoint)
    f1    = zeros(T, n1, n2, nt, nx, Threads.maxthreadid())
    f2    = zeros(T, n1, n2, nt,     Threads.maxthreadid())  # previous-trace filter (rolling)
    fbar  = zeros(T, n1, n2, nt, nx, Threads.maxthreadid())
    fprev = zeros(T, n1, n2,         Threads.maxthreadid())  # rolling fprev for Pass 1
    α     = zeros(T, nt, nx,         Threads.maxthreadid())
    c     = zeros(T, nt, nx,         Threads.maxthreadid())

    # holders for padded arrays with filter length
    ntpad = 2*n1 + nt
    nxpad = n2 + nx
    xpad  = zeros(T, ntpad, nxpad, Threads.maxthreadid())
    mpad  = zeros(T, ntpad, nxpad, Threads.maxthreadid())

    # adjoint of δf1 chain (propagated backward in i1), and of δf2 (propagated backward in i2)
    g1   = zeros(T, n1, n2, Threads.maxthreadid())
    g2   = zeros(T, n1, n2, nt, Threads.maxthreadid())
    dav  = zeros(T, n1, n2, Threads.maxthreadid())

    fill!(δm, zero(T))

    # loop over trailing dims
    @inbounds @threads for I in trailing_inds
        idx = I.I
        tid = Threads.threadid()

        xpad[n1+1:n1+nt, n2+1:end, tid] .= mₒ[:,:,idx...]

        fill!(@view(f2[:, :, :,    tid]), zero(T))
        fill!(@view(g2[:, :, :,    tid]), zero(T))

        hn = div(n1, 2)

        # Pass 1: reconstruct full filter history + α, c from mₒ
        for i2 = 1:nx
            fill!(@view(f1[:, :, :, i2, tid]), zero(T))
            fpv = @view fprev[:, :, tid]
            fill!(fpv, zero(T))  # fprev = 0 (cold start)
            for i1 = 1:nt
                xv   = @view xpad[n1+i1-hn:2*n1+i1-1-hn, i2+1:n2+i2, tid]
                f2v  = @view f2[:, :, i1, tid]
                sv   = @view xv[hn+1:end, end]
                xTx  = dot(xv, xv) - dot(sv, sv)
                @. fbar[:,:,i1,i2,tid] = (λ12 * fpv + λ22 * f2v) / λ
                fbv  = @view fbar[:, :, i1, i2, tid]
                xTf  = dot(xv, fbv)
                ei   = xpad[n1+i1, n2+i2, tid] + xTf
                ci   = λ + xTx
                αi   = ei / ci
                f1v  = @view f1[:, :, i1, i2, tid]
                f1v .= fbv .- αi .* xv
                f1v[hn+1:end, end] .= 0
                f2[:, :, i1, tid] .= f1v
                α[i1, i2, tid] = αi
                c[i1, i2, tid] = ci
                fpv = f1v  # fprev for next step (view into stored history, no allocation)
            end
        end

        # Pass 2: reverse-time adjoint sweep (backward in both i2 and i1)
        fill!(@view(mpad[:, :, tid]), zero(T))
        fill!(@view(g2[:, :, :, tid]), zero(T))

        for i2 = nx:-1:1
            fill!(@view(g1[:, :, tid]), zero(T))
            for i1 = nt:-1:1
                xv   = @view xpad[n1+i1-hn:2*n1+i1-1-hn, i2+1:n2+i2, tid]
                fbv  = @view fbar[:, :, i1, i2, tid]
                fi   = @view f1[:, :, i1, i2, tid]
                g1v  = @view g1[:, :, tid]
                g2v  = @view g2[:, :, i1, tid]
                dv   = @view dav[:, :, tid]
                αi   = α[i1, i2, tid]
                ci   = c[i1, i2, tid]
                di   = δd[i1, i2, idx...]

                # Combine gradients from i1-backward and i2-backward chains
                g1v .+= g2v
                g1v[hn+1:end, end] .= 0

                # Adjoint of application
                mpad[n1+i1, n2+i2, tid] += di
                dv   .= fi .* di
                g1v .+= xv .* di
                g1v[hn+1:end, end] .= 0

                # Adjoint of δf1 update
                dα_bar = -dot(xv, g1v)
                dv    .-= αi .* g1v

                de_bar = dα_bar / ci
                dc_bar = -αi * dα_bar / ci

                # Adjoint of δe_i
                mpad[n1+i1, n2+i2, tid] += de_bar
                g1v .+= de_bar .* xv
                g1v[hn+1:end, end] .= 0
                dv  .+= de_bar .* fbv

                # Adjoint of δc_i
                sv   = @view xv[hn+1:end, end]
                dsv  = @view dv[hn+1:end, end]
                dv  .+= 2 .* xv .* dc_bar
                dsv .-= 2 .* sv .* dc_bar

                # Adjoint of δfbar blending: overwrite g2v, scale g1v
                g2v .= (λ22 / λ) .* g1v
                g1v .*= (λ12 / λ)

                mpad[n1+i1-hn:2*n1+i1-1-hn, i2+1:n2+i2, tid] .+= dv
            end
        end

        δm[:, :, idx...] .= mpad[n1+1:n1+nt, n2+1:end, tid]
    end
    δm
end