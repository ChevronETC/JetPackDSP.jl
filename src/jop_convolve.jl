"""
    A = JopConvolve(dom, rng, h [, optional parameters])

`A` is an n-dimension convolution (using the filter `h::Array`) operator with domain
`dom::JetSpace` and range `rng::JetSpace`, and with the following optional named arguments:

* `x0` is a tuple defining the origin of the upper-left corner of `h`

* `dx` is a tuple defining the spacing along each dimension of `h`

# Examples

## 1D, causal
```julia
A = JopConvolve(JetSpace(Float64,128), JetSpace(Float64,128), rand(32))
m = zeros(domain(A))
m[64] = 1.0
d = A*m
```

## 1D, zero-phase
```julia
A = JopConvolve(JetSpace(Float64,128), JetSpace(Float64,128), rand(32), dx=(1.0,), x0=(-16.0,))
m = zeros(domain(A))
m[64] = 1.0
d = A*m
```

## 2D, zero-phase
```julia
A = JopConvolve(JetSpace(Float64,128,128), JetSpace(Float64,128,128), rand(32,32), dx=(1.0,1.0), x0=(-16.0,-16.0))
m = zeros(domain(A))
m[64,64] = 1.0
d = A*m
```

# Notes
* It is often the case that the domain and range of the convolution operator are the same.  For this use-case, we provide
a convenience method for construction the operator:

    A = JopConvolve(spc, h [, optional parameters])

where `spc::JetSpace` and is used for both `dom` and `rng`.

* Since smoothing is a common use-case for JopConvolve, we provide a convenience method for creating `A` specific
to n-dimensional smoothing:

    A = JopConvolve(spc [, optional arguments])

where the optional arguments and their default values are:

* `smoother=:gaussian` choose between `:gaussian`, `:triang` and `:rect`

* `n=(128,)` choose the size of the smoothing window in each dimension.  If `length(n)=1`, then we assume a square window.

* `sigma=(0.5,)` for a gaussian window choose the shape of the window.  If `length(sigma)=1`, then we assume the same shape in each dimension.

# 2D Smoothing Example
```julia
P = JopPad(JetSpace(Float64,256,256), -10:256+11, -10:256+11, extend=true)
S = JopConvolve(range(P), smoother=:rect, n=(1,1))
R = JopPad(JetSpace(Float64,256,256), -10:256+11, -10:256+11, extend=false)
m = rand(domain(P))
d = R'∘S∘P*m
```
"""
function JopConvolve(dom::JetSpace{T,N}, rng::JetSpace{T,N}, h::AbstractArray{T,N}; x0=(0.0,), dx=(1.0,)) where {T<:Real,N}
    x0 = (length(x0) == 1 ? ntuple(i->x0[1], N) : x0)::NTuple{N,T}
    dx = (length(dx) == 1 ? ntuple(i->dx[1], N) : dx)::NTuple{N,T}

    length(x0) == N || error("Expected length(x0)=1 or length(x0)=ndims(h), got length(x0)=$(length(x0))")
    length(dx) == N || error("Expected length(dx)=1 or length(dx)=ndims(h), got length(dx)=$(length(dx))")

    nker = size(h)
    ndom = size(dom)
    nrng = size(rng)
    ntot = ntuple(idim->nextprod((2,3,5,7), nker[idim] + ndom[idim] - 1), N)::NTuple{N,Int}

    ntot < nrng && error("Range must be greater than or equal size(model)+size(h)-1")

    nfft = ntuple(idim->idim == 1 ? div(ntot[idim],2) + 1 : ntot[idim], N)::NTuple{N,Int}
    dfft = ntuple(idim->2*T(pi)/dx[idim]/ntot[idim], N)::NTuple{N,T}

    k = Vector{Vector{T}}(undef, N)
    k[1] = dfft[1]*collect(0:nfft[1]-1)
    for idim = 2:N
        nfft_p = div(nfft[idim]+1, 2)
        nfft_n = div(nfft[idim], 2)
        k[idim] = dfft[idim]*[collect(0:nfft_p-1) ; collect(-nfft_n:1:-1)]
    end

    hpad = zeros(T, ntot)
    for i in CartesianIndices(nker)
        hpad[i] = h[i]
    end

    H = rfft(hpad)

    phaseshift!(H, k, x0)

    JopLn(dom = dom, rng = rng, df! = JopConvolve_df!, df′! = JopConvolve_df′!,
        s = (mpad=Array{T,N}(undef,ntot), dpad=Array{T,N}(undef,ntot), H=H))
end
JopConvolve(spc::JetSpace, h::Array; x0=(0.0,), dx=(1.0,)) = JopConvolve(spc, spc, h, x0=x0, dx=dx)

# Convenience constructor for n-dimensional smoothing
function JopConvolve(spc::JetSpace{T,N}; smoother=:gaussian, n=(128,), sigma=(0.5,)) where {T,N}
    if smoother != :gaussian && smoother != :triang && smoother != :rect
        error("expected smoother=:gaussian, smoother=:triang or smoother=:rect, got smoother=$(smoother)")
    end
    n = length(n) == 1 ? ntuple(idim->2*n[1]+1, N) : ntuple(idim->2*n[idim]+1, N)
    sigma = length(sigma) == 1 ? ntuple(idim->sigma[1], N) : sigma
    dx = ntuple(idim->one(T), N)
    x0 = ntuple(idim->-one(T)*div(n[idim]-1,2), N)

    w = Vector{Vector{T}}(undef, N)
    for idim = 1:N
        if smoother == :gaussian
            w[idim] = gaussian(n[idim],sigma[idim])
        elseif smoother == :triang
            w[idim] = triang(n[idim])
        elseif smoother == :rect
            w[idim] = rect(n[idim])
        end
    end

    h = filter!(ones(T, n), w)

    JopConvolve(spc, spc, h, x0=x0, dx=dx)
end

export JopConvolve

function phaseshift!(H::Array{T,N}, k, x0) where {T,N}
    for i in CartesianIndices(H)
        f = zero(T)
        for idim = 1:N
            f += k[idim][i[idim]]*x0[idim]
        end
        H[i] *= @fastmath exp(-im*f)
    end
end

function filter!(h::Array{T,N}, w) where {T,N}
    for i in CartesianIndices(h)
        h[i] = 1
        for idim = 1:N
            h[i] *= w[idim][i[idim]]
        end
    end
    h ./= sum(h)
    h
end

function JopConvolve_df!(d::AbstractArray{T,N}, m::AbstractArray{T,N}; mpad, dpad, H, kwargs...) where {T<:Real,N}
    ndom = size(m)
    nrng = size(d)
    nfft = size(H)
    ntot = size(mpad)
    mpad .= 0
    for i in CartesianIndices(ndom)
        mpad[i] = m[i]
    end
    M = rfft(mpad)
    D = Array{Complex{T}}(undef, nfft)
    for i = 1:length(D)
        D[i] = H[i]*M[i]
    end
    dd = irfft(D, ntot[1])
    for i in CartesianIndices(nrng)
        d[i] = dd[i]
    end
    d
end

function JopConvolve_df′!(m::AbstractArray{T,N}, d::AbstractArray{T,N}; mpad, dpad, H, kwargs...) where {T<:Real,N}
    ndom = size(m)
    nrng = size(d)
    nfft = size(H)
    ntot = size(mpad)
    dpad .= 0
    for i in CartesianIndices(nrng)
        dpad[i] = d[i]
    end
    D = rfft(dpad)
    M = Array{Complex{T}}(undef, nfft)
    for i = 1:length(M)
        M[i] = conj(H[i]) * D[i]
    end
    mm = irfft(M, ntot[1])
    for i in CartesianIndices(ndom)
        m[i] = mm[i]
    end
    m
end
