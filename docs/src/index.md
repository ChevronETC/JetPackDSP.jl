
# JetPackDSP - Jets.jl DSP operator pack

- [`JetPackDSP.JopConvolve`](#JetPackDSP.JopConvolve)
- [`JetPackDSP.JopEnvelope`](#JetPackDSP.JopEnvelope)
- [`JetPackDSP.JopFilter`](#JetPackDSP.JopFilter)

# JetPackDSP.JopConvolve
```julia
A = JopConvolve(dom, rng, h [, optional parameters])
```

`A` is an n-dimension convolution (using the filter `h::Array`) operator with domain `dom::JetSpace` and range `rng::JetSpace`, and with the following optional named arguments:

  * `x0` is a tuple defining the origin of the upper-left corner of `h`
  * `dx` is a tuple defining the spacing along each dimension of `h`

**Examples**

**1D, causal**

```julia
using Jets, JetPackDSP
A = JopConvolve(JetSpace(Float64,128), JetSpace(Float64,128), rand(32))
m = zeros(domain(A))
m[64] = 1.0
d = A*m
```

**1D, zero-phase**

```julia
A = JopConvolve(JetSpace(Float64,128), JetSpace(Float64,128), rand(32), dx=(1.0,), x0=(-16.0,))
m = zeros(domain(A))
m[64] = 1.0
d = A*m
```

**2D, zero-phase**

```julia
A = JopConvolve(JetSpace(Float64,128,128), JetSpace(Float64,128,128), rand(32,32), dx=(1.0,1.0), x0=(-16.0,-16.0))
m = zeros(domain(A))
m[64,64] = 1.0
d = A*m
```

**Notes**

  * It is often the case that the domain and range of the convolution operator are the same.  For this use-case, we provide

a convenience method for construction the operator:

```
A = JopConvolve(spc, h [, optional parameters])
```

where `spc::JetSpace` and is used for both `dom` and `rng`.

  * Since smoothing is a common use-case for JopConvolve, we provide a convenience method for creating `A` specific

to n-dimensional smoothing:

```
A = JopConvolve(spc [, optional arguments])
```

where the optional arguments and their default values are:

  * `smoother=:gaussian` choose between `:gaussian`, `:triang` and `:rect`
  * `n=(128,)` choose the size of the smoothing window in each dimension.  If `length(n)=1`, then we assume a square window.
  * `sigma=(0.5,)` for a gaussian window choose the shape of the window.  If `length(sigma)=1`, then we assume the same shape in each dimension.

**2D Smoothing Example**

```julia
using Jets, JetPack, JetPackDSP
P = JopPad(JetSpace(Float64,256,256), -10:256+11, -10:256+11, extend=true)
S = JopConvolve(range(P), smoother=:rect, n=(1,1))
R = JopPad(JetSpace(Float64,256,256), -10:256+11, -10:256+11, extend=false)
m = rand(domain(P))
d = R'∘S∘P*m
```

# JetPackDSP.JopEnvelope
```julia
F = JopEnvelope(spc[, power=0.5, damping=0.0])
```

where `F` is the envelope operator with doman and range given by `spc::JetSpace`. The Envelope is taken along the fastest dimension of the space.  For example, if `spc=JetSpace{Float64,10,11}` and `A=rand(spc)`, then the envelope would be along each column of `A`.

The envelope of `d` is computed as: `(d^2 + (Hd)^2 + damping)^(power/2)` where `Hd` is the Hilbert transform of `d`.  The evelope is computed when `power=1`.  If `power=2`, then the square of the envelope is computed, and so on.

**Notes**

The passed in power is multiplied by 1/2, and is the power for simple envelope, applied to the sum of the squares (d^2 + (Hd)^2). For example:

```
power = 2.0 : (d^2 + (H d)^2)^1
```

If power < 2, and damping is not > 0, you may get NaN when envelope value is zero Default damping factor is eps(T). If you know your traces are not zero, set damping=0 in constructor to avoid over damping.

**Example**

**1D**
```julia
using Jets, JetPackDSP
F = JopEnvelope(JetSpace(Float64,64))
m = -1 .+ 2*rand(domain(F))
d = F*m
```

# JetPackDSP.JopFilter
```julia
A = JopFilter(spc, responsetype, designmethod)
```

where `A` is a filter applied to a signal in `spc::JotSpace`, and built using `responsetype` and `designmethod`.  The `responsetype` and `designmethod` are described in `https://github.com/JuliaDSP/DSP.jl`. The filter is applied along the fast dimension of the space.

**Examples**

**1D**

```
using DSP, JetPackDSP, Jets
A = JopFilter(JetSpace(Float64,512), Highpass(10.0, fs=125), Butterworth(4))
d = A*rand(domain(A))
```

**2D**

```
using DSP, JetPackDSP, Jets
A = JopFilter(JetSpace(Float64,512,10), Highpass(10.0, fs=125), Butterworth(4))
d = A*rand(domain(A))
```

