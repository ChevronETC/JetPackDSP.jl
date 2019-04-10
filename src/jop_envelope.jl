"""
    F = JopEnvelope(spc[, power=0.5])

where `F` is the envelope operator with doman and range given by `spc::JetSpace`.
The Envelope is taken along the fastest dimension of the space.  For example, if
`spc=JetSpace{Float64,10,11}` and `A=rand(spc)`, then the envelope would be along each
column of `A`.

The envelope of `d` is computed as: `(d^2 + (Hd)^2)^(power/2)` where `Hd` is the Hilbert
transform of `d`.  The evelope is computed when `power=1`.  If `power=2`, then the square
of the envelope is computed, and so on.

# Notes
The passed in power is multiplied by 1/2, and is the power for simple envelope,
applied to the sum of the squares (d^2 + (Hd)^2). For example:

    power = 2.0 : (d^2 + (H d)^2)^1
"""
function JopEnvelope(spc::JetSpace{T,N}, power::Real=1.0) where {T,N}
    JopNl(dom = spc, rng = spc, f! = JopEnvelope_f!, df! = JopEnvelope_df!, df′! = JopEnvelope_df′!,
        s = (power=T(power/2), eu=zeros(spc), hu=zeros(spc)))
end
export JopEnvelope

function JopEnvelope_update!(m::AbstractArray, hu, eu)
    hu .= imag.(hilbert(m))
    eu .= m.^2 .+ hu.^2
end

function JopEnvelope_f!(d::AbstractArray, m::AbstractArray; power, hu, eu)
    JopEnvelope_update!(m, hu, eu)
    d .= eu.^power
end

# (1/e) [diag(u) + diag(Hu) H] δ
# (1/e) [u ∘ δ + Hu ∘ H δ]
function JopEnvelope_df!(d::AbstractArray, m::AbstractArray; mₒ, power, hu, eu)
    JopEnvelope_update!(mₒ, hu, eu)
    d .= (power * 2) .* (mₒ .* m .+ hu .* imag.(hilbert(m))) .* eu.^(power - 1)
end

# [diag(u) - H diag(Hu)] (1/e) δ
# (u ∘ δ)/e - H(Hu ∘ δ/e)
function JopEnvelope_df′!(m::AbstractArray, d::AbstractArray; mₒ, hu, eu, power)
    JopEnvelope_update!(mₒ, hu, eu)
    Ed = (power * 2) .* d .* eu.^(power - 1)
    m .= mₒ .* Ed .- imag(hilbert(hu .* Ed))
end
