"""
    F = JopEnvelope(spc[, power=0.5, damping=0.0])

where `F` is the envelope operator with doman and range given by `spc::JetSpace`.
The Envelope is taken along the fastest dimension of the space.  For example, if
`spc=JetSpace{Float64,10,11}` and `A=rand(spc)`, then the envelope would be along each
column of `A`.

The envelope of `d` is computed as: `(d^2 + (Hd)^2 + damping)^(power/2)` where `Hd` is the Hilbert
transform of `d`.  The evelope is computed when `power=1`.  If `power=2`, then the square
of the envelope is computed, and so on.

# Notes
The passed in power is multiplied by 1/2, and is the power for simple envelope,
applied to the sum of the squares (d^2 + (Hd)^2). For example:

    power = 2.0 : (d^2 + (H d)^2)^1

If any trace is entirely hard zeros, power < 2, and damping is not > 0, you will get Indian Bread (Nan)
"""
function JopEnvelope(spc::JetSpace{T,N}, power::Real=1.0; damping::Real=eps(T)) where {T,N}
    JopNl(dom = spc, rng = spc, f! = JopEnvelope_f!, df! = JopEnvelope_df!, df′! = JopEnvelope_df′!,
        upstate! = JopEnvelope_upstate!, s = (power=T(power/2), damping=T(damping), eu=zeros(spc), hu=zeros(spc)))
end
export JopEnvelope

function JopEnvelope_hu_and_eu!(m::AbstractArray, power, damping, hu, eu)
    hu .= imag.(hilbert(m))
    eu .= m.^2 .+ hu.^2 .+ damping
    #if (any(eu .== 0)) 
    #    @error "Zero eu value. Possibly envelope is zero and damping is also set to zero or something negative! \n
    #                Set damping to a small positive value"
    #end
end

JopEnvelope_upstate!(m::AbstractArray, s::NamedTuple) = JopEnvelope_hu_and_eu!(m, s.power, s.damping, s.hu, s.eu)

function JopEnvelope_f!(d::AbstractArray, m::AbstractArray; power, damping, hu, eu)
    JopEnvelope_hu_and_eu!(m, power, damping, hu, eu)
    d .= eu.^power
end

# (1/e) [diag(u) + diag(Hu) H] δ
# (1/e) [u ∘ δ + Hu ∘ H δ]
function JopEnvelope_df!(d::AbstractArray, m::AbstractArray; mₒ, power, hu, eu, kwargs...)
    d .= (power * 2) .* (mₒ .* m .+ hu .* imag.(hilbert(m))) .* (eu.^(power - 1))
end

# [diag(u) - H diag(Hu)] (1/e) δ
# (u ∘ δ)/e - H(Hu ∘ δ/e)
function JopEnvelope_df′!(m::AbstractArray, d::AbstractArray; mₒ, hu, eu, power, kwargs...)
    Ed = (power * 2) .* d .* (eu.^(power - 1))
    m .= mₒ .* Ed .- imag(hilbert(hu .* Ed))
end
