"""
    A = JopFocusing1D(dom; weights=nothing, alpha=1, conserve_energy=false)

Apply a focusing operator along the first dimension (dim=1) to move energy towards the central sample.
The focusing consists of stretching and squeezing the coordinate axis 
t -> t * (1 + alpha * w * |t| / T) 

Arguments:
- 'dom :: JetSpace{T}': domain (and range) of the operator
- 'weights :: Union{AbstractArray{T}, Nothing}': weights to multiply alpha for each trace - preferrably >=0 and <= 1
- 'alpha :: Real': global focusing strength: 0 ⇒ identity 
- 'conserve_energy :: Bool': whether to conserve energy during focusing (default: false)

Forward mode returns:
- 'd :: AbstractArray{T}` containing focused data with same shape as `dom`
"""
function JopFocusing1D(dom::JetSpace{T}; weights::Union{AbstractArray{T}, Nothing}=nothing, alpha::Real=1, conserve_energy::Bool=false) where {T<:AbstractFloat}
    rng = dom
    trailing_shape = size(dom)[2:end]

    w = (weights === nothing ? ones(T, 1, trailing_shape...) : weights)
    @assert size(w)[1] == 1 "weights must have size 1 along dim=1"
    @assert size(w)[2:end] == trailing_shape "weights trailing dims must match dom"

    JopLn(; dom, rng, df! = JopFocusing1D_df!, df′! = JopFocusing1D_df′!, s = (; weights=w, alpha=alpha, conserve_energy=conserve_energy) )
end

export JopFocusing1D

function JopFocusing1D_df!(d::AbstractArray{T}, m::AbstractArray{T}; weights::AbstractArray{T}, alpha::Real, conserve_energy::Bool, kwargs...) where {T<:AbstractFloat}
    nt = size(m, 1)
    alphaT = T(abs(alpha))
    t = (1:nt) .- div(nt+1, 2)
    T_half = T(div(nt, 2))

    trailing_shape = size(m)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    # Precompute trace-independent quantities once
    abs_t_over_T = abs.(t) ./ T_half          # length-nt Vector

    @inbounds @threads for I in trailing_inds
        idx = I.I
        wgt = weights[1, idx...]

        # scaled time axis for this trace
        scale = alphaT * wgt
        s_energy = conserve_energy ? sqrt(1 + 2 * scale) : one(T)  # only needed if conserve_energy varies
        for it in 1:nt
            t_new = t[it] * (1 + scale * abs_t_over_T[it])
            pos   = clamp(t_new + T_half + 1, T(1), T(nt))
            il    = clamp(floor(Int, pos), 1, nt)
            ih    = clamp(il + 1, 1, nt)
            frac  = pos - il
            jac_t = conserve_energy ? sqrt(1 + 2 * scale * abs_t_over_T[it]) : one(T)
            d[it, idx...] = jac_t * ((1 - frac) * m[il, idx...] + frac * m[ih, idx...])
        end
    end
    d
end


function JopFocusing1D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; weights::AbstractArray{T}, alpha::Real, conserve_energy::Bool, kwargs...) where {T<:AbstractFloat}
    nt = size(d, 1)
    alphaT = T(abs(alpha))
    t = (1:nt) .- div(nt+1, 2)
    T_half = T(div(nt, 2))

    trailing_shape = size(d)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    fill!(m, zero(T))

    # Precompute trace-independent quantities once
    abs_t_over_T = abs.(t) ./ T_half          # length-nt Vector

    @inbounds @threads for I in trailing_inds
        idx = I.I
        wgt = weights[1, idx...]

        scale = alphaT * wgt
        for it in 1:nt
            t_new = t[it] * (1 + scale * abs_t_over_T[it])
            pos   = clamp(t_new + T_half + 1, T(1), T(nt))
            il    = clamp(floor(Int, pos), 1, nt)
            ih    = clamp(il + 1, 1, nt)
            frac  = pos - il
            jac_t = conserve_energy ? sqrt(1 + 2 * scale * abs_t_over_T[it]) : one(T)
            val   = jac_t * d[it, idx...]
            m[il, idx...] += (1 - frac) * val
            m[ih, idx...] +=      frac  * val
        end
    end
    m
end