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

function JopFocusing1D_df!(d::AbstractArray{T}, m::AbstractArray{T}; weights, alpha, conserve_energy, kwargs...) where {T<:AbstractFloat}
    d = focusing_forward(m, weights, alpha, conserve_energy)
    d
end

function JopFocusing1D_df′!(m::AbstractArray{T}, d::AbstractArray{T}; weights, alpha, conserve_energy, kwargs...) where {T<:AbstractFloat}
    m = focusing_adjoint(d, weights, alpha, conserve_energy)
    m
end

function focusing_forward(m::AbstractArray{T},
                  weights::AbstractArray{T},
                  alpha::Real=1,
                  conserve_energy::Bool=false) where {T<:AbstractFloat}
    nt = size(m, 1)
    alphaT = T(abs(alpha))
    t = (1:nt) .- div(nt+1, 2)
    T_half = T(div(nt, 2))

    trailing_shape = size(m)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    # Output array
    d = similar(m)

    # Reshape weights for broadcasting
    abs_t_over_T = reshape(abs.(t) ./ T_half, nt, ntuple(_ -> 1, ndims(weights)-1)...)
    t_vec        = reshape(t,                nt, ntuple(_ -> 1, ndims(weights)-1)...)

    @inbounds begin
        @threads for I in trailing_inds
        # for I in trailing_inds
            idx = I.I
            wgt = weights[1, idx...]

            # scaled time axis for this trace
            t_new = t_vec[:, ones(Int, ndims(weights)-1)...] .* (1 .+ alphaT .* abs_t_over_T[:, ones(Int, ndims(weights)-1)...] .* wgt)
            pos = clamp.(t_new .+ T_half .+ 1, 1, nt)

            idx_low  = clamp.(floor.(pos), 1, nt)
            idx_high = clamp.(idx_low .+ 1, 1, nt)
            w = pos .- idx_low

            idx_low_i  = Int.(idx_low)
            idx_high_i = Int.(idx_high)

            # Jacobian scaling for this trace
            jac = 1 .+ 2 .* alphaT .* abs_t_over_T[:, ones(Int, ndims(weights)-1)...] .* wgt
            s   = conserve_energy ? sqrt.(jac) : ones(T, nt)

            xlow  = m[idx_low_i, idx...]
            xhigh = m[idx_high_i, idx...]
            d[:, idx...] .= s .* ((1 .- w) .* xlow .+ w .* xhigh)
        end
    end
    d
end

function focusing_adjoint(d::AbstractArray{T},
                  weights::AbstractArray{T},
                  alpha::Real=1,
                  conserve_energy::Bool=false) where {T<:AbstractFloat}
    nt = size(d, 1)
    alphaT = T(abs(alpha))
    t = (1:nt) .- div(nt+1, 2)
    T_half = T(div(nt, 2))

    trailing_shape = size(d)[2:end]
    trailing_inds = CartesianIndices(trailing_shape)

    # Output array
    m = similar(d)
    fill!(m, zero(T))

    # Reshape weights for broadcasting
    abs_t_over_T = reshape(abs.(t) ./ T_half, nt, ntuple(_ -> 1, ndims(weights)-1)...)
    t_vec        = reshape(t,                nt, ntuple(_ -> 1, ndims(weights)-1)...)

    @inbounds begin
        @threads for I in trailing_inds
        # for I in trailing_inds
            idx = I.I
            wgt = weights[1, idx...]

            # scaled time axis for this trace
            t_new = t_vec[:, ones(Int, ndims(weights)-1)...] .* (1 .+ alphaT .* abs_t_over_T[:, ones(Int, ndims(weights)-1)...] .* wgt)
            pos = clamp.(t_new .+ T_half .+ 1, 1, nt)

            idx_low  = clamp.(floor.(pos), 1, nt)
            idx_high = clamp.(idx_low .+ 1, 1, nt)
            w = pos .- idx_low

            idx_low_i  = Int.(idx_low)
            idx_high_i = Int.(idx_high)

            # Jacobian scaling for this trace
            jac = 1 .+ 2 .* alphaT .* abs_t_over_T[:, ones(Int, ndims(weights)-1)...] .* wgt
            s   = conserve_energy ? sqrt.(jac) : ones(T, nt)

            xlow  = d[idx_low_i, idx...]
            xhigh = d[idx_high_i, idx...]

            @inbounds for it in 1:nt
                m[idx_low_i[it], idx...] += s[it] * (1 - w[it]) * d[it, idx...]
                m[idx_high_i[it], idx...] += s[it] * w[it] * d[it, idx...]
            end
        end
    end
    m
end