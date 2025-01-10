module JetPackDSP

using DSP, FFTW, Jets, Base.Threads

include("jop_convolve.jl")
include("jop_envelope.jl")
include("jop_filter.jl")
include("jop_CubicSpline.jl")

end
