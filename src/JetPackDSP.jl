module JetPackDSP

using DSP, FFTW, Jets

include("jop_convolve.jl")
include("jop_envelope.jl")
include("jop_filter.jl")

end
