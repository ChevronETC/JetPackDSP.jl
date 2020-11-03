# set random seed to promote repeatability in CI unit tests
using Random
Random.seed!(101)

for filename in (
        "jop_convolve.jl",
        "jop_envelope.jl",
        "jop_filter.jl")
    include(filename)
end
