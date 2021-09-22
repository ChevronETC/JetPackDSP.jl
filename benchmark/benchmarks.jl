using BenchmarkTools, JetPackDSP, Jets, LinearAlgebra

const SUITE = BenchmarkGroup()

s = JetSpace(Float64, 50, 51, 52)
n = 5
A = JopConvolve(s, smoother=:rect, n=(n,n,n))
m = rand(domain(A))
d = rand(range(A))

SUITE["JopConvolve"] = BenchmarkGroup()
SUITE["JopConvolve"]["construct smoother"] = @benchmarkable JopConvolve($s; smoother=:rect, n=($n,$n,$n))
SUITE["JopConvolve"]["forward"] = @benchmarkable mul!($d, $A, $m)
SUITE["JopConvolve"]["adjoint"] = @benchmarkable mul!($m, ($A)', $d)

SUITE
