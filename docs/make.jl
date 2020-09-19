using Documenter, JetPackDSP

makedocs(sitename="JetPackDSP", modules=[JetPackDSP])

deploydocs(
    repo = "github.com/ChevronETC/JetPackDSP.jl.git",
)

