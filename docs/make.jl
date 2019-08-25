using Documenter, DocumenterMarkdown, JetPackDSP

makedocs(
    format = Markdown(),
    sitename = "JetPackDSP"
)
cp("build/README.md", "../README.md", force=true)
