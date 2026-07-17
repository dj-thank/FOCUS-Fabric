[CmdletBinding()]
param(
    [ValidateSet("preflight", "dry-run", "execute")]
    [string]$Mode = "preflight",
    [string]$Hypothesis = "H001-forward-influence-routing",
    [string]$Codex = "codex",
    [string]$Output,
    [switch]$AutoPromote
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python is missing: $Python. Run the repository setup first."
}
if ($AutoPromote -and $Mode -ne "execute") {
    throw "-AutoPromote is valid only with -Mode execute."
}

$RunnerArguments = @(
    "scripts/autonomy/run_codex_loop.py",
    "--mode", $Mode,
    "--hypothesis", $Hypothesis,
    "--max-hypotheses", "1",
    "--codex", $Codex
)
if ($Output) {
    $RunnerArguments += @("--output", $Output)
}
if ($AutoPromote) {
    $RunnerArguments += "--auto-promote"
}

Push-Location $Root
try {
    & $Python @RunnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Autonomy pipeline failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
