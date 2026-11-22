. (Join-Path $PSScriptRoot 'set-local-env.ps1')

$ollamaRuntime = Join-Path $ragStorage 'ollama\runtime-v0.40.16'
$ollamaExecutable = Join-Path $ollamaRuntime 'ollama.exe'
$ollamaHome = Join-Path $ragStorage 'ollama\home'
$ollamaLogs = Join-Path $ragStorage 'ollama\logs'

if (-not (Test-Path -LiteralPath $ollamaExecutable -PathType Leaf)) {
    throw "Ollama executable not found: $ollamaExecutable"
}

New-Item -ItemType Directory -Force -Path `
    $ollamaHome, $ollamaLogs | Out-Null

$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NO_CLOUD = '1'

$ollamaOriginalProfile = $env:USERPROFILE

try {
    $env:USERPROFILE = $ollamaHome

    Write-Host 'Starting Ollama at http://127.0.0.1:11434'
    Write-Host "Server log: $ollamaLogs\server.log"
    Write-Host 'Press Ctrl+C to stop.'

    & $ollamaExecutable serve 2>> "$ollamaLogs\server.log"
}
finally {
    $env:USERPROFILE = $ollamaOriginalProfile
}