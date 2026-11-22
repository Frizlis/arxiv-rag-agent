$ragProjectRoot = Split-Path -Parent $PSScriptRoot
$ragStorage = Join-Path $ragProjectRoot 'llm\local'

if ([System.IO.Path]::GetPathRoot($ragStorage) -ne 'D:\') {
    throw 'Локальное хранилище проекта должно находиться на D:.'
}

$ragPaths = @{
    OLLAMA_MODELS   = "$ragStorage\ollama\models"
    HF_HOME         = "$ragStorage\huggingface"
    HF_HUB_CACHE    = "$ragStorage\huggingface\hub"
    HF_XET_CACHE    = "$ragStorage\huggingface\xet"
    UV_CACHE_DIR    = "$ragStorage\cache\uv"
    CUDA_CACHE_PATH = "$ragStorage\cache\cuda"
    TEMP            = "$ragStorage\tmp"
    TMP             = "$ragStorage\tmp"
}

foreach ($entry in $ragPaths.GetEnumerator()) {
    New-Item -ItemType Directory -Path $entry.Value -Force |
        Out-Null

    [Environment]::SetEnvironmentVariable(
        $entry.Key,
        $entry.Value,
        'Process'
    )
}

$ragNoProxyHosts = @(
    ($env:NO_PROXY -split ',')
    '127.0.0.1'
    'localhost'
)

$env:NO_PROXY = (
    $ragNoProxyHosts |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ } |
        Select-Object -Unique
) -join ','

$env:HF_HUB_OFFLINE = '1'      # При необходимости скачать новую модель эту переменную потребуется отключить в терминале загрузки.

Write-Host "Локальное хранилище: $ragStorage"