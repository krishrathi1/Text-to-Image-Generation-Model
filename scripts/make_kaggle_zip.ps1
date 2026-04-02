param(
    [string]$OutZip = (Join-Path (Resolve-Path ..) 'reckit_kaggle.zip')
)

$ErrorActionPreference = 'Stop'

if (Test-Path $OutZip) {
    Remove-Item -Force $OutZip
}

$excludePatterns = @(
    '\\.git\\',
    '\\.venv\\',
    '\\.venv-gpu\\',
    '\\.uv-cache\\',
    '\\__pycache__\\',
    '\\.pytest_cache\\',
    '\\.mypy_cache\\',
    '\\.idea\\',
    '\\.vscode\\',
    '\\.ipynb_checkpoints\\'
)

$files = Get-ChildItem -Path . -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $p = $_.FullName
    -not ($excludePatterns | Where-Object { $p -match $_ })
}

Compress-Archive -Path $files.FullName -DestinationPath $OutZip -CompressionLevel Optimal
Get-Item $OutZip | Select-Object FullName,Length,LastWriteTime
