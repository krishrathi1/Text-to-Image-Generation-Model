param(
    [string]$KaggleToken = $env:KAGGLE_API_TOKEN,
    [switch]$PersistToken
)

$ErrorActionPreference = 'Stop'

Write-Host '[1/4] Installing Kaggle CLI...'
python -m pip install --upgrade kaggle

if ([string]::IsNullOrWhiteSpace($KaggleToken)) {
    Write-Host '[2/4] No token passed. Skipping token export.'
    Write-Host 'Set token for current shell:'
    Write-Host '  $env:KAGGLE_API_TOKEN="<your_token>"'
} else {
    Write-Host '[2/4] Setting KAGGLE_API_TOKEN for this shell...'
    $env:KAGGLE_API_TOKEN = $KaggleToken
    if ($PersistToken) {
        [Environment]::SetEnvironmentVariable('KAGGLE_API_TOKEN', $KaggleToken, 'User')
        Write-Host 'Saved KAGGLE_API_TOKEN in user environment.'
    }
}

Write-Host '[3/4] Verifying Kaggle auth...'
try {
    kaggle competitions list -s titanic | Select-Object -First 5
    Write-Host '[4/4] Kaggle CLI is ready.'
} catch {
    Write-Host 'Kaggle verification failed. Check token/network and try again.'
    throw
}
