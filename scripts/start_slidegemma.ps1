param(
    [string]$ServerUrl   = "http://127.0.0.1:8080",
    [string]$ModelRef    = "ggml-org/gemma-4-E2B-it-GGUF",
    [int]$StartupTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvPython  = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$LlamaServer = Join-Path $ProjectRoot "llama_cuda\llama-server.exe"

function Test-ServerHealth {
    try {
        $health = "$ServerUrl/health" -replace '([^/])$', '$1/'
        $response = Invoke-WebRequest -Uri "${health}" -TimeoutSec 3 -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch { return $false }
}

if (-not (Test-ServerHealth)) {
    if (Test-Path $LlamaServer) {
        Write-Host "Starting llama-server with model $ModelRef ..."
        Start-Process -FilePath $LlamaServer -ArgumentList "-hf", $ModelRef, "--reasoning", "off" -WindowStyle Normal
        $elapsed = 0
        while ($elapsed -lt $StartupTimeoutSeconds) {
            Start-Sleep -Seconds 2
            $elapsed += 2
            if (Test-ServerHealth) { break }
        }
        if (-not (Test-ServerHealth)) {
            Write-Warning "llama-server did not become healthy within $StartupTimeoutSeconds seconds."
        }
    } else {
        Write-Host "No llama-server found at $LlamaServer. Start the AI service manually."
    }
}

if (Test-Path $VenvPython) {
    Start-Process -FilePath $VenvPython -ArgumentList (Join-Path $ProjectRoot "tools\gui.py") -WorkingDirectory $ProjectRoot
} else {
    Write-Warning ".venv not found. Run: python tools/gui.py"
}
