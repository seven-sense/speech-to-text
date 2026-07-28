# One-shot setup for Windows 10/11.
# Run from PowerShell. Requires administrator on first run for choco install.
# Idempotent - safe to re-run.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $repoRoot

Write-Host "==> Speech-To-Text setup for Windows" -ForegroundColor Cyan

# 1. Chocolatey check
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey not found. Install from https://chocolatey.org/install and re-run this script." -ForegroundColor Yellow
    exit 1
}

# 2. System packages
Write-Host "==> Installing system packages via choco" -ForegroundColor Cyan
choco install -y python311 ffmpeg "sox.portable" git

# Refresh PATH so freshly-installed binaries are visible to this shell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
            [System.Environment]::GetEnvironmentVariable("Path","User")

# 3. Python venv
if (-not (Test-Path "venv")) {
    Write-Host "==> Creating venv with python -m venv" -ForegroundColor Cyan
    python -m venv venv
}

# 4. Activate + install Python deps
Write-Host "==> Installing Python dependencies" -ForegroundColor Cyan
. .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 4a. Optional CUDA torch - only if nvidia-smi is on PATH
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "==> NVIDIA driver detected - installing CUDA-enabled torch first" -ForegroundColor Cyan
    try {
        pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    } catch {
        Write-Host "==> CUDA torch install failed, continuing with default wheel" -ForegroundColor Yellow
    }
}

pip install -r requirements.txt

# 5. .env scaffold (HF_TOKEN is optional - only needed for gated HF models)
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
    } else {
        Set-Content -Path ".env" -Value "HF_TOKEN="
    }
    Write-Host "==> Created .env (HF_TOKEN is optional, leave empty unless you use gated models)" -ForegroundColor Yellow
}

# 6. Verify
Write-Host "==> Running verify" -ForegroundColor Cyan
python main.py verify

Write-Host ""
Write-Host "Setup complete. Activate the venv with:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "Then run:"
Write-Host "  python main.py serve"
