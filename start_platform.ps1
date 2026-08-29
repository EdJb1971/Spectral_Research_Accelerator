# SpectralEarth Platform Local Startup Utility
# Starts the scientific backend and the Vite frontend together.
#
# Fixes defect D7: the previous version launched uvicorn inside Start-Job, which runs in a
# fresh runspace whose working directory is the user's home folder. `src.api.main` was
# therefore not importable, the backend died immediately, and the frontend came up looking
# fine against a dead API. This version pins the working directory, waits on the
# /api/v1/health endpoint, and fails loudly if the backend never becomes ready.

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   SpectralEarth Scientific Platform Local Setup & Runner  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------- Interpreter selection
# Prefer the project virtual environment; fall back to whatever python is on PATH.
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
    Write-Host "[*] Using project virtual environment: .venv" -ForegroundColor Green
} else {
    $Python = "python"
    Write-Host "[!] No .venv found - using system python." -ForegroundColor Yellow
    Write-Host "    Recommended:  python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "[*] Checking Python environment..." -ForegroundColor White
$pythonVersion = & $Python --version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found. Install Python 3.10+ and retry." }
Write-Host "    Found: $pythonVersion" -ForegroundColor Green

Write-Host "[*] Checking Node.js environment..." -ForegroundColor White
$nodeVersion = & node --version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "Node.js not found. Install Node.js LTS and retry." }
Write-Host "    Found Node.js: $nodeVersion" -ForegroundColor Green

# ---------------------------------------------------------------- Optional dependency install
$installDeps = Read-Host "[?] Check and install/update dependencies? (y/N)"
if ($installDeps -eq "y" -or $installDeps -eq "Y") {
    Write-Host "[*] Installing Python backend dependencies..." -ForegroundColor Cyan
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed." }

    Write-Host "[*] Installing Node.js frontend dependencies..." -ForegroundColor Cyan
    Push-Location (Join-Path $ProjectRoot "frontend")
    & npm install
    $npmExit = $LASTEXITCODE
    Pop-Location
    if ($npmExit -ne 0) { Write-Error "npm install failed." }
    Write-Host "[+] Dependencies updated." -ForegroundColor Green
}

# ---------------------------------------------------------------- Local environment
# .env.local holds machine-local secrets and opt-in flags (GEMINI_API_KEY,
# SPECTRALEARTH_ALLOW_NETWORK). Nothing in the Python source loads it: the backend reads
# os.getenv directly (zarr_source.network_enabled) and python-dotenv is not a dependency.
# Vite does not load it either - it reads env files from frontend/ and only exposes
# VITE_-prefixed names to the browser. The launcher is therefore what makes the file real,
# by promoting it into this process's environment, which the uvicorn job and `npm run dev`
# both inherit as child processes.
#
# A variable already set in the calling shell wins over the file, so a one-off override
# still works without editing anything.
Write-Host "[*] Loading local environment from .env.local ..." -ForegroundColor White
$EnvFile = Join-Path $ProjectRoot ".env.local"
if (Test-Path $EnvFile) {
    $loadedNames = @()
    $skippedNames = @()
    foreach ($line in Get-Content $EnvFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        $split = $trimmed.IndexOf("=")
        if ($split -lt 1) {
            Write-Host "    Ignoring unparseable line: $trimmed" -ForegroundColor DarkGray
            continue
        }
        $name = $trimmed.Substring(0, $split).Trim()
        $value = $trimmed.Substring($split + 1).Trim()
        # Strip one matched pair of surrounding quotes so KEY="v" and KEY=v agree.
        if ($value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ([Environment]::GetEnvironmentVariable($name, "Process")) {
            $skippedNames += $name
            continue
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
        $loadedNames += $name
    }
    # Names only. These values are secrets and must never reach the console or a log.
    if ($loadedNames.Count -gt 0) {
        Write-Host "    Loaded: $($loadedNames -join ', ')" -ForegroundColor Green
    }
    if ($skippedNames.Count -gt 0) {
        Write-Host "    Already set in this shell, file ignored for: $($skippedNames -join ', ')" -ForegroundColor DarkGray
    }
} else {
    Write-Host "    No .env.local found - network access stays disabled and no API keys are set." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- Backend
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "                  Starting Services                       " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "[*] Launching FastAPI backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan

# D7: $using:ProjectRoot + Set-Location gives the job the correct working directory, so
# `src.api.main` resolves. Without this the import fails and the job exits silently.
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:ProjectRoot
    & $using:Python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
}

# Wait for readiness rather than racing the frontend against an unstarted API.
Write-Host "[*] Waiting for backend health check..." -ForegroundColor White
$ready = $false
for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 1
    if ($backendJob.State -eq "Failed" -or $backendJob.State -eq "Completed") { break }
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2
        if ($resp.status -eq "ok") {
            $ready = $true
            Write-Host "    Backend ready (db=$($resp.database), device=$($resp.torch_device), datasets=$($resp.datasets_available))" -ForegroundColor Green
            break
        }
    } catch {
        # not up yet; keep waiting
    }
}

if (-not $ready) {
    Write-Host "[X] Backend failed to become ready. Captured output:" -ForegroundColor Red
    Receive-Job $backendJob
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    Write-Error "Backend did not start. Not launching the frontend against a dead API."
}

# ---------------------------------------------------------------- Frontend
Write-Host "[*] Launching Vite frontend on http://localhost:3000 ..." -ForegroundColor Cyan
Write-Host "[!] Press Ctrl+C to stop both servers." -ForegroundColor Yellow

try {
    Push-Location (Join-Path $ProjectRoot "frontend")
    & npm run dev
} finally {
    Pop-Location
    Write-Host "[*] Stopping backend..." -ForegroundColor Yellow
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    $backendOutput = Receive-Job $backendJob -ErrorAction SilentlyContinue
    if ($backendOutput) {
        Write-Host "[*] Backend output tail:" -ForegroundColor DarkGray
        $backendOutput | Select-Object -Last 10 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    Write-Host "[+] Servers stopped." -ForegroundColor Green
}
