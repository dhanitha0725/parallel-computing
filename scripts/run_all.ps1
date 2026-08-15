# PowerShell Script to Build and Run OpenMP Benchmarks
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "       HPC PARALLEL CLASSIFICATION: C++ & OPENMP BENCHMARK SUITE        " -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$WorkspacePath = (Get-Location).Path
$DriveLetter = $WorkspacePath.Substring(0, 1).ToLower()
$SubPath = $WorkspacePath.Substring(2).Replace('\', '/')
$WslPath = "/mnt/$DriveLetter$SubPath"

Write-Host "[1/2] Compiling C++ OpenMP binaries..." -ForegroundColor Yellow
wsl bash -c "cd '$WslPath' && chmod +x scripts/*.sh && ./scripts/build.sh"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[2/2] Running Sequential and OpenMP Scaling Benchmarks (1, 2, 4, 8 threads)..." -ForegroundColor Yellow
wsl bash -c "cd '$WslPath' && ./scripts/run_all.sh"
