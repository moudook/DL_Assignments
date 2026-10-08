# launch.ps1 — start the Assignment-4 pipeline detached, and keep the machine awake.
#
# Purpose
# -------
# A full run is hours of GPU work. Starting it from an interactive terminal means
# it dies when that terminal closes, and on a laptop it dies when Windows decides
# to sleep. This script does three things:
#
#   1. Starts the pipeline as a DETACHED process (Start-Process), so it survives
#      closing this terminal.
#   2. Prevents sleep and hibernation for the duration, which is what "never put
#      this PC to sleep" actually requires at the OS level.
#   3. Prints how to monitor progress and how to restore power settings after.
#
# What this does NOT protect against
# ----------------------------------
# Detached is not crash-proof. A Windows Update reboot, a power loss, or a driver
# reset will still kill the run. That is what the checkpoint/resume machinery in
# run_tracker.py is for: re-running launch.ps1 picks up from the last checkpoint
# instead of starting over. Nothing here replaces checkpointing.
#
# Usage
# -----
#   powershell -ExecutionPolicy Bypass -File launch.ps1
#   powershell -ExecutionPolicy Bypass -File launch.ps1 -MaxEpochs 10000
#   powershell -ExecutionPolicy Bypass -File launch.ps1 -Only 1 2
#   powershell -ExecutionPolicy Bypass -File launch.ps1 -RestorePower
#
# Monitor (from another terminal):
#   python monitor.py --outdir results --watch --gpu

param(
    [int]$MaxEpochs = 10000,
    [int[]]$Only = @(),
    [string]$OutDir = "results",
    [switch]$Quick,
    [switch]$RestorePower
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── Restore power settings and exit ─────────────────────────────────────────
if ($RestorePower) {
    Write-Host "Restoring default power settings..."
    powercfg /change standby-timeout-ac 30
    powercfg /change hibernate-timeout-ac 60
    powercfg /change monitor-timeout-ac 30
    Write-Host "Done: AC standby 30 min, hibernate 60 min, monitor 30 min."
    exit 0
}

# ── Disable sleep while training ────────────────────────────────────────────
# Set on AC power, which is the relevant case for a laptop on mains. These are
# process-wide changes to Windows power settings, so the matching restore is
# important: run with -RestorePower when training is done, otherwise the machine
# will never sleep again.
Write-Host "Disabling sleep/hibernate on AC power..."
powercfg /change standby-timeout-ac 0        # 0 = never sleep
powercfg /change hibernate-timeout-ac 0       # 0 = never hibernate
powercfg /change monitor-timeout-ac 0         # keep display on too
Write-Host "  AC sleep: never, hibernate: never, monitor: never"
Write-Host "  RESTORE LATER: powershell -ExecutionPolicy Bypass -File launch.ps1 -RestorePower"
Write-Host ""

# ── Build the argument list ─────────────────────────────────────────────────
$argList = @("run_all.py", "--outdir", $OutDir, "--max-epochs", "$MaxEpochs")
if ($Quick)    { $argList += "--quick" }
if ($Only.Count -gt 0) {
    $argList += "--only"
    $argList += ($Only | ForEach-Object { "$_" })
}

$logDir = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "pipeline.out.log"
$stderr = Join-Path $logDir "pipeline.err.log"

Write-Host "Starting pipeline detached:"
Write-Host "  python $($argList -join ' ')"
Write-Host ""
Write-Host "  stdout -> $stdout"
Write-Host "  stderr -> $stderr"
Write-Host ""

# Start-Process with -WindowStyle Hidden detaches from this shell: the pipeline
# keeps running after the terminal is closed. -RedirectStandardOutput/Error
# capture the output, which is what monitor.py and post-mortem debugging read.
# Note this is a .ps1 wrapper rather than a bare `python run_all.py &` because
# PowerShell has no job-control background operator; Start-Process is the native
# equivalent.
$proc = Start-Process -FilePath "python" -ArgumentList $argList `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 3

if ($proc.HasExited) {
    Write-Host "FAILED: process exited immediately (exit code $($proc.ExitCode))."
    Write-Host "Check $stderr"
    Get-Content $stderr -Tail 30 | Write-Host
    exit 1
}

Write-Host "Running. PID = $($proc.Id)"
Write-Host ""
Write-Host "Monitor with:"
Write-Host "  python monitor.py --outdir $OutDir --watch --gpu"
Write-Host ""
Write-Host "Stop with:  Stop-Process -Id $($proc.Id)"
Write-Host "Check the log:  Get-Content $stdout -Tail 40 -Wait"