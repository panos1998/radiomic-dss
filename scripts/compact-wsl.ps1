# Requires: Run as Administrator
# Purpose: Cap WSL memory (optional) and compact Docker Desktop & WSL distro VHDX files to reclaim disk space.

[CmdletBinding()] param(
  [string]$Memory = "4GB",
  [int]$Processors = 4,
  [switch]$SkipWslconfig
)

$ErrorActionPreference = "Stop"

function Require-Admin {
  $isAdmin = ([Security.Principal.WindowsPrincipal]
    [Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $isAdmin) {
    Write-Error "Please run this script in an elevated PowerShell (Run as Administrator)."; exit 1
  }
}

function Ensure-OptimizeVHD {
  $cmd = Get-Command Optimize-VHD -ErrorAction SilentlyContinue
  if (-not $cmd) {
    Write-Warning "Optimize-VHD not available. Enable Hyper-V and reboot:"
    Write-Host "  Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All"
    exit 1
  }
}

function Write-WslConfigIfNeeded {
  if ($SkipWslconfig) { return }
  $wslPath = Join-Path $env:USERPROFILE ".wslconfig"
  $content = @"[wsl2]
memory=$Memory
processors=$Processors
swap=0
localhostForwarding=true
pageReporting=false
"@
  Set-Content -Path $wslPath -Value $content -Encoding UTF8
  Write-Host "Updated $wslPath with memory=$Memory, processors=$Processors, swap=0"
}

function Shutdown-WSLAndDocker {
  Write-Host "Stopping Docker Desktop (if running)..."
  Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Write-Host "Shutting down WSL..."
  wsl --shutdown
}

function Compact-VHDXs {
  Ensure-OptimizeVHD
  # Docker Desktop VHDX
  $dockerVhdx = Join-Path $env:LOCALAPPDATA "Docker\wsl\data\ext4.vhdx"
  if (Test-Path $dockerVhdx) {
    Write-Host "Compacting Docker Desktop VHDX: $dockerVhdx"
    Optimize-VHD -Path $dockerVhdx -Mode Full
  } else { Write-Host "Docker Desktop VHDX not found at $dockerVhdx" }

  # Common WSL distro vendors
  $vendors = 'Canonical','Ubuntu','Debian','SUSE','kali'
  $packagesRoot = Join-Path $env:LOCALAPPDATA 'Packages'
  if (Test-Path $packagesRoot) {
    Get-ChildItem $packagesRoot -Directory | Where-Object { $vendors | ForEach-Object { $_ -as [string] } -contains ($_.Name.Split('_')[0]) } | ForEach-Object {
      $vhdx = Join-Path $_.FullName 'LocalState\ext4.vhdx'
      if (Test-Path $vhdx) {
        Write-Host "Compacting WSL distro VHDX: $vhdx"
        Optimize-VHD -Path $vhdx -Mode Full
      }
    }
  }
}

function Show-Sizes {
  Write-Host "Current VHDX sizes:"
  $dockerVhdx = Join-Path $env:LOCALAPPDATA "Docker\wsl\data\ext4.vhdx"
  if (Test-Path $dockerVhdx) { Get-Item $dockerVhdx | Select-Object FullName,Length | Format-Table }
  $packagesRoot = Join-Path $env:LOCALAPPDATA 'Packages'
  if (Test-Path $packagesRoot) {
    Get-ChildItem $packagesRoot -Directory | ForEach-Object {
      $vhdx = Join-Path $_.FullName 'LocalState\ext4.vhdx'
      if (Test-Path $vhdx) { Get-Item $vhdx | Select-Object FullName,Length }
    } | Format-Table
  }
}

function Restart-Docker {
  Write-Host "Restarting Docker Desktop..."
  Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -ErrorAction SilentlyContinue
}

# --- Main ---
Require-Admin
Write-WslConfigIfNeeded
Shutdown-WSLAndDocker
Compact-VHDXs
Show-Sizes
Restart-Docker

Write-Host "Done. You can also run 'docker system df' to verify space reclaimed."