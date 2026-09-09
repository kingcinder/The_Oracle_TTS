# The Oracle - one-stop manager for Windows.
#
#   .\oracle.ps1 install     set up the venv, register launchers, verify (doctor)
#   .\oracle.ps1 start       launch the desktop GUI
#   .\oracle.ps1 update      refresh an existing install (keeps your data)
#   .\oracle.ps1 uninstall   remove launchers and the local venv
#
# Each command delegates to the same managed installer the per-action scripts
# use, so behavior matches install_oracle_tts.ps1 / run_oracle_tts.ps1 exactly.

param(
    [Parameter(Position = 0)]
    [string]$Action = "help"
)

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Show-Help {
    Write-Host @"
The Oracle - manager

  .\oracle.ps1 install      Set up the venv, register launchers, run the doctor.
  .\oracle.ps1 start        Launch the desktop GUI.
  .\oracle.ps1 update       Refresh an existing install (your data is kept).
  .\oracle.ps1 uninstall    Remove launchers and the local venv.
  .\oracle.ps1 doctor       Diagnostics only.
  .\oracle.ps1 bootstrap    Venv + dependencies without Start Menu integration.
  .\oracle.ps1 help         Show this message.
"@
}

function Select-Python {
    foreach ($candidate in @("py -3.12", "py -3.11", "python")) {
        try {
            $null = Invoke-Expression "$candidate -c `"import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[: 3] < (3, 13) else 1)`"" 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
        catch { continue }
    }
    return $null
}

switch ($Action.ToLower()) {
    "install" { $managed = "install" }
    "start" { $managed = "run" }
    "run" { $managed = "run" }
    "update" { $managed = "update" }
    "uninstall" { $managed = "uninstall" }
    "doctor" { $managed = "doctor" }
    "bootstrap" { $managed = "bootstrap" }
    "help" { Show-Help; exit 0 }
    "-h" { Show-Help; exit 0 }
    "--help" { Show-Help; exit 0 }
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Show-Help
        exit 1
    }
}

$python = Select-Python
if (-not $python) {
    Write-Host "FAIL: Need Python 3.11 or 3.12 (try: py -3.12)." -ForegroundColor Red
    exit 1
}

Invoke-Expression "$python `"$RepoRoot\scripts\manage_install.py`" $managed"
exit $LASTEXITCODE
