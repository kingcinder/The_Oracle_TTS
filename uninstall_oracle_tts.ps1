$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-SupportedPython {
    $candidates = @(
        @("py", "-3.12"),
        @("py", "-3.11"),
        @("python")
    )

    foreach ($candidate in $candidates) {
        $command = $candidate[0]
        $arguments = @($candidate | Select-Object -Skip 1)
        try {
            & $command @arguments -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:3] < (3, 13) else 1)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return @($command) + $arguments
            }
        } catch {
        }
    }

    throw "Need Python 3.11 or 3.12 with venv support."
}

$python = Get-SupportedPython
& $python[0] @($python | Select-Object -Skip 1) "$repoRoot\scripts\manage_install.py" uninstall @args
exit $LASTEXITCODE
