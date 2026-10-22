<#
    Guest side of the air-gapped offline-install proof.

    Started by scripts/airgap_proof.py through a one-shot logon scheduled task
    on a QEMU guest booted with `-nic none`, so the machine has no network
    adapter at all. It installs The Oracle from the offline bundle, verifies the
    seeded model cache, renders through the managed launcher, writes
    <ProofDir>\airgap-result.json and then powers the guest off: with no network
    there is no other way for the host to learn that the run finished.

    Everything it touches lives on the data volume (resolved by label, since
    drive letters shift), never on the C: development install.
#>
param(
    [string]$VolumeLabel = "ORACLEDATA",
    [string]$Bundle,
    [string]$InstallRoot,
    [string]$ProofDir,
    [string]$HubCache
)

# The data volume's letter is not stable (the answer ISO takes D:), so resolve
# it by label: everything below lives on that data volume, never on C:.
$volume = Get-Volume -FileSystemLabel $VolumeLabel -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $volume -or -not $volume.DriveLetter) {
    Write-Host "FAIL: no volume labelled $VolumeLabel is mounted"
    shutdown.exe /s /t 5 /c "air-gap proof aborted: data volume missing"
    exit 1
}
$VolumeRoot = "$($volume.DriveLetter):"
if (-not $Bundle) { $Bundle = Join-Path $VolumeRoot "bundle" }
if (-not $InstallRoot) { $InstallRoot = Join-Path $VolumeRoot "OracleOffline" }
if (-not $ProofDir) { $ProofDir = Join-Path $VolumeRoot "airgap" }
if (-not $HubCache) { $HubCache = Join-Path $VolumeRoot "hfhub" }

# One shot: a completed (or attempted-then-finished) proof must never replay on
# a later networked boot and clobber its own evidence.
if (Test-Path (Join-Path $ProofDir "done.flag")) {
    Write-Host "air-gap proof already completed ($ProofDir\done.flag); nothing to do"
    shutdown.exe /s /t 5 /c "air-gap proof already completed"
    exit 0
}

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $ProofDir | Out-Null
$log = Join-Path $ProofDir "airgap.log"
Start-Transcript -Path $log -Append | Out-Null

$script:failures = New-Object System.Collections.Generic.List[string]
function Step([string]$message) {
    Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $message)
}
function Assert([bool]$ok, [string]$message) {
    if ($ok) { Write-Host "PASS: $message" } else { Write-Host "FAIL: $message"; $script:failures.Add($message) }
}

Step "air-gap proof: volume=$VolumeRoot bundle=$Bundle install=$InstallRoot cache=$HubCache"

# --- the air gap itself ------------------------------------------------------
$adapters = @(Get-NetAdapter -ErrorAction SilentlyContinue)
Step ("network adapters: " + $adapters.Count)
Assert ($adapters.Count -eq 0) "guest has no network adapter"
$reachable = Test-NetConnection -ComputerName 1.1.1.1 -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue
Assert (-not $reachable) "1.1.1.1:443 unreachable (no egress)"
$dns = $false
try { $dns = [bool](Resolve-DnsName pypi.org -ErrorAction Stop) } catch { $dns = $false }
Assert (-not $dns) "DNS for pypi.org does not resolve"

# --- install from the bundle -------------------------------------------------
$manifestPath = Join-Path $Bundle "manifest.json"
Assert (Test-Path $manifestPath) "bundle manifest present"
if (-not (Test-Path $manifestPath)) {
    Stop-Transcript | Out-Null
    shutdown.exe /s /t 5 /c "air-gap proof aborted: bundle manifest missing"
    exit 1
}
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
Step ("manifest: commit=" + $manifest.repo_commit + " platforms=" + ($manifest.platforms -join ",") + " wheels=" + ($manifest.wheel_counts | ConvertTo-Json -Compress))
$wheelDir = Join-Path $Bundle "wheels\windows"
Assert (Test-Path $wheelDir) "bundle carries a windows wheel set"
Get-ChildItem $wheelDir -Filter *.whl -ErrorAction SilentlyContinue | Measure-Object | ForEach-Object { Step ("windows wheels on disk: " + $_.Count) }

# The installer must not be able to reach an index even by accident, and the
# models must be seeded into a cache that starts empty, so a pass proves the
# bundle - not C:'s existing cache - supplied every weight.
$env:HF_HUB_CACHE = $HubCache
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:PIP_NO_INDEX = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
# The installer and its doctor run resolve `the-oracle` from a *fresh shell*
# PATH; a logon-launched process starts with a minimal PATH, so add the launcher
# dir the installer is about to create exactly as a logged-in desktop shell would.
$launcherDir = Join-Path $env:APPDATA "Python\Scripts"
if (($env:PATH -split ";") -notcontains $launcherDir) { $env:PATH = "$launcherDir;$env:PATH" }
Remove-Item -Recurse -Force $HubCache -ErrorAction SilentlyContinue

$installLog = Join-Path $ProofDir "install.log"
$installStart = Get-Date
Step "running install.bat (offline)"
& cmd.exe /c "`"$Bundle\install.bat`" `"$InstallRoot`"" 2>&1 | Tee-Object -FilePath $installLog
$installExit = $LASTEXITCODE
$installSeconds = [math]::Round(((Get-Date) - $installStart).TotalSeconds, 1)
Step "install.bat exit=$installExit elapsed=${installSeconds}s"
Assert ($installExit -eq 0) "install.bat exited 0"

Assert (Test-Path (Join-Path $InstallRoot ".oracle_offline")) "offline marker .oracle_offline written"
Assert (Test-Path (Join-Path $InstallRoot ".venv\Scripts\python.exe")) "venv created from bundled wheels"

# --- every pinned model came out of the bundle -------------------------------
foreach ($repo in $manifest.model_pins.PSObject.Properties) {
    $sha = [string]$repo.Value
    $dir = Join-Path $HubCache ("models--" + $repo.Name.Replace("/", "--"))
    Assert (Test-Path (Join-Path $dir "snapshots\$sha")) ("seeded snapshot " + $repo.Name + " @" + $sha.Substring(0, 12))
    $refsMain = Join-Path $dir "refs\main"
    # Byte-exact: huggingface_hub uses the raw file contents as the revision, so
    # trailing whitespace hides the snapshot from every hub-level loader.
    $pinned = if (Test-Path $refsMain) { (Get-Content $refsMain -Raw) } else { "" }
    Assert ($pinned -eq $sha) ("refs/main pins " + $repo.Name)
}

# --- render through the managed launcher ------------------------------------
$launcher = Join-Path $env:APPDATA "Python\Scripts\the-oracle.cmd"
Assert (Test-Path $launcher) "managed launcher the-oracle.cmd installed"
# No HF_HUB_OFFLINE here: the .oracle_offline marker is what must set it.
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue

$outdir = Join-Path $ProofDir "Output"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$renderLog = Join-Path $ProofDir "render.log"
$renderStart = Get-Date
Step "render: the-oracle render --input cli_short.txt"
& cmd.exe /c "`"$launcher`" render --input `"$InstallRoot\Input\cli_short.txt`" --outdir `"$outdir`"" 2>&1 | Tee-Object -FilePath $renderLog
$renderExit = $LASTEXITCODE
$renderSeconds = [math]::Round(((Get-Date) - $renderStart).TotalSeconds, 1)
Step "render exit=$renderExit elapsed=${renderSeconds}s"
Assert ($renderExit -eq 0) "render exited 0"

$flac = Get-ChildItem $outdir -Filter *.flac -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
Assert ($null -ne $flac) "render produced a FLAC file"
$audioFacts = ""
if ($flac) {
    $probe = @'
import json, sys
import soundfile as sf
data, rate = sf.read(sys.argv[1])
print(json.dumps({"seconds": round(len(data) / rate, 2), "rate": rate, "peak": round(float(abs(data).max()), 3)}))
'@
    $probePath = Join-Path $ProofDir "probe.py"
    Set-Content -Path $probePath -Value $probe -Encoding ascii
    try { $audioFacts = (& (Join-Path $InstallRoot ".venv\Scripts\python.exe") $probePath $flac.FullName 2>$null) | Select-Object -Last 1 } catch {}
    if (-not $audioFacts -or $audioFacts -notmatch "seconds") {
        # Fall back to the ffmpeg that provisioning installed, so a missing
        # soundfile in the venv cannot mask a good render.
        $ffprobe = (Get-Command ffprobe.exe -ErrorAction SilentlyContinue).Source
        if ($ffprobe -and (Test-Path $ffprobe)) {
            $duration = & $ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 $flac.FullName
            $audioFacts = (@{ seconds = $duration; source = "ffprobe" } | ConvertTo-Json -Compress)
        }
    }
    Step ("flac: " + $flac.FullName + " " + $flac.Length + " bytes " + $audioFacts)
    # A 4 s, 24 kHz mono clip compresses to ~80 KB; the duration assertion below
    # is the real content check, this is only a floor against an empty header.
    Assert ($flac.Length -gt 20000) "FLAC is non-trivial in size"
    $parsed = $null
    try { $parsed = $audioFacts | ConvertFrom-Json } catch {}
    Assert ($null -ne $parsed -and [double]$parsed.seconds -gt 1.0) "rendered audio is longer than 1s"
}

# --- still air-gapped at the end ---------------------------------------------
$adaptersEnd = @(Get-NetAdapter -ErrorAction SilentlyContinue)
Assert ($adaptersEnd.Count -eq 0) "guest still has no network adapter"

$result = [ordered]@{
    result = $(if ($script:failures.Count -eq 0) { "PASS" } else { "FAIL" })
    finished_at = (Get-Date).ToString("o")
    repo_commit = $manifest.repo_commit
    adapters = $adapters.Count
    install_seconds = $installSeconds
    render_seconds = $renderSeconds
    flac = $(if ($flac) { $flac.FullName } else { $null })
    audio = $audioFacts
    failures = $script:failures
}
$resultPath = Join-Path $ProofDir "airgap-result.json"
($result | ConvertTo-Json -Depth 5) | Set-Content -Path $resultPath -Encoding utf8
Step ("RESULT=" + $result.result + " -> " + $resultPath)

# One-shot: never replay this proof on the next (networked) boot. Then power
# off so the host knows the offline phase is over; it boots the guest with a
# network again only to read $resultPath.
Set-Content -Path (Join-Path $ProofDir "done.flag") -Value (Get-Date).ToString("o")
Stop-Transcript | Out-Null
shutdown.exe /s /t 5 /c "air-gap proof finished"
