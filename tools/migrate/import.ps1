# Sets Figuro (2D to 3D) up on THIS PC from a package made by export.ps1 on the old PC, checks every
# file count against the manifest, and resumes what was running. See tools/migrate/MIGRATE.md.
#
#   powershell -ExecutionPolicy Bypass -File import.ps1 -Package "E:\figuro-migration.tar" -AppDir "D:\2D to 3D"
param(
  [Parameter(Mandatory = $true)][string]$Package,
  [string]$AppDir = 'D:\2D to 3D',
  [string]$Branch = 'learning-fix',
  [int]$Port = 8000,
  [switch]$NoResume
)
$ErrorActionPreference = 'Stop'
$api = "http://127.0.0.1:$Port"
if (-not (Test-Path -LiteralPath $Package)) { throw "Package not found: $Package" }
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
if (Test-Path -LiteralPath (Join-Path $AppDir 'data\ledger')) { throw "$AppDir already has a data folder. Move it away first; this will not overwrite work." }

# 1. Python 3.12
$hasPy = $false
try { & py -3.12 --version | Out-Null; $hasPy = ($LASTEXITCODE -eq 0) } catch { }
if (-not $hasPy) {
  "Installing Python 3.12..."
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

# 2. The app (branch with the learning fix), unless already here
if (-not (Test-Path -LiteralPath (Join-Path $AppDir 'webapp'))) {
  "Downloading the app ($Branch)..."
  $zip = Join-Path $env:TEMP 'figuro-app.zip'; $x = Join-Path $env:TEMP 'figuro-app'
  Invoke-WebRequest "https://github.com/brownhippie/anatomy3d-print/archive/refs/heads/$Branch.zip" -OutFile $zip -UseBasicParsing
  Remove-Item -LiteralPath $x -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive -LiteralPath $zip -DestinationPath $x -Force
  Copy-Item (Join-Path $x "anatomy3d-print-$Branch\*") -Destination $AppDir -Recurse -Force
}
$vpy = Join-Path $AppDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $vpy)) { & py -3.12 -m venv (Join-Path $AppDir '.venv') }
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -r (Join-Path $AppDir 'requirements.txt')
& $vpy -m pip install -e $AppDir

# 3. Unpack and check against the manifest
"Unpacking $Package ..."
tar -xf $Package -C $AppDir
if ($LASTEXITCODE -ne 0) { throw "Unpacking failed." }
$extra = Join-Path $AppDir 'figuro-export'
$m = Get-Content -LiteralPath (Join-Path $extra 'manifest.json') -Raw | ConvertFrom-Json
$data = Join-Path $AppDir 'data'
function Count($p) { $f = Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction SilentlyContinue; @{ files = @($f).Count; bytes = [int64](($f | Measure-Object Length -Sum).Sum) } }
$bad = @()
foreach ($k in 'lab', 'study', 'ledger') {
  $c = Count (Join-Path $data $k)
  if ($c.files -ne $m.data.$k.files -or $c.bytes -ne $m.data.$k.bytes) { $bad += "$k`: $($c.files) files/$($c.bytes) B, expected $($m.data.$k.files)/$($m.data.$k.bytes)" }
}
foreach ($p in $m.ledger_sha256.PSObject.Properties) {
  $h = (Get-FileHash -LiteralPath (Join-Path $data "ledger\$($p.Name)") -Algorithm SHA256).Hash
  if ($h -ne $p.Value) { $bad += "ledger\$($p.Name) differs" }
}
if ($bad) {
  Remove-Item -LiteralPath $data, $extra -Recurse -Force -ErrorAction SilentlyContinue  # so a fresh copy can be imported
  throw ("The package does not match its manifest - copy it again, then rerun:`n" + ($bad -join "`n"))
}
"Checked: every file count, size and ledger fingerprint matches the old PC."

# 4. Tuned values into data\ (where the start file now points), models into the user cache
if (Test-Path -LiteralPath (Join-Path $extra 'tuned.json')) {
  if (Test-Path -LiteralPath (Join-Path $data 'tuned.json')) { Move-Item -LiteralPath (Join-Path $data 'tuned.json') (Join-Path $data 'tuned.railway.json') -Force }
  Copy-Item -LiteralPath (Join-Path $extra 'tuned.json') (Join-Path $data 'tuned.json')
}
if (Test-Path -LiteralPath (Join-Path $extra 'models-cache')) {
  $models = Join-Path $env:USERPROFILE '.cache\anatomy3d-print'
  New-Item -ItemType Directory -Force -Path $models | Out-Null
  Copy-Item (Join-Path $extra 'models-cache\*') -Destination $models -Recurse -Force
}

Copy-Item -LiteralPath (Join-Path $extra 'manifest.json') (Join-Path $data 'migrated-from.json') -Force
Remove-Item -LiteralPath $extra -Recurse -Force

# 5. Start file (keeps the tuned values in data\ from now on)
@'
@echo off
cd /d "%~dp0"
set "ANATOMY3D_LAB_DIR=%~dp0data\lab"
set "ANATOMY3D_STUDY_DIR=%~dp0data\study"
set "ANATOMY3D_LEDGER_DIR=%~dp0data\ledger"
set "ANATOMY3D_TUNED=%~dp0data\tuned.json"
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn app:app --app-dir webapp --port 8000
'@ | Set-Content -LiteralPath (Join-Path $AppDir 'Start 2D to 3D.bat') -Encoding ASCII

# 6. Start, then resume what was running on the old PC
Start-Process -FilePath (Join-Path $AppDir 'Start 2D to 3D.bat') -WorkingDirectory $AppDir
"Starting the app..."
$up = $false
for ($i = 0; $i -lt 60 -and -not $up; $i++) { Start-Sleep 3; try { Invoke-RestMethod "$api/health" -TimeoutSec 3 | Out-Null; $up = $true } catch { } }
if (-not $up) { throw "The app did not start - look at its black window for the error." }
$was = $m.was
if (-not $NoResume) {
  foreach ($r in @($was.lab_running)) { if ($r) { Invoke-RestMethod -Method Post "$api/api/lab/runs/$r/resume" | Out-Null; "Resumed Lab run $r" } }
  if ($was.study_running) { Invoke-RestMethod -Method Post "$api/api/study/$($was.study_running)/resume" | Out-Null; "Resumed study $($was.study_running)" }
  if ($was.autopilot_on) { Invoke-RestMethod -Method Post "$api/api/study/autopilot?on=true" | Out-Null; "Autopilot back on" }
}

# 7. Report
$s = Invoke-RestMethod "$api/api/study"
$runs = (Invoke-RestMethod "$api/api/lab/runs").runs
""
"DONE on $env:COMPUTERNAME. Tuned values from: $($m.tuned_from)"
"Autopilot: $($s.autopilot_now.state) - $($s.autopilot_now.text)"
$runs | Select-Object -First 5 | ForEach-Object { "Lab run $($_.run_id) ($($_.photo_name)): $($_.status), $($_.done)/$($_.total) done" }
"Use 'Start 2D to 3D.bat' in $AppDir from now on. Keep the old PC's app closed."
