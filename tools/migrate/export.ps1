# Packs this PC's Figuro (2D to 3D) work so another PC can carry on exactly where it stopped.
# Run on the OLD PC. It pauses what is running, stops the app, and writes ONE package file plus a
# manifest. It never deletes or changes your data. See tools/migrate/MIGRATE.md.
#
#   powershell -ExecutionPolicy Bypass -File tools\migrate\export.ps1 -AppDir "D:\2D to 3D" -OutDir "E:\"
param(
  [string]$AppDir = 'D:\2D to 3D',
  [string]$OutDir = '',
  [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
if (-not $OutDir) { $OutDir = Join-Path $AppDir 'migration' }
$api = "http://127.0.0.1:$Port"
$data = Join-Path $AppDir 'data'
if (-not (Test-Path -LiteralPath (Join-Path $data 'ledger'))) { throw "No data folder at $data (expected lab, study, ledger inside)." }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stage = Join-Path $env:TEMP 'figuro-export'
Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Up { try { Invoke-RestMethod "$api/health" -TimeoutSec 3 | Out-Null; $true } catch { $false } }

# 1. What is running now (so the new PC resumes the same things), then pause it
$was = [ordered]@{ autopilot_on = $false; study_running = $null; lab_running = @(); app_was_up = $false }
if (Up) {
  $was.app_was_up = $true
  $s = Invoke-RestMethod "$api/api/study"
  $was.autopilot_on = [bool]$s.autopilot.on
  $run = $s.studies | Where-Object { $_.status -eq 'running' } | Select-Object -First 1
  if ($run) { $was.study_running = $run.sid }
  $runs = (Invoke-RestMethod "$api/api/lab/runs").runs
  $was.lab_running = @($runs | Where-Object { $_.status -eq 'running' } | ForEach-Object { $_.run_id })

  "Pausing: autopilot off, study $($was.study_running), Lab runs $($was.lab_running -join ', ')"
  Invoke-RestMethod -Method Post "$api/api/study/autopilot?on=false" | Out-Null
  if ($was.study_running) { Invoke-RestMethod -Method Post "$api/api/study/$($was.study_running)/pause" | Out-Null }
  foreach ($r in $was.lab_running) { Invoke-RestMethod -Method Post "$api/api/lab/runs/$r/pause" | Out-Null }

  # let the build in progress finish: wait until the counts stop moving for 30 s (at most 10 min)
  "Waiting for the build in progress to finish..."
  $last = ''; $still = 0; $t0 = Get-Date
  while ($still -lt 30 -and ((Get-Date) - $t0).TotalMinutes -lt 10) {
    $now = ((Invoke-RestMethod "$api/api/lab/runs").runs | ForEach-Object { "$($_.run_id):$($_.done)" }) -join ','
    $now += '|' + (((Invoke-RestMethod "$api/api/study").studies | ForEach-Object { "$($_.sid):$($_.evals)" }) -join ',')
    if ($now -eq $last) { $still += 5 } else { $still = 0; $last = $now }
    Start-Sleep 5
  }
}

# 2. Stop the app so nothing writes while copying
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn*app:app*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 3
if (Up) { throw "The app is still running on port $Port. Close it and run this again." }

# 3. The tuned values: the app on Windows kept them in TEMP (no ANATOMY3D_TUNED set), Railway's copy is data\tuned.json
$tempTuned = Join-Path $env:TEMP 'anatomy3d-tuned.json'
$dataTuned = Join-Path $data 'tuned.json'
$tunedFrom = 'none'
if (Test-Path -LiteralPath $tempTuned) {
  if (Test-Path -LiteralPath $dataTuned) { Copy-Item -LiteralPath $dataTuned (Join-Path $stage 'tuned.railway.json') }
  Copy-Item -LiteralPath $tempTuned (Join-Path $stage 'tuned.json'); $tunedFrom = 'this PC (%TEMP%\anatomy3d-tuned.json)'
} elseif (Test-Path -LiteralPath $dataTuned) {
  $tunedFrom = 'data\tuned.json (from Railway)'
}

# 4. The AI models already downloaded (so the new PC works offline at once)
$models = Join-Path $env:USERPROFILE '.cache\anatomy3d-print'
if (Test-Path -LiteralPath $models) { Copy-Item -LiteralPath $models (Join-Path $stage 'models-cache') -Recurse }

# 5. Manifest: what is in the package, to check on the other side
function Count($p) { $f = Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction SilentlyContinue; [ordered]@{ files = @($f).Count; bytes = [int64](($f | Measure-Object Length -Sum).Sum) } }
$ledgerHashes = [ordered]@{}
Get-ChildItem -LiteralPath (Join-Path $data 'ledger') -File | ForEach-Object { $ledgerHashes[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
$manifest = [ordered]@{
  made = (Get-Date).ToString('s'); from_pc = $env:COMPUTERNAME; app_dir = $AppDir
  was = $was; tuned_from = $tunedFrom
  data = [ordered]@{ lab = (Count (Join-Path $data 'lab')); study = (Count (Join-Path $data 'study')); ledger = (Count (Join-Path $data 'ledger')) }
  ledger_sha256 = $ledgerHashes
  models = (Count (Join-Path $stage 'models-cache'))
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding UTF8

# 6. One package file: data + extras
$pkg = Join-Path $OutDir 'figuro-migration.tar'
"Packing to $pkg (this takes a few minutes)..."
tar -cf $pkg -C $AppDir data -C $env:TEMP figuro-export
if ($LASTEXITCODE -ne 0) { throw "Packing failed." }
Copy-Item -LiteralPath (Join-Path $stage 'manifest.json') (Join-Path $OutDir 'figuro-migration.manifest.json') -Force
$size = [math]::Round((Get-Item -LiteralPath $pkg).Length / 1GB, 2)
""
"DONE. Package: $pkg ($size GB) and its manifest beside it."
"Lab $($manifest.data.lab.files) files, study $($manifest.data.study.files), ledger $($manifest.data.ledger.files); tuned values from: $tunedFrom"
"Was running: autopilot $($was.autopilot_on), study $($was.study_running), Lab $($was.lab_running -join ', ')"
"This PC's copy is untouched. Do NOT start the app here again once the new PC has taken over (two PCs writing would split the history)."
