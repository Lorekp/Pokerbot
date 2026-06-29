param(
  [string]$Run = "run_emd_k500_pluribus_20260625",
  [int]$Hands = 100,
  [int]$Seeds = 1,
  [int]$NumBuckets = 500,
  [string]$ShipName = "emd_k500_pluribus",
  [string]$Bucket = "video-processing-6789",
  [string]$Prefix = "pokerbot/prod6000/native_cfr",
  [string]$Region = "eu-central-1",
  [string]$CrateSrc = "runs/bp_crate_emd/battery_native",
  [string]$Python = "",
  [switch]$SkipBuild,
  [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

if (-not $Python) {
  if ($env:PYTHON -and (Get-Command $env:PYTHON -ErrorAction SilentlyContinue)) {
    $Python = $env:PYTHON
  } elseif (Test-Path "python") {
    $Python = "python"
  } else {
    $Python = "python"
  }
}

$RunPrefix = "$Prefix/$Run"
$WorkDir = Join-Path $Root ("runs\eval_speedup_{0}" -f $Run)
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

Write-Host "=== EMD eval speed benchmark ===" -ForegroundColor Cyan
Write-Host "run=$Run  hands=$Hands  seeds=$Seeds  buckets=$NumBuckets"
Write-Host "crate=$CrateSrc"
Write-Host "python=$Python"
Write-Host "workdir=$WorkDir"
Write-Host ""

if (-not $SkipBuild) {
  Write-Host "[1/5] Building the pipeline native evaluator..." -ForegroundColor Cyan
  $Manifest = Join-Path $Root (Join-Path $CrateSrc "Cargo.toml")
  if (-not (Test-Path $Manifest)) { throw "crate manifest not found: $Manifest" }
  $OldPyo3Python = $env:PYO3_PYTHON
  $env:PYO3_PYTHON = $Python
  & cargo build --release --features python --lib --manifest-path $Manifest
  $env:PYO3_PYTHON = $OldPyo3Python
  if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

  $ReleaseDir = Join-Path $Root (Join-Path $CrateSrc "target\release")
  $Dll = Join-Path $ReleaseDir "battery_native.dll"
  $Pyd = Join-Path $ReleaseDir "battery_native.pyd"
  $RootPyd = Join-Path $Root "battery_native.pyd"
  if (Test-Path $Dll) {
    Copy-Item -LiteralPath $Dll -Destination $Pyd -Force
    Copy-Item -LiteralPath $Dll -Destination $RootPyd -Force
    Write-Host "      refreshed $Pyd"
    Write-Host "      refreshed $RootPyd"
  } else {
    Write-Host "      no Windows .dll found; native_solver will use any existing .pyd/.so in target/release" -ForegroundColor Yellow
  }
} else {
  Write-Host "[1/5] Skipping native build (-SkipBuild)." -ForegroundColor Yellow
}

$ServeLocal = Join-Path $WorkDir "$ShipName.pkl"
if ($SkipDownload) {
  Write-Host "[2/5] Reusing local staged EMD serve artifact + frozen table (-SkipDownload)." -ForegroundColor Yellow
  if (-not (Test-Path $ServeLocal)) { throw "missing local serve artifact: $ServeLocal" }
  $FrozenLocal = Get-ChildItem -LiteralPath $WorkDir -Filter "*.emdfrzn" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
  if (-not $FrozenLocal) { throw "no local .emdfrzn table found in $WorkDir" }
  Write-Host "      serve:  $ServeLocal"
  Write-Host "      frozen: $FrozenLocal"
} else {
  Write-Host "[2/5] Downloading staged EMD serve artifact + latest frozen table..." -ForegroundColor Cyan
  $ServeS3 = "s3://$Bucket/$RunPrefix/serve/$ShipName.pkl"
  & aws s3 cp $ServeS3 $ServeLocal --region $Region --no-progress
  if ($LASTEXITCODE -ne 0) { throw "failed to download $ServeS3" }

  $ListJson = (& aws s3api list-objects-v2 --bucket $Bucket --prefix "$RunPrefix/frozen/" --region $Region --output json | Out-String)
  if ($LASTEXITCODE -ne 0) { throw "failed to list s3://$Bucket/$RunPrefix/frozen/" }
  $List = $ListJson | ConvertFrom-Json
  $Contents = @()
  if ($List.Contents) { $Contents = @($List.Contents) }
  $FrozenObj = $Contents |
    Where-Object { $_.Key -like "*.emdfrzn" } |
    Sort-Object LastModified -Descending |
    Select-Object -First 1
  if (-not $FrozenObj) { throw "no .emdfrzn table found under s3://$Bucket/$RunPrefix/frozen/" }

  $FrozenName = ($FrozenObj.Key -split "/")[-1]
  $FrozenS3 = "s3://$Bucket/$($FrozenObj.Key)"
  $FrozenLocal = Join-Path $WorkDir $FrozenName
  & aws s3 cp $FrozenS3 $FrozenLocal --region $Region --no-progress
  if ($LASTEXITCODE -ne 0) { throw "failed to download $FrozenS3" }
  Write-Host "      serve:  $ServeS3"
  Write-Host "      frozen: $FrozenS3"
}
Write-Host ""

Write-Host "[3/5] Checking Python imports/native symbol..." -ForegroundColor Cyan
$OldPyPath = $env:PYTHONPATH
if ($OldPyPath) { $env:PYTHONPATH = "$Root;$OldPyPath" } else { $env:PYTHONPATH = $Root }
$ProbeCode = @"
import sys
import jax, numpy, phevaluator
from pokerbot.eval import lbr_native as L
print(sys.executable)
print("native_available=", L.available())
import battery_native
print("battery_native=", getattr(battery_native, "__file__", "<builtin>"))
h = battery_native.Lbr6Harness({}, [True]*11, [True]*11, $NumBuckets, [])
print("has_set_emd_frozen=", hasattr(h, "set_emd_frozen"))
if not L.available():
    raise SystemExit("native_available=False")
if not hasattr(h, "set_emd_frozen"):
    raise SystemExit("battery_native.Lbr6Harness.set_emd_frozen missing")
"@
$ProbePath = Join-Path $WorkDir "_bench_probe.py"
Set-Content -LiteralPath $ProbePath -Value $ProbeCode -Encoding UTF8
& $Python $ProbePath
if ($LASTEXITCODE -ne 0) {
  $env:PYTHONPATH = $OldPyPath
  throw "Python preflight failed. Install/check jax, numpy, phevaluator, and the built battery_native extension."
}
Write-Host ""

function Invoke-EvalBench {
  param(
    [string]$Label,
    [bool]$UseNative
  )

  $OutJson = Join-Path $WorkDir "$Label.json"
  $LogPath = Join-Path $WorkDir "$Label.log"
  $NativeArg = if ($UseNative) { "--native" } else { "--no-native" }
  $Args = @(
    "scripts/eval/eval_emd_vs_iter62000.py",
    "emd",
    "--checkpoint", $ServeLocal,
    "--emd-frozen", $FrozenLocal,
    "--out", $OutJson,
    "--seeds", "$Seeds",
    "--hands", "$Hands",
    "--num-buckets", "$NumBuckets",
    "--value-only",
    $NativeArg
  )

  Write-Host ("[4/5] Running {0} ({1})..." -f $Label, $NativeArg) -ForegroundColor Cyan
  $Sw = [Diagnostics.Stopwatch]::StartNew()
  $OldEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $Output = & $Python @Args 2>&1
    $Exit = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $OldEap
    $Sw.Stop()
  }
  $Output | Tee-Object -FilePath $LogPath
  if ($Exit -ne 0) { throw "$Label eval failed; see $LogPath" }

  $Json = Get-Content -LiteralPath $OutJson -Raw | ConvertFrom-Json
  return [pscustomobject]@{
    Label = $Label
    Method = [string]$Json.result.method
    CommandWallSec = [math]::Round($Sw.Elapsed.TotalSeconds, 3)
    EvalWallSec = [math]::Round([double]$Json.result.wall_s, 3)
    AnchorBb100 = [math]::Round([double]$Json.anchor.always_fold_bb_per_100, 3)
    PooledBb100 = [math]::Round([double]$Json.result.pooled_bb_per_100, 3)
    Json = $OutJson
    Log = $LogPath
  }
}

$Reference = Invoke-EvalBench -Label "python_reference" -UseNative:$false
$Native = Invoke-EvalBench -Label "rust_native" -UseNative:$true

function Format-Speedup([double]$Slow, [double]$Fast) {
  if ($Fast -le 0.0) { return "n/a" }
  return ("{0:n2}x" -f ($Slow / $Fast))
}

Write-Host ""
Write-Host "[5/5] Result" -ForegroundColor Cyan
$Reference | Format-List
$Native | Format-List
Write-Host ("blueprint-loop speedup (result.wall_s): {0}" -f (Format-Speedup $Reference.EvalWallSec $Native.EvalWallSec)) -ForegroundColor Green
Write-Host ("end-to-end speedup (command wall):       {0}" -f (Format-Speedup $Reference.CommandWallSec $Native.CommandWallSec)) -ForegroundColor Green
Write-Host ""
Write-Host "Artifacts:"
Write-Host "  $WorkDir"
Write-Host ""
Write-Host "Note: $Hands hands x $Seeds seed(s) is a throughput smoke test, not a verdict-quality exploitability estimate." -ForegroundColor Yellow

Write-Host "Keeping downloaded artifacts/logs in $WorkDir for inspection."

$env:PYTHONPATH = $OldPyPath
