param(
  [int]$Iterations = 250,
  [int]$Deals = 64,
  [int]$Frequency = 19,
  [string]$Frozen = "data\abstraction\emd_frozen_flop_turn_160_r400.emdfrzn",
  [string]$Out = "artifacts\emd_k160_cargo_flamegraph.svg"
)

$ErrorActionPreference = "Stop"

if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
  throw "cargo not found on PATH"
}
if (!(Test-Path $Frozen)) {
  throw "frozen EMD lookup not found: $Frozen"
}

$stackFile = Join-Path (Get-Location) "cargo-flamegraph.stacks"
if (Test-Path -LiteralPath $stackFile) {
  Remove-Item -LiteralPath $stackFile -Force
}

# Windows sampling uses the blondie/ETW backend and may require an elevated
# PowerShell. Keep debug info light and the sample rate conservative; otherwise
# Windows stack post-processing can emit enormous cargo-flamegraph.stacks files.
$env:CARGO_PROFILE_RELEASE_DEBUG = "line-tables-only"

try {
  cargo flamegraph `
    --manifest-path tools\battery_native\Cargo.toml `
    --example bench_emd `
    -F $Frequency `
    --min-width 0.1 `
    --deterministic `
    -o $Out `
    --title "K160 EMD trainer cargo flamegraph" `
    --subtitle "bench_emd $Iterations iters x $Deals deals @ ${Frequency}Hz" `
    -- `
    $Iterations `
    $Deals `
    $Frozen

  if ($LASTEXITCODE -ne 0) {
    throw "cargo flamegraph failed with exit code $LASTEXITCODE"
  }
  Write-Host "wrote $Out"
} finally {
  # cargo flamegraph normally removes this. If post-processing fails, clean up
  # the huge intermediate so Git/status tooling does not try to hash it.
  if (Test-Path -LiteralPath $stackFile) {
    try {
      $bytes = (Get-Item -LiteralPath $stackFile).Length
      Remove-Item -LiteralPath $stackFile -Force
      Write-Host "removed temporary cargo-flamegraph.stacks ($bytes bytes)"
    } catch {
      Write-Warning "could not remove temporary cargo-flamegraph.stacks; close the profiler shell and delete it manually"
    }
  }

  # The profiled Rust example intentionally avoids the Python feature to keep
  # symbol data smaller. Restore the local PyO3 extension after the run so
  # Python tooling continues to import NlheCFRTrainer correctly.
  cargo build --manifest-path tools\battery_native\Cargo.toml --release --features python --lib
  if ($LASTEXITCODE -eq 0) {
    $dll = "tools\battery_native\target\release\battery_native.dll"
    if (Test-Path $dll) {
      Copy-Item -LiteralPath $dll -Destination "tools\battery_native\target\release\battery_native.pyd" -Force
      Copy-Item -LiteralPath $dll -Destination "battery_native.pyd" -Force
      Write-Host "refreshed local PyO3 battery_native.pyd"
    }
  }
}
