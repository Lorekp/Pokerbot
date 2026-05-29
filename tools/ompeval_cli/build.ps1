# Build enumerate_hu_equity + hu_table_native + hu_table_build + bench_native (OMPEval required)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Test-Path "vendor/OMPEval/omp/HandEvaluator.h")) {
    Write-Error "OMPEval vendor missing. Run: git submodule update --init vendor/OMPEval"
}

function Find-CMake {
    if (Get-Command cmake -ErrorAction SilentlyContinue) { return "cmake" }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\JetBrains\CLion 2025.1.3\bin\cmake\win\x64\bin\cmake.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    throw "cmake not found; install CMake or VS Build Tools"
}

$cmake = Find-CMake
& $cmake -B build -DUSE_OMPEVAL=ON -DBUILD_BENCH_NATIVE=ON
& $cmake --build build --config Release

Write-Host "Built:"
Write-Host "  build/Release/enumerate_hu_equity.exe"
Write-Host "  build/Release/hu_table_native*.pyd"
Write-Host "  build/Release/hu_table_build.exe"
Write-Host "  build/Release/bench_native.exe"
