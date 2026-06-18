param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Join-Path $PSScriptRoot ".."
$ArtifactsExecutables = Join-Path $ProjectRoot "artifacts\executables"
$ArtifactsObjects = Join-Path $ProjectRoot "artifacts\objects"

if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    Write-Error "Erro: gcc nao esta disponivel no PATH."
    exit 1
}

New-Item -ItemType Directory -Force -Path $ArtifactsExecutables | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactsObjects | Out-Null

$DefaultCFlags = @("-O3", "-mtune=native", "-fno-math-errno")
$CFlags = if ($env:CFLAGS) {
    $env:CFLAGS.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
} else {
    $DefaultCFlags
}

$CompileArgs = @(
    $CFlags
    "code/runner.c"
    "code/k_means_clustering_utils.c"
    "code/k_means_clustering_sequencial.c"
    "code/k_means_clustering_openmp.c"
    "code/k_means_clustering_openmp_gpu.c"
    "code/k_means_clustering_cuda.c"
    "-Icode/headers"
    "-fopenmp"
    "-o"
    "artifacts/executables/kmeans.exe"
    "-lm"
)

Push-Location $ProjectRoot
try {
    & gcc @CompileArgs
    if ($LASTEXITCODE -ne 0) {
        throw "gcc terminou com codigo $LASTEXITCODE."
    }

    Copy-Item -Force "artifacts/executables/kmeans.exe" "artifacts/executables/kmeans_seq.exe"
} finally {
    Pop-Location
}
