param(
    [int]$Observations = 1000000,
    [int]$Clusters = 5,
    [string]$Algorithm = "s"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Join-Path $PSScriptRoot ".."
$Executable = Join-Path $ProjectRoot "artifacts\executables\kmeans.exe"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "artifacts\images\eps") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "artifacts\images\png") | Out-Null

if (-not (Test-Path $Executable)) {
    Write-Error "Erro: executavel nao encontrado em $Executable. Execute .\scripts\compile.ps1 primeiro."
    exit 1
}

$Elapsed = Measure-Command {
    & $Executable $Observations $Clusters $Algorithm
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable terminou com codigo $LASTEXITCODE."
    }
}

Write-Host ("Tempo total: {0:N6}s" -f $Elapsed.TotalSeconds)
