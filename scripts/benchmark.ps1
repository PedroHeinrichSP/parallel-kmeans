param(
    [int]$Observations = 1000000,
    [int]$Clusters = 5,
    [int]$Runs = 3,
    [string[]]$ThreadsCsv = @("1,2,4,8")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CpuCount = [Environment]::ProcessorCount
$Executable = Join-Path $PSScriptRoot "..\artifacts/executables/kmeans.exe"
$ThreadsCsvText = $ThreadsCsv -join ","

function Show-Usage {
    Write-Error "Uso: .\scripts\benchmark.ps1 [observacoes] [clusters] [repeticoes] [threads_csv]"
    Write-Error "Exemplo: .\scripts\benchmark.ps1 1000000 5 3 1,2,4,8"
}

if ($Observations -lt 1 -or $Clusters -lt 1 -or $Runs -lt 1) {
    Show-Usage
    exit 1
}

$Threads = @(
    $ThreadsCsvText.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) |
        ForEach-Object { $_.Trim() }
)

if ($Threads.Count -eq 0) {
    Write-Error "Erro: informe ao menos uma contagem de threads."
    exit 1
}

foreach ($ThreadCountText in $Threads) {
    $ParsedThreadCount = 0
    if (-not [int]::TryParse($ThreadCountText, [ref]$ParsedThreadCount) -or $ParsedThreadCount -lt 1) {
        Write-Error "Erro: valor invalido na lista de threads: $ThreadCountText"
        exit 1
    }
}

if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    Write-Error "Erro: gcc nao esta disponivel no PATH."
    exit 1
}

Write-Host "Compilando o projeto..."

$ArtifactsDir = Join-Path $PSScriptRoot "..\artifacts\executables"
New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null

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

Push-Location (Join-Path $PSScriptRoot "..")
try {
    & gcc @CompileArgs *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "gcc terminou com codigo $LASTEXITCODE."
    }

    Copy-Item -Force "artifacts/executables/kmeans.exe" "artifacts/executables/kmeans_seq.exe"
} finally {
    Pop-Location
}

$Results = New-Object System.Collections.Generic.List[object]

function Invoke-MeasureCase {
    param(
        [string]$Label,
        [string]$Algorithm,
        [int]$ThreadCount
    )

    $WaitPolicy = "ACTIVE"
    if ($ThreadCount -gt $CpuCount) {
        $WaitPolicy = "PASSIVE"
    }

    Write-Host ""
    Write-Host "Executando $Label"

    $PreviousSkipEps = $env:KMEANS_SKIP_EPS
    $PreviousOmpDynamic = $env:OMP_DYNAMIC
    $PreviousOmpProcBind = $env:OMP_PROC_BIND
    $PreviousOmpWaitPolicy = $env:OMP_WAIT_POLICY
    $PreviousOmpNumThreads = $env:OMP_NUM_THREADS

    try {
        $env:KMEANS_SKIP_EPS = "1"
        $env:OMP_DYNAMIC = "FALSE"
        $env:OMP_PROC_BIND = "TRUE"
        $env:OMP_WAIT_POLICY = $WaitPolicy
        $env:OMP_NUM_THREADS = [string]$ThreadCount

        for ($RunIndex = 1; $RunIndex -le $Runs; $RunIndex++) {
            $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            & $Executable $Observations $Clusters $Algorithm *> $null
            $ExitCode = $LASTEXITCODE
            $Stopwatch.Stop()

            if ($ExitCode -ne 0) {
                throw "$Executable terminou com codigo $ExitCode."
            }

            $Elapsed = $Stopwatch.Elapsed.TotalSeconds
            $Results.Add([pscustomobject]@{
                Label = $Label
                Threads = $ThreadCount
                Run = $RunIndex
                Elapsed = $Elapsed
            }) | Out-Null

            Write-Host ("  run {0}/{1}: {2:N6}s" -f $RunIndex, $Runs, $Elapsed)
        }
    } finally {
        $env:KMEANS_SKIP_EPS = $PreviousSkipEps
        $env:OMP_DYNAMIC = $PreviousOmpDynamic
        $env:OMP_PROC_BIND = $PreviousOmpProcBind
        $env:OMP_WAIT_POLICY = $PreviousOmpWaitPolicy
        $env:OMP_NUM_THREADS = $PreviousOmpNumThreads
    }
}

Invoke-MeasureCase -Label "seq" -Algorithm "s" -ThreadCount 1

foreach ($ThreadCountText in $Threads) {
    $ThreadCount = [int]$ThreadCountText
    Invoke-MeasureCase -Label "omp-$($ThreadCount)t" -Algorithm "o" -ThreadCount $ThreadCount
}

Write-Host ""
Write-Host "Resumo"
Write-Host ("{0,-10} {1,-8} {2,-10} {3,-10} {4,-10} {5,-10}" -f "modo", "threads", "media(s)", "min(s)", "max(s)", "speedup")

$SeqGroup = $Results | Where-Object { $_.Label -eq "seq" }
$SeqAvg = ($SeqGroup | Measure-Object -Property Elapsed -Average).Average

$OrderedLabels = $Results | Select-Object -ExpandProperty Label -Unique

foreach ($Label in $OrderedLabels) {
        $Group = @($Results | Where-Object { $_.Label -eq $Label })
        $Stats = $Group | Measure-Object -Property Elapsed -Average -Minimum -Maximum
        $Average = $Stats.Average
        $Speedup = "n/a"

        if ($Average -gt 0 -and $SeqAvg -gt 0) {
            $Speedup = "{0:N2}x" -f ($SeqAvg / $Average)
        }

        Write-Host ("{0,-10} {1,-8} {2,-10:N6} {3,-10:N6} {4,-10:N6} {5,-10}" -f `
            $Label, $Group[0].Threads, $Stats.Average, $Stats.Minimum, $Stats.Maximum, $Speedup)
}

Write-Host ""
Write-Host "Parametros usados: observacoes=$Observations clusters=$Clusters repeticoes=$Runs threads=$ThreadsCsvText"
