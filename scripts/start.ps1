param([switch]$Stop)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runtimeDir = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$pidFile = Join-Path $runtimeDir 'services.json'
if ($Stop) {
    if (Test-Path -LiteralPath $pidFile) {
        $services = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
        foreach ($service in $services) {
            $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($service.pid)" -ErrorAction SilentlyContinue
            if ($processInfo -and $processInfo.ExecutablePath -eq $pythonExe -and $processInfo.CommandLine.Contains($service.module)) {
                # Windows venv launchers can spawn a child Python process. Stop the
                # verified service's descendants before stopping its launcher.
                $allProcesses = @(Get-CimInstance Win32_Process)
                $ownedIds = @([int]$service.pid)
                $descendantIds = @()
                for ($depth = 0; $depth -lt 8; $depth++) {
                    $children = @($allProcesses | Where-Object { $_.ParentProcessId -in $ownedIds -and $_.ProcessId -notin $ownedIds })
                    if ($children.Count -eq 0) { break }
                    $childIds = @($children | ForEach-Object { [int]$_.ProcessId })
                    $descendantIds += $childIds
                    $ownedIds += $childIds
                }
                [array]::Reverse($descendantIds)
                foreach ($childId in $descendantIds) { Stop-Process -Id $childId -ErrorAction SilentlyContinue }
                Stop-Process -Id $service.pid -ErrorAction SilentlyContinue
            }
        }
        Remove-Item -LiteralPath $pidFile
    }
    Write-Host 'Project services stopped.'
    exit
}
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Create .venv and install the project first; see README.md.' }
if (Test-Path -LiteralPath $pidFile) { throw 'Run scripts/start.ps1 -Stop before starting again.' }
foreach ($port in @(8000, 8501)) {
    if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. No existing process was stopped."
    }
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$specs = @(
    @{name='api'; module='uvicorn'; args='-m uvicorn visualguard.api.main:app --host 127.0.0.1 --port 8000'},
    @{name='worker'; module='visualguard.worker.main'; args='-m visualguard.worker.main'},
    @{name='dashboard'; module='scripts/dashboard.py'; args='scripts/dashboard.py --server.port 8501'}
)
$started = @()
try {
    foreach ($spec in $specs) {
        $service = Start-Process -FilePath $pythonExe -ArgumentList $spec.args -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir "$($spec.name).log") -RedirectStandardError (Join-Path $runtimeDir "$($spec.name).error.log")
        $started += @{name=$spec.name; pid=$service.Id; module=$spec.module}
        if ($spec.name -eq 'api') {
            $ready = $false
            for ($attempt = 0; $attempt -lt 30; $attempt++) {
                if ($service.HasExited) { throw 'API exited during startup; inspect runtime/api.error.log.' }
                try {
                    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 1
                    if ($health.status -eq 'ok') { $ready = $true; break }
                } catch { Start-Sleep -Milliseconds 500 }
            }
            if (-not $ready) { throw 'API did not become ready; inspect runtime/api.error.log.' }
        }
    }
    ConvertTo-Json -InputObject $started | Set-Content -LiteralPath $pidFile
    Write-Host 'Dashboard: http://127.0.0.1:8501 | API: http://127.0.0.1:8000/docs'
    Write-Host 'Logs and process IDs are in runtime/. Stop with: ./scripts/start.ps1 -Stop'
} catch {
    foreach ($service in $started) { Stop-Process -Id $service.pid -ErrorAction SilentlyContinue }
    throw
}
