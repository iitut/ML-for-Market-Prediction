@'
$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $Repo

function Sync-Git {
  try {
    git lfs install | Out-Null
    git lfs track "*.csv" | Out-Null
    git lfs track "*.parquet" | Out-Null
    git add .gitattributes 2>$null

    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { return }

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Auto-sync (watcher): $ts" | Out-Null

    try {
      git push -u origin main
    } catch {
      git pull --rebase origin main
      git push -u origin main
    }
  } catch {
    Write-Host "[watch] Error: $($_.Exception.Message)"
  }
}

# Watch the entire repo
$fsw = New-Object IO.FileSystemWatcher $Repo, "*"
$fsw.IncludeSubdirectories = $true
$fsw.EnableRaisingEvents   = $true

$ids = @()
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Changed -SourceIdentifier FSChanged)
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Created -SourceIdentifier FSCreated)
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Deleted -SourceIdentifier FSDeleted)
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Renamed -SourceIdentifier FSRenamed)

Write-Host "[watch] Started for $Repo (Ctrl+C to stop)"

while ($true) {
  Wait-Event -SourceIdentifier FS* | Out-Null
  Start-Sleep -Seconds 3
  Get-Event | Remove-Event -ErrorAction SilentlyContinue
  Sync-Git
}
'@ | Set-Content ".\tools\git\watch_and_push.ps1"
