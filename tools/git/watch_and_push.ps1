# watch_and_push.ps1 — auto-commit & push on file changes (debounced)
# Location: C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\git_set\watch_and_push.ps1

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP"
Set-Location $Repo

function Sync-Git {
  try {
    # Ensure LFS for big files (csv/parquet)
    git lfs install | Out-Null
    git lfs track "*.csv" | Out-Null
    git lfs track "*.parquet" | Out-Null
    git add .gitattributes 2>$null

    # Stage everything and check if anything is staged
    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { return }

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Auto-sync (watcher): $ts" | Out-Null

    try {
      git push -u origin main
    } catch {
      # If remote moved, rebase then push
      git pull --rebase origin main
      git push -u origin main
    }
  } catch {
    Write-Host "[watch] Error: $($_.Exception.Message)"
  }
}

# --- FileSystemWatcher setup ---
$fsw = New-Object IO.FileSystemWatcher $Repo, "*"
$fsw.IncludeSubdirectories = $true
$fsw.EnableRaisingEvents = $true

# Register events (Changed/Created/Deleted/Renamed)
$ids = @()
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Changed -SourceIdentifier FSChanged)
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Created -SourceIdentifier FSCreated)
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Deleted -SourceIdentifier FSDeleted)
$ids += (Register-ObjectEvent -InputObject $fsw -EventName Renamed -SourceIdentifier FSRenamed)

Write-Host "[watch] Started for $Repo (Ctrl+C to stop)"

# Debounce loop: wait for any FS* event, pause briefly, drain queue, then sync
while ($true) {
  Wait-Event -SourceIdentifier FS* | Out-Null
  Start-Sleep -Seconds 3               # debounce burst of saves
  Get-Event | Remove-Event -ErrorAction SilentlyContinue
  Sync-Git
}
