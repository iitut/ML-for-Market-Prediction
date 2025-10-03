@'
$ErrorActionPreference = "Stop"

# Repo root = two levels above tools\git
$Repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $Repo

git lfs install | Out-Null
git lfs track "*.csv" | Out-Null
git lfs track "*.parquet" | Out-Null
git add .gitattributes 2>$null

git add -A
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
  Write-Host "[autopush] Nothing to commit."
  exit 0
}

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Auto-sync: $ts" | Out-Null

try {
  git push -u origin main
} catch {
  Write-Host "[autopush] Push rejected, rebasing..."
  git pull --rebase origin main
  git push -u origin main
}
Write-Host "[autopush] Done."
'@ | Set-Content ".\tools\git\autopush.ps1"
