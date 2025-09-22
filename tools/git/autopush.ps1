# autopush.ps1 — safe auto-commit & push for ML_MP
# Location: C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\git_set\autopush.ps1

$ErrorActionPreference = "Stop"

# --- repo root ---
$Repo = "C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP"
Set-Location $Repo

# --- sanity checks ---
# Must have a remote named origin set once (you already have it)
git remote -v | Out-Null

# --- LFS (needed for files >100MB) ---
git lfs install | Out-Null
git lfs track "*.csv" | Out-Null
git lfs track "*.parquet" | Out-Null
# make sure .gitattributes is tracked if changed
git add .gitattributes 2>$null

# --- stage everything (respecting any .gitignore if you add one later) ---
git add -A

# nothing staged? exit quietly
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
  Write-Host "[autopush] Nothing to commit."
  exit 0
}

# --- commit with timestamp ---
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Auto-sync: $ts" | Out-Null

# --- try push; if rejected, rebase then push ---
try {
  git push -u origin main
} catch {
  Write-Host "[autopush] Push rejected, rebasing..."
  git pull --rebase origin main
  git push -u origin main
}

Write-Host "[autopush] Done."
