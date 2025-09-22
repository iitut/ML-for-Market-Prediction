# Git helper tools

- **sync_now.bat** — one click: stage all, commit, push.
- **autopush.ps1** — PowerShell version of the same.
- **watch_and_push.ps1** — watches the repo and pushes on every save (debounced).

## Quick start (Windows)
1. Double-click \	ools\git\sync_now.bat\ to sync once.
2. Or run:  
   \powershell -ExecutionPolicy Bypass -File tools\git\watch_and_push.ps1\  
   (keeps syncing on every change; close the window to stop.)

## Requirements
- Git + Git LFS installed
- Origin remote set to your GitHub repo (see “Re-connect on a new laptop” in the top-level README).
