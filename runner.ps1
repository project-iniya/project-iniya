# ===============================
# Portable Runner + Timestamp Log
# (FORCE VENV MODE)
# ===============================


# --- Get executable/script directory safely ---
if ($MyInvocation.MyCommand.Path) {
    $BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $BaseDir = Split-Path -Parent ([System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName)
}

# --- Ensure log folder exists ---
$LogDir = Join-Path $BaseDir "log"
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# --- Build timestamped log filename ---
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile = Join-Path $LogDir ("iniya.$Timestamp.log")

# --- Force use of venv python ---
$VenvPython = Join-Path $BaseDir ".venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
    Write-Host "ERROR: .venv python not found"
    exit 1
}

# --- Target script ---
$MainPy = Join-Path $BaseDir "main.example.py"

if (!(Test-Path $MainPy)) {
    Write-Host "ERROR: main.example.py not found"
    exit 1
}

# --- Run program using venv python (console + log) ---
& $VenvPython $MainPy *>&1 | Tee-Object -FilePath $LogFile -Append
