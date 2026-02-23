# Gateway Watchdog - Runs every 5 minutes via Task Scheduler
# Checks if the OpenClaw gateway is responding, restarts if not

$logFile = "$env:USERPROFILE\clawd\tools\gateway_watchdog.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.Connect("127.0.0.1", 18789)
    $tcp.Close()
    # Gateway is alive, do nothing
    Add-Content $logFile "$timestamp OK - Gateway listening on port 18789"
} catch {
    # Gateway is down - restart it
    Add-Content $logFile "$timestamp DOWN - Restarting gateway..."
    
    # Also run the discord config watchdog first to ensure config is intact
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pythonPath) {
        & $pythonPath "$env:USERPROFILE\clawd\tools\discord_watchdog.py" --fix 2>&1 | Out-Null
        Add-Content $logFile "$timestamp WATCHDOG - Discord config checked/restored"
    }
    
    # Start the gateway
    $nodePath = "C:\Program Files\nodejs\node.exe"
    $openclawPath = "$env:APPDATA\npm\node_modules\openclaw\dist\index.js"
    Start-Process -FilePath $nodePath -ArgumentList $openclawPath,"gateway","start" -WindowStyle Hidden
    Add-Content $logFile "$timestamp RESTARTED - Gateway start command issued"
}

# Trim log to last 200 lines
if (Test-Path $logFile) {
    $lines = Get-Content $logFile -Tail 200
    Set-Content $logFile $lines
}
