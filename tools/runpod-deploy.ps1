# RunPod deploy + auto-setup
# Usage: .\runpod-deploy.ps1 [-GpuType "NVIDIA GeForce RTX 3090"] [-CloudType "SECURE"]
param(
    [string]$GpuType = "NVIDIA GeForce RTX 3090",
    [string]$CloudType = "SECURE",
    [string]$Name = "jarvis-pipeline"
)

$env:RUNPOD_API_KEY = (Get-Content "$env:USERPROFILE\.openclaw\.env" | Select-String "RUNPOD_API_KEY" | ForEach-Object { $_.ToString().Split('=',2)[1] })

Write-Host "Deploying $Name ($GpuType, $CloudType)..."

# Deploy pod
$pod = python -c "
import runpod, json
runpod.api_key = '$env:RUNPOD_API_KEY'
pod = runpod.create_pod(name='$Name', gpu_type_id='$GpuType', cloud_type='$CloudType',
    template_id='runpod-torch-v240', volume_in_gb=20, container_disk_in_gb=10)
print(json.dumps({'id': pod['id']}))
" | ConvertFrom-Json

$podId = $pod.id
Write-Host "Pod created: $podId"

# Wait for runtime
Write-Host "Waiting for pod to boot..."
$maxWait = 300
$elapsed = 0
while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds 15
    $elapsed += 15
    $status = python -c "
import runpod
runpod.api_key = '$env:RUNPOD_API_KEY'
pod = runpod.get_pod('$podId')
rt = pod.get('runtime')
if rt: print('UP')
else: print('BOOTING')
"
    Write-Host "  ${elapsed}s: $status"
    if ($status.Trim() -eq "UP") { break }
}

if ($status.Trim() -ne "UP") {
    Write-Host "ERROR: Pod failed to boot in ${maxWait}s"
    exit 1
}

# Upload and run setup script via ssh.runpod.io proxy
Write-Host "Running setup script..."
$sshHost = "$podId-64411eb6@ssh.runpod.io"
$keyPath = "$env:USERPROFILE\.ssh\id_ed25519"
$setupScript = "$PSScriptRoot\runpod-setup.sh"

# Copy setup script
scp -i $keyPath -o StrictHostKeyChecking=no $setupScript "${sshHost}:/tmp/setup.sh"
# Run it
ssh -i $keyPath -o StrictHostKeyChecking=no $sshHost "bash /tmp/setup.sh"

Write-Host ""
Write-Host "=== POD READY ==="
Write-Host "ID: $podId"
Write-Host "SSH: ssh $sshHost -i ~/.ssh/id_ed25519"
Write-Host "SSH (direct): ssh runpod  (uses ~/.ssh/config)"
