# Run as Administrator: Right-click PowerShell > Run as Administrator, then run this script
# Restricts RDP and SMB to Private network only (home), blocks on Public WiFi

Write-Host "Hardening Windows Firewall..." -ForegroundColor Cyan

# Restrict RDP to Private network only
Set-NetFirewallRule -DisplayName "Remote Desktop - User Mode (TCP-In)" -Profile Private
Set-NetFirewallRule -DisplayName "Remote Desktop - User Mode (UDP-In)" -Profile Private
Set-NetFirewallRule -DisplayName "Remote Desktop - Shadow (TCP-In)" -Profile Private
Get-NetFirewallRule -DisplayName "Remote Desktop - (TCP-WSS-In)" | Set-NetFirewallRule -Profile Private
Write-Host "[OK] RDP restricted to Private network only" -ForegroundColor Green

# Restrict SMB/File Sharing to Private only
Get-NetFirewallRule -DisplayName "*File and Printer Sharing*" -ErrorAction SilentlyContinue |
    Where-Object { $_.Enabled -eq "True" } |
    Set-NetFirewallRule -Profile Private
Write-Host "[OK] SMB/File sharing restricted to Private network only" -ForegroundColor Green

# Block UPnP on Public
Get-NetFirewallRule -DisplayName "*UPnP*" -ErrorAction SilentlyContinue |
    Set-NetFirewallRule -Profile Private
Write-Host "[OK] UPnP restricted to Private network only" -ForegroundColor Green

# Verify
Write-Host "`nVerification:" -ForegroundColor Yellow
Get-NetFirewallRule -DisplayName "*Remote Desktop*" | Where-Object { $_.Enabled -eq "True" } |
    Select-Object DisplayName, Profile | Format-Table -AutoSize

Write-Host "Done. RDP/SMB/UPnP now blocked on public WiFi." -ForegroundColor Green
