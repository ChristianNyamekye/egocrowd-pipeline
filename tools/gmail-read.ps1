param(
    [int]$Count = 5,
    [string]$Folder = "INBOX",
    [switch]$Unread
)

Add-Type -AssemblyName System.Net.Mail
Add-Type -AssemblyName System.Net.Security

$user = "autonomousnyamekye@gmail.com"
$pass = "cetgmwrksfknhzkh"

# Use .NET TcpClient + SslStream for IMAP
$tcp = New-Object System.Net.Sockets.TcpClient("imap.gmail.com", 993)
$ssl = New-Object System.Net.Security.SslStream($tcp.GetStream(), $false)
$ssl.AuthenticateAsClient("imap.gmail.com")

$reader = New-Object System.IO.StreamReader($ssl)
$writer = New-Object System.IO.StreamWriter($ssl)
$writer.AutoFlush = $true

function Send-IMAP($cmd) {
    $tag = "A$(Get-Random -Maximum 9999)"
    $writer.WriteLine("$tag $cmd")
    $lines = @()
    while ($true) {
        $line = $reader.ReadLine()
        $lines += $line
        if ($line -match "^$tag ") { break }
    }
    return $lines
}

# Greeting
$reader.ReadLine() | Out-Null

# Login
Send-IMAP "LOGIN $user $pass" | Out-Null

# Select folder
$selectResult = Send-IMAP "SELECT $Folder"
$totalMessages = 0
foreach ($line in $selectResult) {
    if ($line -match "(\d+) EXISTS") { $totalMessages = [int]$Matches[1] }
}

if ($totalMessages -eq 0) {
    Write-Host "No messages in $Folder"
    Send-IMAP "LOGOUT" | Out-Null
    exit
}

# Fetch latest N
$start = [Math]::Max(1, $totalMessages - $Count + 1)
$range = "${start}:${totalMessages}"

if ($Unread) {
    $searchResult = Send-IMAP "SEARCH UNSEEN"
    $ids = @()
    foreach ($line in $searchResult) {
        if ($line -match "^\* SEARCH (.+)") {
            $ids = $Matches[1].Trim().Split(" ") | Select-Object -Last $Count
        }
    }
    if ($ids.Count -eq 0) {
        Write-Host "No unread messages"
        Send-IMAP "LOGOUT" | Out-Null
        exit
    }
    $range = $ids -join ","
}

$fetchResult = Send-IMAP "FETCH $range (BODY[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS)"

$currentEmail = @{}
$emails = @()

foreach ($line in $fetchResult) {
    if ($line -match "From:\s*(.+)") { $currentEmail.From = $Matches[1].Trim() }
    if ($line -match "Subject:\s*(.+)") { $currentEmail.Subject = $Matches[1].Trim() }
    if ($line -match "Date:\s*(.+)") { $currentEmail.Date = $Matches[1].Trim() }
    if ($line -match "FLAGS \(([^)]*)\)") { 
        $currentEmail.Flags = $Matches[1]
        if ($currentEmail.From -or $currentEmail.Subject) {
            $emails += $currentEmail.Clone()
        }
        $currentEmail = @{}
    }
}

Write-Host "`n$Folder - $($emails.Count) messages:`n"
foreach ($e in $emails) {
    $unreadMark = if ($e.Flags -notmatch "\\Seen") { "[NEW] " } else { "      " }
    Write-Host "${unreadMark}From: $($e.From)"
    Write-Host "  Subject: $($e.Subject)"
    Write-Host "  Date: $($e.Date)"
    Write-Host ""
}

Send-IMAP "LOGOUT" | Out-Null
$ssl.Close()
$tcp.Close()
