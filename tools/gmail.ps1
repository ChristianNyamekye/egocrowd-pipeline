param(
    [Parameter(Mandatory=$true)][string]$Action,
    [string]$To,
    [string]$Subject,
    [string]$Body,
    [int]$Count = 5
)

$user = "autonomousnyamekye@gmail.com"
$pass = "cetgmwrksfknhzkh"
$secPass = ConvertTo-SecureString $pass -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential($user, $secPass)

switch ($Action) {
    "send" {
        if (-not $To -or -not $Subject) { Write-Error "Usage: -Action send -To <email> -Subject <subject> -Body <body>"; exit 1 }
        $params = @{
            From = $user
            To = $To
            Subject = $Subject
            Body = $Body
            SmtpServer = "smtp.gmail.com"
            Port = 587
            UseSsl = $true
            Credential = $cred
        }
        Send-MailMessage @params
        Write-Host "Email sent to $To"
    }
    "test" {
        $params = @{
            From = $user
            To = $user
            Subject = "Jarvis Test - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
            Body = "This is an automated test from Jarvis. If you see this, Gmail access is working."
            SmtpServer = "smtp.gmail.com"
            Port = 587
            UseSsl = $true
            Credential = $cred
        }
        Send-MailMessage @params
        Write-Host "Test email sent to $user"
    }
    default { Write-Error "Unknown action: $Action. Use 'send' or 'test'" }
}
