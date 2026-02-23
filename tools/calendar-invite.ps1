param(
    [Parameter(Mandatory=$true)][string]$To,
    [Parameter(Mandatory=$true)][string]$Subject,
    [string]$Body = "",
    [Parameter(Mandatory=$true)][string]$Start,
    [Parameter(Mandatory=$true)][string]$End,
    [string]$Location = ""
)

$user = "autonomousnyamekye@gmail.com"
$pass = "cetgmwrksfknhzkh"

$startDt = [DateTime]::Parse($Start).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$endDt = [DateTime]::Parse($End).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$uid = [guid]::NewGuid().ToString()
$now = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")

$ical = @"
BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Jarvis//EN
METHOD:REQUEST
BEGIN:VEVENT
UID:$uid
DTSTART:$startDt
DTEND:$endDt
DTSTAMP:$now
ORGANIZER;CN=Jarvis:mailto:$user
ATTENDEE;ROLE=REQ-PARTICIPANT;RSVP=TRUE:mailto:$To
SUMMARY:$Subject
DESCRIPTION:$Body
LOCATION:$Location
STATUS:CONFIRMED
SEQUENCE:0
END:VEVENT
END:VCALENDAR
"@

$smtpClient = New-Object System.Net.Mail.SmtpClient("smtp.gmail.com", 587)
$smtpClient.EnableSsl = $true
$smtpClient.Credentials = New-Object System.Net.NetworkCredential($user, $pass)

$msg = New-Object System.Net.Mail.MailMessage
$msg.From = New-Object System.Net.Mail.MailAddress($user, "Jarvis")
$msg.To.Add($To)
$msg.Subject = $Subject
$msg.Body = $Body

$calBytes = [System.Text.Encoding]::UTF8.GetBytes($ical)
$calStream = New-Object System.IO.MemoryStream(,$calBytes)
$calAttachment = New-Object System.Net.Mail.Attachment($calStream, "invite.ics", "text/calendar")
$msg.Attachments.Add($calAttachment)

$altView = [System.Net.Mail.AlternateView]::CreateAlternateViewFromString($ical, [System.Text.Encoding]::UTF8, "text/calendar")
$altView.TransferEncoding = [System.Net.Mime.TransferEncoding]::Base64
$msg.AlternateViews.Add($altView)

$smtpClient.Send($msg)
Write-Host "Calendar invite sent to $To for $Start"

$msg.Dispose()
$calStream.Dispose()
$smtpClient.Dispose()
