# outreach-tracker.ps1 — CRM helper for robotics outreach pipeline
# Usage: .\outreach-tracker.ps1 -Action <status|mark-sent|schedule-followup> [-Company <name>]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("status","mark-sent","due-followups")]
    [string]$Action,
    [string]$Company
)

$crmPath = "$PSScriptRoot\..\memory\outreach-crm.json"
$crm = Get-Content $crmPath -Raw | ConvertFrom-Json

switch ($Action) {
    "status" {
        Write-Host "`n=== OUTREACH PIPELINE ===" -ForegroundColor Cyan
        foreach ($lead in $crm.pipeline) {
            $color = switch ($lead.status) {
                "pending_send" { "Yellow" }
                "sent" { "Green" }
                "responded" { "Magenta" }
                "meeting_booked" { "Cyan" }
                default { "Gray" }
            }
            Write-Host "  [$($lead.status.ToUpper().PadRight(15))] $($lead.company) — $($lead.contact)" -ForegroundColor $color
            if ($lead.sentDate) { Write-Host "    Sent: $($lead.sentDate)" -ForegroundColor DarkGray }
            if ($lead.response) { Write-Host "    Response: $($lead.response)" -ForegroundColor Green }
        }
        Write-Host "`n=== COMPETITORS ===" -ForegroundColor Red
        foreach ($comp in $crm.competitors) {
            Write-Host "  $($comp.name)" -ForegroundColor Yellow
            Write-Host "    Model: $($comp.model)" -ForegroundColor DarkGray
            Write-Host "    Our Edge: $($comp.ourEdge)" -ForegroundColor Green
        }
    }
    "mark-sent" {
        if (-not $Company) { Write-Host "Need -Company parameter"; exit 1 }
        $lead = $crm.pipeline | Where-Object { $_.company -like "*$Company*" }
        if ($lead) {
            $lead.status = "sent"
            $lead.sentDate = (Get-Date).ToString("yyyy-MM-dd")
            $lead.followUp1 = (Get-Date).AddDays(4).ToString("yyyy-MM-dd")
            $lead.followUp2 = (Get-Date).AddDays(10).ToString("yyyy-MM-dd")
            $crm.lastUpdated = (Get-Date).ToString("o")
            $crm | ConvertTo-Json -Depth 4 | Set-Content $crmPath
            Write-Host "Marked $($lead.company) as SENT. Follow-up 1: $($lead.followUp1), Follow-up 2: $($lead.followUp2)"
        } else { Write-Host "Company '$Company' not found" }
    }
    "due-followups" {
        $today = (Get-Date).ToString("yyyy-MM-dd")
        $due = $crm.pipeline | Where-Object { 
            ($_.followUp1 -eq $today -or $_.followUp2 -eq $today) -and $_.status -ne "responded"
        }
        if ($due) {
            Write-Host "`n=== FOLLOW-UPS DUE TODAY ===" -ForegroundColor Yellow
            foreach ($d in $due) {
                Write-Host "  $($d.company) — $($d.contact)" -ForegroundColor Cyan
            }
        } else {
            Write-Host "No follow-ups due today."
        }
    }
}
