"""
Full dashboard data updater.
Reads memory files, cron runs, and heartbeat history to generate a complete data.js
Run after each heartbeat or on demand.
"""
import json, os, re, glob
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
MEMORY = os.path.join(os.path.dirname(BASE), "memory")
CRON_RUNS = os.path.expanduser("~/.openclaw/cron/runs")
DATA_JS = os.path.join(BASE, "data.js")
HISTORY_FILE = os.path.join(BASE, "heartbeat-history.json")
OPENCLAW_CMD = os.path.expanduser("~/AppData/Roaming/npm/openclaw.cmd")

# Agent definitions (static metadata)
AGENTS = [
    {"name": "Jarvis", "color": "#3b82f6", "screen": "MAIN\nCORE", "hair": "#1a2a44", "shirt": "#3b82f6"},
    {"name": "Research", "color": "#22c55e", "screen": "REPORT\nDONE", "hair": "#1a3a1a", "shirt": "#22c55e"},
    {"name": "Builder", "color": "#f97316", "screen": "MEDIA\nPIPE", "hair": "#3a2a1a", "shirt": "#f97316"},
    {"name": "Outreach", "color": "#a855f7", "screen": "EXA\nSEARCH", "hair": "#2a1a3a", "shirt": "#a855f7"},
    {"name": "Scanner", "color": "#ef4444", "screen": "SCAN\nIDLE", "hair": "#3a1a1a", "shirt": "#ef4444"},
]


def get_today_memory():
    """Read today's and yesterday's memory files for task extraction."""
    now = datetime.now()
    dates = [now.strftime("%Y-%m-%d"), (now - timedelta(days=1)).strftime("%Y-%m-%d")]
    content = ""
    for d in dates:
        path = os.path.join(MEMORY, f"{d}.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content += f"\n--- {d} ---\n" + f.read()
    return content


def extract_tasks(content):
    """Extract completed, in-progress, and up-next tasks from memory content."""
    completed = []
    in_progress = []
    up_next = []
    learned = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove markdown formatting
        clean = re.sub(r'\*\*', '', line)
        clean = re.sub(r'^#{1,4}\s*', '', clean)  # strip markdown headers
        clean = re.sub(r'^[-•*]\s*', '', clean)
        clean = re.sub(r'^\d+\.\s*', '', clean)
        clean = re.sub(r'\[x\]\s*', '', clean, flags=re.IGNORECASE)  # strip checkboxes
        clean = re.sub(r'\[\s\]\s*', '', clean)

        lower = clean.lower()

        # Detect completed items
        if any(marker in lower for marker in ['✅', '✓', 'shipped', 'completed', 'done', 'built', 'deployed', 'created']):
            task = clean.lstrip('✅✓ ')
            if len(task) > 10 and task not in completed:
                completed.append(task[:100])

        # Detect in-progress
        elif any(marker in lower for marker in ['in progress', 'working on', 'building', 'wip', '🔨']):
            task = clean.lstrip('🔨 ')
            if len(task) > 10 and task not in in_progress:
                in_progress.append(task[:100])

        # Detect up-next / TODO
        elif any(marker in lower for marker in ['todo', 'up next', 'next:', 'planned', 'need to', 'should']):
            task = clean
            if len(task) > 10 and task not in up_next:
                up_next.append(task[:100])

        # Detect learned
        elif any(marker in lower for marker in ['learned', 'discovered', 'found', 'insight', 'til:']):
            if len(clean) > 10 and clean not in learned:
                learned.append(clean[:80])

    return completed[:15], in_progress[:8], up_next[:8], learned[:8]


def get_latest_heartbeats(limit=20):
    """Get the last N heartbeat entries from cron runs."""
    # Find heartbeat job
    if not os.path.exists(CRON_RUNS):
        return []

    entries = []
    for jsonl_file in glob.glob(os.path.join(CRON_RUNS, "*.jsonl")):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("action") == "finished" and entry.get("summary"):
                            entries.append(entry)
                    except:
                        continue
        except:
            continue

    # Sort by timestamp descending
    entries.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return entries[:limit]


def parse_heartbeat_items(summary):
    """Parse heartbeat summary into feed items."""
    if not summary or summary.strip() == "HEARTBEAT_OK":
        return ["All systems nominal"]

    items = []
    for line in summary.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'\*\*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        if line.startswith('#') or line.startswith('HEARTBEAT') or len(line) < 10:
            continue
        items.append(line[:120])

    return items[:8] if items else ["All systems nominal"]


def determine_agent_status(content):
    """Determine which agents are working/idle based on recent activity."""
    agents = []
    lower = content.lower()

    # Jarvis - always working if there's any activity
    jarvis = AGENTS[0].copy()
    jarvis["status"] = "working"
    jarvis["task"] = "Coordinating operations"
    if "mission control" in lower:
        jarvis["task"] = "Upgrading Mission Control"
    elif "heartbeat" in lower:
        jarvis["task"] = "Running heartbeat checks"
    agents.append(jarvis)

    # Research
    research = AGENTS[1].copy()
    if any(w in lower for w in ["paper", "arxiv", "research", "study", "survey"]):
        research["status"] = "working"
        research["task"] = "Scanning research papers"
    else:
        research["status"] = "idle"
        research["task"] = "Monitoring feeds"
    agents.append(research)

    # Builder
    builder = AGENTS[2].copy()
    if any(w in lower for w in ["building", "built", "coding", "pipeline", "tool", "script"]):
        builder["status"] = "working"
        builder["task"] = "Building tools & pipelines"
    else:
        builder["status"] = "idle"
        builder["task"] = "Awaiting build tasks"
    agents.append(builder)

    # Outreach
    outreach = AGENTS[3].copy()
    if any(w in lower for w in ["outreach", "email", "contact", "reply", "crm"]):
        outreach["status"] = "working"
        outreach["task"] = "Managing outreach pipeline"
    else:
        outreach["status"] = "idle"
        outreach["task"] = "Monitoring inbox"
    agents.append(outreach)

    # Scanner
    scanner = AGENTS[4].copy()
    if any(w in lower for w in ["scan", "market", "competitor", "monitor"]):
        scanner["status"] = "working"
        scanner["task"] = "Running market scan"
    else:
        scanner["status"] = "idle"
        scanner["task"] = "Next scan queued"
    agents.append(scanner)

    return agents


def load_heartbeat_history():
    """Load accumulated heartbeat history."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []


def save_heartbeat_history(history):
    """Save heartbeat history (keep last 20)."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-20:], f, indent=2, ensure_ascii=False)


def get_gateway_status():
    """Get gateway health info."""
    import subprocess
    try:
        result = subprocess.run(
            [OPENCLAW_CMD, "gateway", "status"],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace"
        )
        output = result.stdout + result.stderr
        # Strip ANSI codes
        output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        
        rpc_ok = "rpc probe: ok" in output.lower() or "probe: ok" in output.lower()
        service_running = "running" in output.lower() and "not running" not in output.lower()
        port = "18789"
        port_match = re.search(r'port[=:]?\s*(\d+)', output)
        if port_match:
            port = port_match.group(1)
        bind = "loopback"
        if "0.0.0.0" in output:
            bind = "0.0.0.0"
        return {
            "rpcStatus": "ok" if rpc_ok else "degraded",
            "serviceRunning": service_running,
            "port": port,
            "bind": bind,
        }
    except Exception as e:
        return {"rpcStatus": "unknown", "serviceRunning": False, "port": "18789", "bind": "loopback"}


def get_skills_status():
    """Get installed vs available skills."""
    import subprocess
    try:
        result = subprocess.run(
            [OPENCLAW_CMD, "skills", "list"],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace"
        )
        output = result.stdout + result.stderr
        # Strip ANSI codes
        output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        
        ready = []
        missing = []
        for line in output.split("\n"):
            # Match lines with box-drawing │ separator
            if "\u2502" not in line and "│" not in line:
                continue
            sep = "\u2502" if "\u2502" in line else "│"
            parts = line.split(sep)
            if len(parts) < 3:
                continue
            status_col = parts[1].strip().lower()
            name_col = parts[2].strip()
            # Remove emoji (anything non-ASCII except hyphen)
            name = re.sub(r'[^\x20-\x7E]', '', name_col).strip()
            name = re.sub(r'\s+', '-', name).strip('-')
            if not name or len(name) < 2:
                continue
            if "ready" in status_col:
                ready.append(name)
            elif "missing" in status_col:
                missing.append(name)
        return {"ready": ready, "missing": missing}
    except Exception as e:
        return {"ready": [], "missing": []}


def build_data():
    """Build the complete MISSION_DATA object."""
    content = get_today_memory()
    completed, in_progress, up_next, learned = extract_tasks(content)

    # Fallback to existing data if memory parsing yields nothing
    if not completed:
        completed = [
            "Outreach to 6 robotics companies",
            "MuJoCo sim v4 — 1.7mm accuracy",
            "E2E proof passed (100 contributors → trained policy)",
            "Demo page deployed to Vercel",
            "Discord server + workflow channels set up",
            "Voice configured (Edge TTS Andrew)",
            "Research report: robotics data buyer landscape",
        ]

    agents = determine_agent_status(content)

    # Heartbeat feed (latest)
    heartbeats = get_latest_heartbeats(20)
    heartbeat_feed = {"time": "", "items": ["Waiting for heartbeat..."]}
    heartbeat_history = load_heartbeat_history()

    if heartbeats:
        latest = heartbeats[0]
        ts = datetime.fromtimestamp(latest["ts"] / 1000, tz=timezone.utc)
        local_time = ts.astimezone()
        time_str = local_time.strftime("%I:%M %p").lstrip("0")
        items = parse_heartbeat_items(latest.get("summary", ""))
        heartbeat_feed = {"time": time_str, "items": items}

        # Add to history if it's new
        latest_id = f"{latest['ts']}"
        existing_ids = {h.get("id") for h in heartbeat_history}
        if latest_id not in existing_ids:
            heartbeat_history.append({
                "id": latest_id,
                "time": local_time.strftime("%Y-%m-%d %I:%M %p"),
                "items": items,
            })
            save_heartbeat_history(heartbeat_history)

    # Build full heartbeat history from cron if local history is sparse
    if len(heartbeat_history) < 5 and heartbeats:
        for hb in heartbeats:
            hb_id = f"{hb['ts']}"
            existing_ids = {h.get("id") for h in heartbeat_history}
            if hb_id not in existing_ids:
                ts = datetime.fromtimestamp(hb["ts"] / 1000, tz=timezone.utc)
                local_time = ts.astimezone()
                heartbeat_history.append({
                    "id": hb_id,
                    "time": local_time.strftime("%Y-%m-%d %I:%M %p"),
                    "items": parse_heartbeat_items(hb.get("summary", "")),
                })
        heartbeat_history.sort(key=lambda x: x.get("id", ""), reverse=True)
        save_heartbeat_history(heartbeat_history)

    data = {
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "agents": agents,
        "completed": completed,
        "inProgress": in_progress if in_progress else ["Dashboard data automation"],
        "upNext": up_next if up_next else [],
        "learned": learned if learned else [
            "Discord API multipart file uploads",
            "Edge TTS voice comparison (Andrew > Ryan)",
            "Proactive agent research papers",
        ],
        "heartbeatFeed": heartbeat_feed,
        "heartbeatHistory": heartbeat_history[-20:],
        "gateway": get_gateway_status(),
        "skills": get_skills_status(),
        "pendingApprovals": [],  # Placeholder for future approvals queue
    }
    return data


def write_data_js(data):
    """Write the complete data.js file."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    content = f"window.MISSION_DATA = {json_str};\n"
    with open(DATA_JS, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated data.js — {len(data['completed'])} completed, {len(data['inProgress'])} in-progress, {len(data.get('heartbeatHistory', []))} heartbeat entries")


def redeploy():
    """Redeploy to Vercel."""
    import subprocess
    try:
        vercel_path = os.path.expanduser("~/AppData/Roaming/npm/vercel.cmd")
        result = subprocess.run(
            [vercel_path, "--yes", "--prod"],
            cwd=BASE,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("Redeployed to Vercel")
        else:
            print(f"Deploy failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"Deploy error: {e}")


def main():
    data = build_data()
    write_data_js(data)
    redeploy()


if __name__ == "__main__":
    main()
