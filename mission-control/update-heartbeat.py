"""
ONLY way to update Mission Control data.js heartbeat feed.
Reads latest heartbeat cron run → updates ONLY the heartbeatFeed field → redeploys.
NEVER overwrites the rest of data.js. Validates output before writing.
"""
import json, os, re, copy
from datetime import datetime, timezone

CRON_RUNS = os.path.expanduser("~/.openclaw/cron/runs")
HEARTBEAT_JOB_ID = "a116194c-990e-4169-84d0-f0c3848040ed"
DATA_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.js")
TEMPLATE_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.template.js")

def get_latest_heartbeat():
    run_file = os.path.join(CRON_RUNS, f"{HEARTBEAT_JOB_ID}.jsonl")
    if not os.path.exists(run_file):
        return None
    with open(run_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in reversed(lines):
        try:
            entry = json.loads(line.strip())
            if entry.get("action") == "finished" and entry.get("summary"):
                return entry
        except:
            continue
    return None

def parse_accomplishments(summary):
    if not summary or summary.strip() == "HEARTBEAT_OK":
        return ["All systems nominal — nothing needed attention"]
    items = []
    for line in summary.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'\*\*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        if line.startswith('#') or line.startswith('HEARTBEAT') or len(line) < 10:
            continue
        # Truncate long lines
        if len(line) > 120:
            line = line[:117] + "..."
        items.append(line)
    return items[:6] if items else ["All systems nominal — nothing needed attention"]

def safe_update_data_js(feed_data):
    """
    Safely update ONLY the heartbeatFeed + lastUpdated in data.js.
    Parses the JS object, modifies only the feed, writes back.
    Validates that the output is valid before writing.
    """
    with open(DATA_JS, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the JSON object from "window.MISSION_DATA = {...};"
    match = re.search(r'window\.MISSION_DATA\s*=\s*(\{.*\})\s*;?\s*$', content, re.DOTALL)
    if not match:
        print("ERROR: Could not parse data.js — skipping update to avoid corruption")
        return False
    
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        print("ERROR: data.js JSON is invalid — skipping update to avoid corruption")
        return False
    
    # Store original for validation
    original_keys = set(data.keys())
    
    # Update ONLY heartbeatFeed and lastUpdated
    data["heartbeatFeed"] = feed_data
    data["lastUpdated"] = datetime.now(timezone.utc).isoformat()
    
    # Validate: all original keys still present
    if not original_keys.issubset(set(data.keys())):
        print("ERROR: Update would remove keys — aborting")
        return False
    
    # Validate: agents array still has 5 entries
    if len(data.get("agents", [])) != 5:
        print("ERROR: agents array corrupted — aborting")
        return False
    
    # Write back
    new_content = "window.MISSION_DATA = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n"
    
    # Final size check — shouldn't be dramatically different
    if len(new_content) < 500:
        print("ERROR: Output suspiciously small — aborting")
        return False
    
    with open(DATA_JS, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def redeploy():
    import subprocess
    # Load Vercel token from env or .env file
    token = os.environ.get("VERCEL_TOKEN")
    if not token:
        env_path = os.path.expanduser("~/.openclaw/.env")
        if os.path.exists(env_path):
            with open(env_path) as ef:
                for line in ef:
                    if line.strip().startswith("VERCEL_TOKEN="):
                        token = line.strip().split("=", 1)[1]
                        break
    if not token:
        print("ERROR: No VERCEL_TOKEN found in env or .env")
        return
    try:
        vercel_path = os.path.expanduser("~/AppData/Roaming/npm/vercel.cmd")
        result = subprocess.run(
            [vercel_path, "--token", token, "--yes", "--prod"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("Redeployed to Vercel")
        else:
            print(f"Deploy failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"Deploy error: {e}")

def main():
    entry = get_latest_heartbeat()
    if not entry:
        print("No heartbeat runs found")
        return
    
    ts = datetime.fromtimestamp(entry["ts"] / 1000, tz=timezone.utc)
    local_time = ts.astimezone()
    time_str = local_time.strftime("%I:%M %p").lstrip("0")
    
    items = parse_accomplishments(entry.get("summary", ""))
    
    feed = {
        "time": time_str,
        "items": items
    }
    
    if safe_update_data_js(feed):
        print(f"Updated heartbeat feed: {time_str} — {len(items)} items")
        redeploy()
    else:
        print("Skipped update due to validation failure")

if __name__ == "__main__":
    main()
