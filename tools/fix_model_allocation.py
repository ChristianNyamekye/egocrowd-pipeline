"""
Fix model allocation for cron jobs.
Writes directly to jobs.json.
"""
import json, shutil, os

JOBS_PATH = os.path.expanduser("~/.openclaw/cron/jobs.json")

# Model allocation rules
SONNET = "anthropic/claude-sonnet-4-6"
OPUS   = "anthropic/claude-opus-4-6"

MODEL_MAP = {
    "heartbeat":              SONNET,
    "market-scanner":         SONNET,
    "dream-cycle":            SONNET,
    "morning-brief":          SONNET,
    "emily-email-reminder":   SONNET,
    "emily-email-reminder-2": SONNET,
    "daily-evolution":        OPUS,
}

# Backup
shutil.copy(JOBS_PATH, JOBS_PATH + ".pre-fix")

with open(JOBS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

changed = []
for job in data["jobs"]:
    name = job.get("name", "")
    if name in MODEL_MAP:
        old = job.get("model", "")
        job["model"] = MODEL_MAP[name]
        changed.append(f"  {name}: '{old}' -> '{MODEL_MAP[name]}'")

with open(JOBS_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Model allocation fixed:")
for c in changed:
    print(c)

# Verify
with open(JOBS_PATH, "r", encoding="utf-8") as f:
    verify = json.load(f)
print("\nVerification:")
for job in verify["jobs"]:
    print(f"  {job.get('name','?'):30s} -> {job.get('model','NONE')}")
