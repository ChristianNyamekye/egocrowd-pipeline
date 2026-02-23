"""
Discord Config Watchdog
Checks if Discord config is present in openclaw.json.
If missing, restores it from the backup values.
Run periodically or after any gateway restart.

Usage: python discord_watchdog.py [--fix]
"""
import json, sys, os

CONFIG_PATH = os.path.expanduser("~/.openclaw/openclaw.json")

def _load_env_key(key):
    """Load a key from ~/.openclaw/.env safely."""
    env_path = os.path.expanduser("~/.openclaw/.env")
    if not os.path.exists(env_path):
        return ""
    with open(env_path) as f:
        for line in f:
            if line.strip().startswith(f"{key}="):
                return line.strip().split("=", 1)[1]
    return ""

DISCORD_CONFIG = {
    "enabled": True,
    "token": _load_env_key("DISCORD_BOT_TOKEN"),
    "dmPolicy": "pairing",
    "groupPolicy": "allowlist",
    "guilds": {
        "1474786420756447372": {
            "requireMention": False,
            "users": ["1045490246550302800"]
        }
    },
    "streaming": "off"
}

def check_config(fix=False):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    issues = []

    # Check Discord channel exists
    channels = config.get("channels", {})
    discord = channels.get("discord", {})

    if not discord.get("enabled"):
        issues.append("MISSING: Discord channel not enabled")
        if fix:
            config.setdefault("channels", {})["discord"] = DISCORD_CONFIG
            print("FIXED: Restored full Discord config")

    if not discord.get("token"):
        issues.append("MISSING: Discord bot token")
        if fix:
            config["channels"]["discord"]["token"] = _load_env_key("DISCORD_BOT_TOKEN")
            print("FIXED: Restored Discord bot token from .env")

    if not discord.get("guilds"):
        issues.append("MISSING: Discord guilds config")
        if fix:
            config["channels"]["discord"]["guilds"] = DISCORD_CONFIG["guilds"]
            print("FIXED: Restored Discord guilds config")

    # Check compaction settings
    compaction = config.get("agents", {}).get("defaults", {}).get("compaction", {})
    if compaction.get("reserveTokensFloor") != 8000:
        issues.append(f"DRIFT: compaction.reserveTokensFloor (current: {compaction.get('reserveTokensFloor')})")
        if fix:
            config["agents"]["defaults"]["compaction"]["reserveTokensFloor"] = 8000
            print("FIXED: reserveTokensFloor set to 8000")

    if not issues:
        print("OK: All checks passed")
    else:
        for issue in issues:
            print(issue)

    if fix and issues:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig written. {len(issues)} issue(s) fixed.")

    return len(issues) == 0

if __name__ == "__main__":
    fix = "--fix" in sys.argv
    ok = check_config(fix=fix)
    sys.exit(0 if ok else 1)
