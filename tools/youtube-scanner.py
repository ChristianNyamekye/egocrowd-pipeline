"""
YouTube Channel Scanner
Checks monitored channels for recent uploads related to robotics/AI topics.
Designed to be run via OpenClaw tools (web_fetch/web_search).
This is a reference script — actual scanning uses web_fetch directly.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

CHANNELS_FILE = Path(__file__).parent / "youtube-channels.json"
KEYWORDS = [
    "robotics data", "manipulation learning", "imitation learning",
    "teleoperation", "humanoid robot", "dexterous hand", "openclaw",
    "robot learning", "embodied ai", "diffusion policy", "action chunking",
    "visuomotor", "bimanual", "grasping", "sim-to-real", "foundation model robot"
]

def load_channels():
    with open(CHANNELS_FILE) as f:
        return json.load(f)["channels"]

def channel_url(name):
    """Best-effort URL for a channel's videos page."""
    handle = name.replace(" ", "")
    return f"https://www.youtube.com/@{handle}/videos"

if __name__ == "__main__":
    channels = load_channels()
    for ch in channels:
        print(f"Channel: {ch['name']}")
        print(f"  URL: {channel_url(ch['name'])}")
        print(f"  Topics: {', '.join(ch['topics'])}")
