#!/usr/bin/env python3
"""YouTube video analyzer — pulls transcript, metadata, and description links."""

import sys
import re
import json
import os
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    if "youtu.be" in url:
        return urlparse(url).path.strip("/")
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return qs.get("v", [None])[0]

def extract_links(text):
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return list(set(re.findall(url_pattern, text)))

def get_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        snippets = list(transcript.snippets)
        
        full_text = ""
        timestamped = []
        for s in snippets:
            minutes = int(s.start // 60)
            seconds = int(s.start % 60)
            ts = f"[{minutes:02d}:{seconds:02d}]"
            timestamped.append(f"{ts} {s.text}")
            full_text += s.text + " "
        
        last = snippets[-1] if snippets else None
        duration = int(last.start + last.duration) if last else 0
        
        return {
            "full_text": full_text.strip(),
            "timestamped": timestamped,
            "duration_seconds": duration
        }
    except Exception as e:
        return {"error": str(e)}

def get_metadata(video_id):
    import subprocess
    try:
        # Find yt-dlp
        yt_dlp = "yt-dlp"
        scripts_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), 
            "Packages", "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0",
            "LocalCache", "local-packages", "Python313", "Scripts")
        if os.path.exists(os.path.join(scripts_dir, "yt-dlp.exe")):
            yt_dlp = os.path.join(scripts_dir, "yt-dlp.exe")
        
        result = subprocess.run(
            [yt_dlp, "--dump-json", "--no-download", f"https://youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30, encoding="utf-8"
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                "title": data.get("title", ""),
                "channel": data.get("channel", ""),
                "upload_date": data.get("upload_date", ""),
                "duration": data.get("duration", 0),
                "view_count": data.get("view_count", 0),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "categories": data.get("categories", []),
            }
        else:
            return {"error": result.stderr[:500]}
    except Exception as e:
        return {"error": str(e)}

def analyze(url):
    video_id = extract_video_id(url)
    if not video_id:
        print(json.dumps({"error": f"Could not extract video ID from: {url}"}))
        return

    meta = get_metadata(video_id)
    transcript = get_transcript(video_id)
    desc_links = extract_links(meta.get("description", "")) if "error" not in meta else []
    
    output = {
        "video_id": video_id,
        "url": url,
        "metadata": meta if "error" not in meta else None,
        "transcript": transcript if "error" not in transcript else None,
        "description_links": desc_links,
        "errors": []
    }
    
    if "error" in meta:
        output["errors"].append(f"metadata: {meta['error']}")
    if "error" in transcript:
        output["errors"].append(f"transcript: {transcript['error']}")
    
    if output["metadata"]:
        m = output["metadata"]
        print(f"{'='*60}")
        print(f"TITLE: {m['title']}")
        print(f"CHANNEL: {m['channel']}")
        print(f"DURATION: {m['duration']//60}m {m['duration']%60}s")
        print(f"VIEWS: {m['view_count']:,}")
        print(f"{'='*60}")
    
    if output["description_links"]:
        print(f"\nLINKS IN DESCRIPTION ({len(output['description_links'])}):")
        for link in output["description_links"]:
            print(f"  - {link}")
    
    if output["transcript"] and output["transcript"].get("full_text"):
        t = output["transcript"]["full_text"]
        print(f"\nTRANSCRIPT ({len(t)} chars):")
        # Print full transcript
        print(t)
    
    if output["errors"]:
        print(f"\nERRORS: {output['errors']}")
    
    json_path = os.path.join(os.path.dirname(__file__), "last_yt_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python youtube.py <youtube_url>")
        sys.exit(1)
    analyze(sys.argv[1])
