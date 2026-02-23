import sys
from youtube_transcript_api import YouTubeTranscriptApi
vid = sys.argv[1] if len(sys.argv) > 1 else "NZ1mKAWJPr4"
try:
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(vid)
    for e in transcript.snippets:
        m, s = divmod(int(e.start), 60)
        print(f"[{m}:{s:02d}] {e.text}")
except Exception as ex:
    print(f"Error: {ex}")
    import youtube_transcript_api
    print(dir(youtube_transcript_api.YouTubeTranscriptApi))
