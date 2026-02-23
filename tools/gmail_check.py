"""Check Gmail inbox for recent replies via IMAP."""
import imaplib, email, sys
from email.header import decode_header

USER = "autonomousnyamekye@gmail.com"
PASS = "cetgmwrksfknhzkh"

def check(n=10):
    m = imaplib.IMAP4_SSL("imap.gmail.com")
    m.login(USER, PASS)
    m.select("INBOX")
    _, data = m.search(None, "ALL")
    ids = data[0].split()[-n:]
    for mid in reversed(ids):
        _, msg_data = m.fetch(mid, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        subj = decode_header(msg["Subject"])[0]
        subj = subj[0].decode(subj[1] or "utf-8") if isinstance(subj[0], bytes) else subj[0]
        fr = msg["From"]
        date = msg["Date"]
        print(f"From: {fr}")
        print(f"Date: {date}")
        print(f"Subject: {subj}")
        print("---")
    m.logout()

if __name__ == "__main__":
    check(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
