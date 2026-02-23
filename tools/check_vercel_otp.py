import imaplib, email, re, time

for attempt in range(5):
    m = imaplib.IMAP4_SSL('imap.gmail.com')
    m.login('autonomousnyamekye@gmail.com', 'cetgmwrksfknhzkh')
    m.select('INBOX')
    status, data = m.search(None, 'FROM', '"vercel"')
    if data[0]:
        ids = data[0].split()
        for eid in ids[-3:]:
            status, msg_data = m.fetch(eid, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])
            print(f"Subject: {msg['Subject']}")
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True).decode(errors='replace')
                    # Look for OTP code
                    codes = re.findall(r'\b\d{6}\b', body)
                    if codes:
                        print(f"OTP CODE: {codes[0]}")
                    print(body[:300])
    else:
        print(f"Attempt {attempt+1}: No Vercel email yet, waiting 10s...")
    m.logout()
    if data[0]:
        break
    time.sleep(10)
