import imaplib, email

m = imaplib.IMAP4_SSL('imap.gmail.com')
m.login('autonomousnyamekye@gmail.com', 'cetgmwrksfknhzkh')
m.select('INBOX')
status, data = m.search(None, 'FROM', '"mailer-daemon"')
if data[0]:
    ids = data[0].split()
    for eid in ids[-5:]:
        status, msg_data = m.fetch(eid, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        print(f"Subject: {msg['Subject']}")
        print(f"Date: {msg['Date']}")
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True).decode(errors='replace')
                print(body[:500])
        print("===")
m.logout()
