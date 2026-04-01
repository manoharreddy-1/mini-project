import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")


def send_real_otp_email(recipient_email, otp):
    """Send an OTP to the user's email address.

    Supports:
      - Brevo API (preferred when BREVO_API_KEY is set)
      - SMTP (when SMTP_USERNAME/SMTP_PASSWORD are set)
      - Mock (falls back if neither is configured)
    """
    subject = "Your ScholarRAG Verification Code"
    body_text = f"""
    Hello,

    Your one-time verification code for ScholarRAG is:

    {otp}

    This code will expire in 10 minutes. Please do not share this code with anyone.
    """

    # Brevo API (recommended)
    if BREVO_API_KEY:
        try:
            payload = {
                "sender": {"name": "ScholarRAG", "email": SMTP_USERNAME or "noreply@scholarrag.com"},
                "to": [{"email": recipient_email}],
                "subject": subject,
                "textContent": body_text,
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "api-key": BREVO_API_KEY,
            }
            resp = requests.post(
                "https://api.brevo.com/v3/smtp/email",
                json=payload,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send email via Brevo API: {e}")
            # Fall back to SMTP if configured

    # SMTP fallback (includes Brevo SMTP relay, Gmail, etc.)
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        print(f"\n[MOCK EMAIL (SMTP Credentials Missing)] To: {recipient_email} | OTP: {otp}\n")
        return True  # Fallback to mock if user hasn't configured it yet

    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body_text, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, recipient_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email via SMTP: {e}")
        return False
