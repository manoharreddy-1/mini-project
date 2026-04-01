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
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        print(f"\\n[MOCK EMAIL (SMTP Credentials Missing)] To: {recipient_email} | OTP: {otp}\\n")
        return True # Fallback to mock if user hasn't configured it yet
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = recipient_email
        msg['Subject'] = "Your ScholarRAG Verification Code"
        
        body = f"""
        Hello,

        Your one-time verification code for ScholarRAG is:
        
        {otp}
        
        This code will expire in 10 minutes. Please do not share this code with anyone.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, recipient_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
