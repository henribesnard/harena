# sync_service/utils/alerts.py
import logging
import smtplib
from email.message import EmailMessage
from user_service.core.config import settings

logger = logging.getLogger(__name__)

def send_admin_alert(subject, message):
    """Envoyer une alerte par email aux administrateurs."""
    if not settings.ALERT_EMAIL_ENABLED:
        logger.warning(f"Alert email disabled: {subject}")
        return False
        
    try:
        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = f"[HARENA] {subject}"
        msg['From'] = settings.ALERT_EMAIL_FROM
        msg['To'] = settings.ALERT_EMAIL_TO
        
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            smtp.send_message(msg)
            
        return True
    except Exception as e:
        logger.error(f"Failed to send alert email: {str(e)}")
        return False

def alert_sync_failure(user_id, bridge_item_id, error_message):
    """Envoyer une alerte pour une erreur de synchronisation critique."""
    subject = f"Critical Sync Failure - User {user_id}"
    message = f"""
A critical synchronization failure occurred:

User ID: {user_id}
Bridge Item ID: {bridge_item_id}
Error: {error_message}

This requires immediate attention.
"""
    return send_admin_alert(subject, message)