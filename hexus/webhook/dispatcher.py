import hashlib
import hmac
import json
import logging
import time
import threading
import requests
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for the request payload."""
    return hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

def dispatch_webhook_sync(
    url: str,
    secret: Optional[str],
    event: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    initial_backoff: float = 1.0,
) -> None:
    """Send webhook payload to the url, retrying with exponential backoff on failure."""
    full_payload = {
        "event": event,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": payload
    }
    
    try:
        body = json.dumps(full_payload)
    except Exception as exc:
        logger.error("Failed to serialize webhook payload for event %s: %s", event, exc)
        return

    body_bytes = body.encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Hexus-Event": event,
    }
    
    if secret:
        headers["X-Hexus-Signature"] = sign_payload(body_bytes, secret)

    backoff = initial_backoff
    for attempt in range(max_retries + 1):
        try:
            logger.debug("Dispatching webhook event %s to %s (attempt %d/%d)", event, url, attempt + 1, max_retries + 1)
            response = requests.post(url, data=body_bytes, headers=headers, timeout=5.0)
            if response.status_code >= 200 and response.status_code < 300:
                logger.debug("Webhook event %s successfully dispatched to %s", event, url)
                return
            else:
                logger.warning(
                    "Webhook response failure for event %s (status=%d): %s",
                    event, response.status_code, response.text[:200]
                )
        except requests.RequestException as exc:
            logger.warning("Webhook dispatch error for event %s on attempt %d: %s", event, attempt + 1, exc)
        
        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2

    logger.error("Failed to dispatch webhook event %s to %s after %d retries", event, url, max_retries)

def dispatch_webhook(
    url: Optional[str],
    secret: Optional[str],
    event: str,
    payload: Dict[str, Any],
) -> None:
    """Asynchronously dispatch webhook in a background daemon thread."""
    if not url:
        return
        
    thread = threading.Thread(
        target=dispatch_webhook_sync,
        args=(url, secret, event, payload),
        name=f"hexus-webhook-{event}",
        daemon=True,
    )
    thread.start()
