"""
Ngrok Tunnel Manager
=====================
Exposes your local Flask dashboard (including camera feed) to a
secure public URL so you can check your home from anywhere.

Install:
    pip install pyngrok
    ngrok authtoken YOUR_TOKEN   (free at ngrok.com)

Usage:
    from security.ngrok_tunnel import NgrokTunnel

    tunnel = NgrokTunnel(port=5050)
    public_url = tunnel.start()
    print(f"Dashboard: {public_url}")
    print(f"Camera:    {public_url}/video_feed")

    # Later:
    tunnel.stop()

Without an authtoken the free plan gives you a random URL each
session (valid 2 hours). With a free account authtoken you get a
stable URL and 8h sessions.
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")


class NgrokTunnel:
    """
    Manages a pyngrok tunnel to a local port.

    Attributes:
        public_url   — The public https:// URL after start()
        camera_url   — Convenience URL for the video feed endpoint
    """

    def __init__(
        self,
        port: int = 5050,
        region: str = "us",
        auth_token: str = NGROK_AUTH_TOKEN,
    ):
        self.port       = port
        self.region     = region
        self.auth_token = auth_token
        self.public_url: Optional[str] = None
        self._tunnel    = None

    def start(self) -> Optional[str]:
        """
        Open the ngrok tunnel and return the public HTTPS URL.
        Returns None if pyngrok is not installed or tunnel fails.
        """
        try:
            from pyngrok import ngrok, conf

            if self.auth_token:
                conf.get_default().auth_token = self.auth_token
                logger.info("Ngrok auth token configured")

            # Open HTTPS tunnel
            self._tunnel   = ngrok.connect(self.port, "http")
            self.public_url= self._tunnel.public_url

            # Upgrade http → https if ngrok gave http
            if self.public_url and self.public_url.startswith("http://"):
                self.public_url = self.public_url.replace("http://", "https://", 1)

            logger.info(f"Ngrok tunnel OPEN: {self.public_url}")
            logger.info(f"  Dashboard : {self.public_url}/")
            logger.info(f"  Camera    : {self.public_url}/video_feed")
            logger.info(f"  API       : {self.public_url}/api/signals")

            return self.public_url

        except ImportError:
            logger.warning(
                "pyngrok not installed — no remote access. "
                "Run: pip install pyngrok"
            )
            return None
        except Exception as e:
            logger.error(f"Ngrok tunnel failed to open: {e}")
            return None

    def stop(self):
        """Close the tunnel and free the public URL."""
        try:
            if self._tunnel:
                from pyngrok import ngrok
                ngrok.disconnect(self._tunnel.public_url)
                self._tunnel    = None
                self.public_url = None
                logger.info("Ngrok tunnel closed")
        except Exception as e:
            logger.debug(f"Ngrok stop error: {e}")

    @property
    def camera_url(self) -> Optional[str]:
        if self.public_url:
            return f"{self.public_url}/video_feed"
        return None

    @property
    def is_active(self) -> bool:
        return self._tunnel is not None and self.public_url is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ─────────────────────────────────────────────────────────────
# Simple helper for run.py
# ─────────────────────────────────────────────────────────────

_global_tunnel: Optional[NgrokTunnel] = None


def start_tunnel(port: int = 5050) -> Optional[str]:
    """
    Start a global ngrok tunnel and return the public URL.
    Safe to call multiple times — returns existing URL if already running.
    """
    global _global_tunnel

    if _global_tunnel and _global_tunnel.is_active:
        return _global_tunnel.public_url

    _global_tunnel = NgrokTunnel(port=port)
    return _global_tunnel.start()


def get_public_url() -> Optional[str]:
    """Return the current public URL, or None if no tunnel is running."""
    if _global_tunnel:
        return _global_tunnel.public_url
    return None


def stop_tunnel():
    """Stop the global tunnel."""
    global _global_tunnel
    if _global_tunnel:
        _global_tunnel.stop()
        _global_tunnel = None
