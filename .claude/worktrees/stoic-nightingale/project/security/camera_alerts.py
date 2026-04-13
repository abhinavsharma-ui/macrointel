"""
Camera Alert System
===================
Provides camera detection and video feed support for the dashboard.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SecuritySuite:
    """
    Complete security suite: detector, re-identification, and video stream.
    """

    def __init__(
        self,
        camera_index: int = 0,
        dashboard_port: int = 5050,
        alert_cooldown_sec: int = 60,
        min_confidence: float = 0.45,
    ):
        self.camera_index = camera_index
        self.dashboard_port = dashboard_port
        self.alert_cooldown = alert_cooldown_sec
        self.min_confidence = min_confidence

        self._detector = None
        self._tracker = None
        self._alerter = None
        self._tunnel = None
        self._running = False

        self._alert_log: list = []
        self._last_alert_time: Dict[str, float] = {}

    def start(self, enable_ngrok: bool = True) -> Dict:
        """
        Start all security components.
        Returns a dict with status and public URL if ngrok is active.
        """
        result = {"camera": False, "ngrok_url": None}

        try:
            from security.humanizer import PersonDetector
            from security.reidentification import PersonTracker

            self._detector = PersonDetector(
                camera_index=self.camera_index,
                confidence_threshold=self.min_confidence,
                target_fps=15,
            )
            self._tracker = PersonTracker(
                detector=self._detector,
                on_new_person=self._handle_new_person,
                on_reidentified=self._handle_reidentified,
            )

            ok = self._tracker.start()
            result["camera"] = ok

            if ok:
                logger.info(f"Camera started (index={self.camera_index})")
            else:
                logger.warning(f"Camera {self.camera_index} failed to start")

        except ImportError as exc:
            logger.error(f"Security dependencies missing: {exc}")
            logger.error("Install: pip install ultralytics opencv-python-headless")

        if enable_ngrok:
            try:
                from security.ngrok_tunnel import NgrokTunnel

                self._tunnel = NgrokTunnel(port=self.dashboard_port)
                public_url = self._tunnel.start()
                result["ngrok_url"] = public_url
            except Exception as exc:
                logger.warning(f"Ngrok start failed: {exc}")

        self._running = True
        return result

    def stop(self):
        """Stop all security components."""
        self._running = False
        if self._tracker:
            self._tracker.stop()
        if self._tunnel:
            self._tunnel.stop()
        logger.info("SecuritySuite stopped")

    def _handle_new_person(self, detection, frame_result):
        """Callback: new unknown person detected."""
        person_id = detection.person_id or "unknown"
        now = time.time()
        if now - self._last_alert_time.get(person_id, 0) < self.alert_cooldown:
            return
        self._last_alert_time[person_id] = now

        alert = {
            "type": "new_person",
            "person_id": person_id,
            "confidence": detection.confidence,
            "timestamp": datetime.now().isoformat(),
        }
        self._alert_log.append(alert)
        logger.warning(f"NEW PERSON DETECTED: {person_id} (conf={detection.confidence:.0%})")

    def _handle_reidentified(self, detection, frame_result):
        """Callback: known person re-identified."""
        logger.debug(f"Re-identified: {detection.person_id}")

    def get_jpeg_frame(self) -> Optional[bytes]:
        """Get latest annotated camera frame as JPEG bytes."""
        if self._detector:
            return self._detector.get_jpeg_bytes()
        return None

    def get_status(self) -> Dict:
        """Return current status of all components."""
        return {
            "running": self._running,
            "camera_active": self._tracker is not None and self._tracker.is_running if self._tracker else False,
            "ngrok_url": getattr(self._tunnel, "public_url", None),
            "alerts_logged": len(self._alert_log),
            "camera_index": self.camera_index,
        }
