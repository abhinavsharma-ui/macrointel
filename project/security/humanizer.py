"""
Humanizer — YOLOv8 Person Detection Engine
==========================================
"Your camera feed becomes a smart security sensor."

Uses Ultralytics YOLOv8 (nano model, ~6MB) to detect humans in real time.
Feeds detections into the re-identification engine to track individuals.

Install deps:
    pip install ultralytics opencv-python-headless

Usage:
    from security.humanizer import PersonDetector
    detector = PersonDetector(camera_index=0)
    detector.start()          # runs in background thread
    detector.get_latest()     # get latest annotated frame + detections
    detector.stop()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Detection result dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One person detected in a single frame."""
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) in pixels
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    person_id: Optional[str] = None    # filled in by re-identification engine
    is_new: bool = True

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the cropped region of this detection from the frame."""
        x1, y1, x2, y2 = self.bbox
        h, w = frame.shape[:2]
        return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


@dataclass
class FrameResult:
    """Everything extracted from a single processed frame."""
    frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[Detection]
    timestamp: datetime = field(default_factory=datetime.now)
    fps: float = 0.0
    latency_ms: float = 0.0

    @property
    def person_count(self) -> int:
        return len(self.detections)

    @property
    def has_new_person(self) -> bool:
        return any(d.is_new for d in self.detections)


# ─────────────────────────────────────────────────────────────
# Model loader (lazy, cached)
# ─────────────────────────────────────────────────────────────

_model_cache: Dict[str, object] = {}


def _load_yolo(model_name: str = "yolov8n.pt"):
    """Load YOLOv8 model (downloads on first call, ~6 MB for nano)."""
    if model_name not in _model_cache:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model: {model_name} (first run downloads ~6 MB)")
            _model_cache[model_name] = YOLO(model_name)
            logger.info("YOLOv8 model loaded successfully")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
    return _model_cache[model_name]


# ─────────────────────────────────────────────────────────────
# Person Detector
# ─────────────────────────────────────────────────────────────

class PersonDetector:
    """
    Real-time person detector using YOLOv8.

    Runs a background thread that continuously reads from the camera,
    runs inference, and stores the latest annotated frame.

    Usage:
        detector = PersonDetector(camera_index=0, confidence_threshold=0.45)
        detector.start()

        # In your main loop:
        result = detector.get_latest()
        if result and result.person_count > 0:
            print(f"Detected {result.person_count} person(s)")

        detector.stop()
    """

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(
        self,
        camera_index: int = 0,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_fps: int = 15,
        on_detection: Optional[Callable[[FrameResult], None]] = None,
    ):
        self.camera_index = camera_index
        self.model_name   = model_name
        self.conf         = confidence_threshold
        self.iou          = iou_threshold
        self.target_fps   = target_fps
        self.on_detection = on_detection   # callback fired when person found

        self._model    = None
        self._cap      = None
        self._thread   = None
        self._running  = False
        self._lock     = threading.Lock()
        self._latest: Optional[FrameResult] = None

        # Stats
        self._frame_count    = 0
        self._detection_count= 0
        self._fps_history: List[float] = []

    def start(self) -> bool:
        """Start the detection loop in a background thread."""
        try:
            import cv2
            self._model = _load_yolo(self.model_name)
            self._cap   = cv2.VideoCapture(self.camera_index)

            if not self._cap.isOpened():
                logger.error(f"Cannot open camera {self.camera_index}")
                return False

            # Set camera resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            self._running = True
            self._thread  = threading.Thread(target=self._detection_loop, daemon=True)
            self._thread.start()

            logger.info(f"PersonDetector started (camera={self.camera_index}, conf={self.conf})")
            return True

        except Exception as e:
            logger.error(f"Failed to start PersonDetector: {e}")
            return False

    def stop(self):
        """Stop detection and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("PersonDetector stopped")

    def get_latest(self) -> Optional[FrameResult]:
        """Thread-safe access to the most recent frame result."""
        with self._lock:
            return self._latest

    def get_jpeg_bytes(self, quality: int = 80) -> Optional[bytes]:
        """Return the latest annotated frame as JPEG bytes (for streaming)."""
        try:
            import cv2
            result = self.get_latest()
            if result is None:
                return None
            _, buf = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return bytes(buf)
        except Exception:
            return None

    def _detection_loop(self):
        """Main inference loop. Runs in background thread."""
        import cv2

        frame_interval = 1.0 / self.target_fps
        last_frame_time = 0.0

        while self._running:
            loop_start = time.time()

            # Rate limit to target FPS
            elapsed = loop_start - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue
            last_frame_time = loop_start

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera read failed — retrying...")
                time.sleep(0.5)
                continue

            # Run YOLOv8 inference (persons only)
            t0 = time.time()
            try:
                results = self._model(
                    frame,
                    classes=[self.PERSON_CLASS_ID],
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"Inference error: {e}")
                continue

            inference_ms = (time.time() - t0) * 1000
            self._frame_count += 1

            # Parse detections
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                    ))
            if detections:
                self._detection_count += 1

            # Draw annotations
            annotated = self._annotate(frame.copy(), detections)

            # FPS tracking
            fps = 1.0 / max(time.time() - loop_start, 1e-6)
            self._fps_history.append(fps)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)

            result = FrameResult(
                frame=frame,
                annotated_frame=annotated,
                detections=detections,
                fps=float(np.mean(self._fps_history)),
                latency_ms=inference_ms,
            )

            with self._lock:
                self._latest = result

            # Fire callback
            if detections and self.on_detection:
                try:
                    self.on_detection(result)
                except Exception as e:
                    logger.debug(f"Detection callback error: {e}")

    def _annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        try:
            import cv2
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                color  = (0, 255, 0) if not det.is_new else (0, 80, 255)
                label  = det.person_id or f"Person ({det.confidence:.0%})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # HUD overlay
            avg_fps = float(np.mean(self._fps_history)) if self._fps_history else 0
            cv2.putText(frame, f"FPS: {avg_fps:.1f}  Persons: {len(detections)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception:
            pass
        return frame

    @property
    def stats(self) -> Dict:
        return {
            "frames_processed":    self._frame_count,
            "frames_with_persons": self._detection_count,
            "detection_rate_pct":  round(100 * self._detection_count / max(self._frame_count, 1), 1),
            "avg_fps":             round(float(np.mean(self._fps_history)), 2) if self._fps_history else 0,
            "is_running":          self._running,
        }


# ─────────────────────────────────────────────────────────────
# MJPEG stream generator (for Flask video feed)
# ─────────────────────────────────────────────────────────────

def mjpeg_generator(detector: PersonDetector, fps: int = 15):
    """
    Yields MJPEG frames for Flask streaming endpoint.

    Usage in Flask:
        @app.route("/video_feed")
        def video_feed():
            return Response(
                mjpeg_generator(detector),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )
    """
    interval = 1.0 / fps
    while True:
        t0 = time.time()
        data = detector.get_jpeg_bytes()
        if data:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            )
        elapsed = time.time() - t0
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)
