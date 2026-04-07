"""
Person Re-Identification Engine
=================================
"Remember if a person is a New Guest or someone seen 10 minutes ago."

Algorithm:
  1. For each detected person crop, compute a 3D color histogram (H-S-V)
  2. Compare against a gallery of previously seen persons using
     Bhattacharyya distance (robust to lighting changes)
  3. If distance < threshold → same person (re-ID)
  4. If distance > threshold → new person → add to gallery
  5. Persons not seen for `expiry_seconds` are removed from gallery

Why color histograms (not deep re-ID)?
  - Zero GPU requirement — runs on any Raspberry Pi or laptop
  - Extremely fast (~0.3ms per comparison vs ~15ms for deep models)
  - Good enough for home/office security where outfit stays consistent
  - No privacy-invasive face recognition

Upgrade path: swap `_compute_signature` for a deep re-ID model
(e.g. OSNet from torchreid) by returning a 512-d embedding instead.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Tracked person record
# ─────────────────────────────────────────────────────────────

@dataclass
class TrackedPerson:
    person_id: str
    signature: np.ndarray       # color histogram vector
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen:  datetime = field(default_factory=datetime.now)
    sighting_count: int = 1
    label: str = ""             # optional custom label ("Owner", "Intruder")

    @property
    def seconds_since_seen(self) -> float:
        return (datetime.now() - self.last_seen).total_seconds()

    @property
    def is_familiar(self) -> bool:
        return self.sighting_count >= 3

    def update(self, new_sig: np.ndarray):
        """Running average of signature for drift adaptation."""
        alpha = 0.85   # weight of historical signature
        self.signature    = alpha * self.signature + (1 - alpha) * new_sig
        self.last_seen    = datetime.now()
        self.sighting_count += 1


# ─────────────────────────────────────────────────────────────
# Signature extractor
# ─────────────────────────────────────────────────────────────

def _compute_signature(crop: np.ndarray, bins: int = 16) -> Optional[np.ndarray]:
    """
    Compute a normalized 3D HSV color histogram for a person crop.

    Returns a flat 1-D array of length bins^3 (4096 for bins=16).
    None if crop is too small to be reliable.
    """
    try:
        import cv2

        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return None

        # Focus on torso (ignore top 20% head, bottom 20% feet for stability)
        y1 = int(h * 0.20)
        y2 = int(h * 0.80)
        torso = crop[y1:y2, :]

        if torso.size == 0:
            return None

        # Convert to HSV (more stable than RGB under lighting changes)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        # 3D histogram — Hue (bins), Saturation (bins), Value (bins//2)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [bins, bins, bins // 2],
            [0, 180, 0, 256, 0, 256],
        )

        # Normalize to [0,1] so size of person doesn't matter
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().astype(np.float32)

    except ImportError:
        raise ImportError("opencv-python not installed. Run: pip install opencv-python-headless")
    except Exception as e:
        logger.debug(f"Signature extraction failed: {e}")
        return None


def _bhattacharyya_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Bhattacharyya distance between two histogram signatures.
    Range: 0 (identical) to 1 (completely different).
    More robust to lighting than correlation or L2.
    """
    try:
        import cv2
        dist = cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_BHATTACHARYYA)
        return float(dist)
    except Exception:
        # Fallback: manual computation
        bc = float(np.sum(np.sqrt(sig_a * sig_b + 1e-9)))
        return float(-np.log(bc + 1e-9))


# ─────────────────────────────────────────────────────────────
# Re-ID gallery
# ─────────────────────────────────────────────────────────────

class ReIdentificationGallery:
    """
    Manages a gallery of known persons and identifies new ones.

    Usage:
        gallery = ReIdentificationGallery()

        # For each detected person crop:
        person_id, is_new = gallery.identify(crop_image)
        # person_id is a stable UUID or custom label
        # is_new = True first time this person is seen
    """

    def __init__(
        self,
        distance_threshold: float = 0.45,
        expiry_seconds: float = 600,      # 10 minutes
        max_gallery_size: int = 50,
        histogram_bins: int = 16,
    ):
        self.threshold       = distance_threshold
        self.expiry_seconds  = expiry_seconds
        self.max_size        = max_gallery_size
        self.bins            = histogram_bins

        self._gallery: Dict[str, TrackedPerson] = {}
        self._id_counter = 0

    def identify(self, crop: np.ndarray) -> Tuple[str, bool]:
        """
        Identify a person from their crop image.
        Returns (person_id, is_new).
        """
        self._evict_expired()

        sig = _compute_signature(crop, self.bins)
        if sig is None:
            pid = f"unknown-{uuid.uuid4().hex[:6]}"
            return pid, True

        # Compare against all gallery entries
        best_pid      = None
        best_distance = float("inf")

        for pid, tracked in self._gallery.items():
            dist = _bhattacharyya_distance(sig, tracked.signature)
            if dist < best_distance:
                best_distance = dist
                best_pid      = pid

        if best_pid and best_distance < self.threshold:
            # Re-identified — update signature with running average
            self._gallery[best_pid].update(sig)
            familiar = self._gallery[best_pid].is_familiar
            logger.debug(f"Re-ID: {best_pid} (dist={best_distance:.3f}, familiar={familiar})")
            return best_pid, False

        # New person — add to gallery
        new_id = self._new_id()
        person = TrackedPerson(
            person_id=new_id,
            signature=sig,
        )
        self._gallery[new_id] = person
        logger.info(f"New person added to gallery: {new_id} (gallery_size={len(self._gallery)})")
        return new_id, True

    def label_person(self, person_id: str, label: str):
        """Assign a human-readable label to a tracked person (e.g. 'Owner')."""
        if person_id in self._gallery:
            self._gallery[person_id].label = label

    def get_all(self) -> List[TrackedPerson]:
        """Return all currently tracked persons."""
        self._evict_expired()
        return list(self._gallery.values())

    def _new_id(self) -> str:
        self._id_counter += 1
        return f"person-{self._id_counter:04d}"

    def _evict_expired(self):
        """Remove persons not seen for longer than expiry_seconds."""
        expired = [
            pid for pid, p in self._gallery.items()
            if p.seconds_since_seen > self.expiry_seconds
        ]
        for pid in expired:
            logger.debug(f"Evicted {pid} from gallery (not seen for {self.expiry_seconds}s)")
            del self._gallery[pid]

        # Hard cap on gallery size
        if len(self._gallery) > self.max_size:
            # Remove least recently seen
            oldest = sorted(self._gallery.items(), key=lambda x: x[1].last_seen)
            for pid, _ in oldest[:len(self._gallery) - self.max_size]:
                del self._gallery[pid]

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    @property
    def stats(self) -> Dict:
        persons = list(self._gallery.values())
        return {
            "gallery_size":      len(persons),
            "familiar_persons":  sum(1 for p in persons if p.is_familiar),
            "new_persons_today": sum(1 for p in persons if
                                     (datetime.now() - p.first_seen).seconds < 86400),
        }


# ─────────────────────────────────────────────────────────────
# Full tracking engine (combines detector + gallery)
# ─────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Orchestrates detection + re-identification.

    Usage:
        from security.humanizer    import PersonDetector
        from security.reidentification import PersonTracker

        detector = PersonDetector(camera_index=0)
        tracker  = PersonTracker(detector)
        tracker.start()
    """

    def __init__(
        self,
        detector,                          # PersonDetector instance
        on_new_person: Optional[callable] = None,
        on_reidentified: Optional[callable] = None,
        gallery_kwargs: Optional[dict] = None,
    ):
        self.detector        = detector
        self.gallery         = ReIdentificationGallery(**(gallery_kwargs or {}))
        self.on_new_person   = on_new_person
        self.on_reidentified = on_reidentified

        # Hook into detector's callback
        self.detector.on_detection = self._process_frame

    def start(self):
        """Start detector."""
        return self.detector.start()

    def stop(self):
        """Stop detector."""
        self.detector.stop()

    def _process_frame(self, frame_result):
        """Called by detector for each frame with persons detected."""
        from security.humanizer import Detection

        for det in frame_result.detections:
            try:
                crop = det.crop(frame_result.frame)
                if crop.size == 0:
                    continue

                person_id, is_new = self.gallery.identify(crop)
                det.person_id = person_id
                det.is_new    = is_new

                if is_new and self.on_new_person:
                    self.on_new_person(det, frame_result)
                elif not is_new and self.on_reidentified:
                    self.on_reidentified(det, frame_result)

            except Exception as e:
                logger.error(f"Tracking error: {e}")
