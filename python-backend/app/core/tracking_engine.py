"""
Vessel Tracking Engine (CSRT-only)

ROI tracking for coronary vessel propagation across frames using CSRT tracker.

References:
- CSRT: Lukezic et al., CVPR 2017
"""

import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    raise ImportError("OpenCV is required for tracking")

logger = logging.getLogger(__name__)


def create_csrt_tracker():
    """
    Create CSRT tracker compatible with different OpenCV versions.

    OpenCV < 4.5.1: cv2.TrackerCSRT_create()
    OpenCV >= 4.5.1: cv2.legacy.TrackerCSRT_create()
    OpenCV >= 4.8.0: cv2.legacy.TrackerCSRT.create()
    """
    # Try new API (OpenCV >= 4.5.1)
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
    except Exception:
        pass

    # Try newer API (OpenCV >= 4.8.0)
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT'):
            return cv2.legacy.TrackerCSRT.create()
    except Exception:
        pass

    # Try old API (OpenCV < 4.5.1)
    try:
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
    except Exception:
        pass

    raise RuntimeError(
        f"CSRT tracker not available in OpenCV {cv2.__version__}. "
        "Install opencv-contrib-python: pip install opencv-contrib-python"
    )


@dataclass
class TrackingResult:
    """Result of a tracking operation."""
    success: bool
    roi: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    confidence: float
    method: str = "csrt"
    error_message: Optional[str] = None


@dataclass
class TrackingState:
    """Internal state for tracking across frames."""
    csrt_tracker: Optional[Any] = None
    previous_frame: Optional[np.ndarray] = None  # Grayscale uint8
    previous_roi: Optional[Tuple[int, int, int, int]] = None
    initial_roi: Optional[Tuple[int, int, int, int]] = None
    confidence_history: List[float] = field(default_factory=list)


class TrackingEngine:
    """
    CSRT-based tracking engine for vessel ROI propagation.

    Uses OpenCV's CSRT tracker for robust ROI tracking across frames.

    Usage:
        engine = TrackingEngine()
        engine.initialize(frame, roi)
        result = engine.track(next_frame)

        if result.success:
            new_roi = result.roi
    """

    # Constants
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for success
    CONFIDENCE_WINDOW = 5  # Rolling window for confidence check

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        fixed_roi_size: bool = True
    ):
        """
        Initialize tracking engine.

        Args:
            confidence_threshold: Minimum confidence to continue tracking
            fixed_roi_size: If True, keep ROI size constant (only track position)
        """
        self.confidence_threshold = confidence_threshold
        self.fixed_roi_size = fixed_roi_size
        self._state = TrackingState()

    @property
    def is_initialized(self) -> bool:
        """Check if tracker is initialized."""
        return self._state.csrt_tracker is not None

    def initialize(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> bool:
        """
        Initialize CSRT tracker with initial frame and ROI.

        Args:
            frame: Initial frame (grayscale or BGR)
            roi: ROI bounding box (x, y, w, h)

        Returns:
            True if initialization successful
        """
        try:
            self.reset()

            # Prepare frame
            frame_gray = self._ensure_grayscale_uint8(frame)
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

            # Validate ROI
            frame_h, frame_w = frame_gray.shape[:2]
            x, y, w, h = roi
            if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                logger.error(f"ROI out of bounds: {roi}, frame: {frame_w}x{frame_h}")
                return False

            if w <= 0 or h <= 0:
                logger.error(f"Invalid ROI dimensions: {roi}")
                return False

            # Initialize CSRT tracker
            self._state.csrt_tracker = create_csrt_tracker()
            success = self._state.csrt_tracker.init(frame_bgr, roi)

            if not success:
                logger.error("CSRT tracker initialization failed")
                return False

            # Store state
            self._state.previous_frame = frame_gray
            self._state.previous_roi = roi
            self._state.initial_roi = roi
            self._state.confidence_history = [1.0]  # Initial confidence

            logger.info(f"CSRT tracking initialized: ROI={roi}")
            return True

        except Exception as e:
            logger.error(f"Tracking initialization failed: {e}")
            return False

    def track(self, frame: np.ndarray) -> TrackingResult:
        """
        Track ROI to next frame using CSRT.

        Args:
            frame: Current frame (grayscale or BGR)

        Returns:
            TrackingResult with tracked ROI
        """
        if not self.is_initialized:
            return TrackingResult(
                success=False, roi=None,
                confidence=0.0, method="csrt",
                error_message="Tracker not initialized"
            )

        try:
            frame_gray = self._ensure_grayscale_uint8(frame)
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

            # CSRT tracking
            success, bbox = self._state.csrt_tracker.update(frame_bgr)

            if not success:
                return TrackingResult(
                    success=False, roi=self._state.previous_roi,
                    confidence=0.0, method="csrt",
                    error_message="CSRT tracking failed"
                )

            new_x, new_y, new_w, new_h = map(int, bbox)

            # Keep original size if fixed_roi_size
            if self.fixed_roi_size and self._state.initial_roi:
                _, _, orig_w, orig_h = self._state.initial_roi
                roi = (new_x, new_y, orig_w, orig_h)
            else:
                roi = (new_x, new_y, new_w, new_h)

            # Calculate confidence based on CSRT internal metrics
            # CSRT doesn't expose confidence directly, so we estimate it
            confidence = self._estimate_confidence(frame_gray, roi)

            # Update state
            self._state.previous_frame = frame_gray
            self._state.previous_roi = roi
            self._state.confidence_history.append(confidence)

            # Keep only recent history
            if len(self._state.confidence_history) > 20:
                self._state.confidence_history = self._state.confidence_history[-20:]

            logger.debug(f"CSRT tracked: ROI={roi}, confidence={confidence:.3f}")

            return TrackingResult(
                success=True, roi=roi,
                confidence=confidence, method="csrt"
            )

        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return TrackingResult(
                success=False, roi=self._state.previous_roi,
                confidence=0.0, method="csrt",
                error_message=str(e)
            )

    def _estimate_confidence(
        self,
        frame_gray: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> float:
        """
        Estimate tracking confidence based on ROI similarity.

        Compares current ROI region with initial ROI region using
        normalized cross-correlation.
        """
        try:
            if self._state.previous_frame is None:
                return 1.0

            x, y, w, h = roi
            frame_h, frame_w = frame_gray.shape[:2]

            # Clamp ROI to frame bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            # Extract current ROI region
            current_roi = frame_gray[y1:y2, x1:x2]

            # Extract previous ROI region
            px, py, pw, ph = self._state.previous_roi
            px1 = max(0, px)
            py1 = max(0, py)
            px2 = min(frame_w, px + pw)
            py2 = min(frame_h, py + ph)

            prev_roi = self._state.previous_frame[py1:py2, px1:px2]

            # Resize to same size if needed
            if current_roi.shape != prev_roi.shape:
                prev_roi = cv2.resize(prev_roi, (current_roi.shape[1], current_roi.shape[0]))

            # Calculate normalized cross-correlation
            if current_roi.size == 0 or prev_roi.size == 0:
                return 0.5

            # Use template matching score as confidence
            result = cv2.matchTemplate(current_roi, prev_roi, cv2.TM_CCOEFF_NORMED)
            confidence = float(np.max(result))

            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            return confidence

        except Exception as e:
            logger.debug(f"Confidence estimation failed: {e}")
            return 0.5

    def should_stop(self, window_size: int = None) -> bool:
        """
        Check if tracking should stop based on confidence history.

        Args:
            window_size: Number of recent frames to consider

        Returns:
            True if tracking should stop
        """
        window_size = window_size or self.CONFIDENCE_WINDOW

        if len(self._state.confidence_history) < window_size:
            return False

        recent = self._state.confidence_history[-window_size:]
        avg_confidence = np.mean(recent)

        should_stop = avg_confidence < self.confidence_threshold

        if should_stop:
            logger.warning(f"Tracking stopped: avg_confidence={avg_confidence:.3f}")

        return should_stop

    def _ensure_grayscale_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale uint8."""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return frame

    def reset(self):
        """Reset tracking state."""
        self._state = TrackingState()
        logger.info("Tracking engine reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current tracking state for debugging."""
        return {
            "is_initialized": self.is_initialized,
            "previous_roi": self._state.previous_roi,
            "initial_roi": self._state.initial_roi,
            "confidence_history": self._state.confidence_history[-10:],
            "avg_confidence": np.mean(self._state.confidence_history[-5:]) if self._state.confidence_history else 0.0
        }


# Singleton instance
_engine_instance: Optional[TrackingEngine] = None


def get_engine() -> TrackingEngine:
    """Get singleton TrackingEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TrackingEngine()
    return _engine_instance
