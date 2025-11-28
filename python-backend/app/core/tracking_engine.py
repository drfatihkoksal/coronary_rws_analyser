"""
Vessel Tracking Engine

ROI tracking for coronary vessel propagation across frames.
Uses best available OpenCV tracker (TrackerVit, TrackerNano, or CSRT).

References:
- CSRT: Lukezic et al., CVPR 2017
- ViT Tracker: OpenCV DNN-based Visual Transformer tracker
"""

import os
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    raise ImportError("OpenCV is required for tracking")


# Model paths for DNN-based trackers
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "trackers")
_VIT_MODEL = os.path.join(_MODEL_DIR, "vitTracker.onnx")
_NANO_BACKBONE = os.path.join(_MODEL_DIR, "nanotrack_backbone.onnx")
_NANO_HEAD = os.path.join(_MODEL_DIR, "nanotrack_head.onnx")


def create_tracker():
    """
    Create object tracker compatible with different OpenCV versions.

    Priority:
    1. TrackerMIL (most reliable for medical images)
    2. TrackerVit (optional, less reliable for low-contrast images)
    """
    # Use TrackerMIL (most reliable for medical images)
    try:
        if hasattr(cv2, 'TrackerMIL') and hasattr(cv2.TrackerMIL, 'create'):
            tracker = cv2.TrackerMIL.create()
            logger.info("Using TrackerMIL")
            return tracker, "mil"
    except Exception as e:
        logger.debug(f"TrackerMIL creation failed: {e}")

    # Try TrackerNano (lightweight, requires OpenCV >= 4.7)
    if hasattr(cv2, 'TrackerNano') and hasattr(cv2.TrackerNano, 'create'):
        if os.path.exists(_NANO_BACKBONE) and os.path.exists(_NANO_HEAD):
            try:
                params = cv2.TrackerNano_Params()
                params.backbone = _NANO_BACKBONE
                params.neckhead = _NANO_HEAD
                tracker = cv2.TrackerNano.create(params)
                logger.info(f"Using TrackerNano")
                return tracker, "nano"
            except Exception as e:
                logger.debug(f"TrackerNano creation failed: {e}")
        else:
            logger.debug(f"TrackerNano models not found")

    # Try CSRT (legacy, OpenCV < 4.12)
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            tracker = cv2.legacy.TrackerCSRT_create()
            logger.info("Using TrackerCSRT (legacy)")
            return tracker, "csrt"
    except Exception:
        pass

    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT'):
            tracker = cv2.legacy.TrackerCSRT.create()
            logger.info("Using TrackerCSRT (legacy)")
            return tracker, "csrt"
    except Exception:
        pass

    try:
        if hasattr(cv2, 'TrackerCSRT_create'):
            tracker = cv2.TrackerCSRT_create()
            logger.info("Using TrackerCSRT")
            return tracker, "csrt"
    except Exception:
        pass

    # Fallback to TrackerMIL (always available)
    try:
        if hasattr(cv2, 'TrackerMIL') and hasattr(cv2.TrackerMIL, 'create'):
            tracker = cv2.TrackerMIL.create()
            logger.info("Using TrackerMIL (fallback)")
            return tracker, "mil"
    except Exception:
        pass

    raise RuntimeError(
        f"No compatible tracker available in OpenCV {cv2.__version__}. "
        "Please install opencv-contrib-python >= 4.7"
    )


@dataclass
class TrackingResult:
    """Result of a tracking operation."""
    success: bool
    roi: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    confidence: float
    method: str = "unknown"  # "vit", "nano", "csrt", "mil"
    error_message: Optional[str] = None


@dataclass
class TrackingState:
    """Internal state for tracking across frames."""
    tracker: Optional[Any] = None
    tracker_type: str = "unknown"  # "vit", "nano", "csrt", "mil"
    previous_frame: Optional[np.ndarray] = None  # Grayscale uint8
    previous_roi: Optional[Tuple[int, int, int, int]] = None
    initial_roi: Optional[Tuple[int, int, int, int]] = None
    confidence_history: List[float] = field(default_factory=list)
    roi_mode: str = "fixed_150x150"  # "fixed_150x150" or "adaptive"


class TrackingEngine:
    """
    Tracking engine for vessel ROI propagation.

    Uses best available OpenCV tracker (TrackerVit, TrackerNano, or CSRT)
    for robust ROI tracking across frames.

    Usage:
        engine = TrackingEngine()
        engine.initialize(frame, roi)
        result = engine.track(next_frame)

        if result.success:
            new_roi = result.roi
    """

    # Constants
    CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for success
    CONFIDENCE_WINDOW = 5  # Rolling window for confidence check
    FIXED_ROI_SIZE = 150  # Fixed ROI size when using fixed_150x150 mode

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        roi_mode: str = "fixed_150x150"
    ):
        """
        Initialize tracking engine.

        Args:
            confidence_threshold: Minimum confidence to continue tracking
            roi_mode: ROI tracking mode - "fixed_150x150" (constant 150x150) or "adaptive" (CSRT-determined size)
        """
        self.confidence_threshold = confidence_threshold
        self.roi_mode = roi_mode
        self._state = TrackingState()

    @property
    def is_initialized(self) -> bool:
        """Check if tracker is initialized."""
        return self._state.tracker is not None

    def initialize(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        roi_mode: str = None
    ) -> bool:
        """
        Initialize CSRT tracker with initial frame and ROI.

        Args:
            frame: Initial frame (grayscale or BGR)
            roi: ROI bounding box (x, y, w, h)
            roi_mode: Override ROI mode for this session ("fixed_150x150" or "adaptive")

        Returns:
            True if initialization successful
        """
        try:
            self.reset()

            # Set ROI mode for this session
            if roi_mode is not None:
                self.roi_mode = roi_mode
            self._state.roi_mode = self.roi_mode

            # Prepare frame with preprocessing for better tracking
            frame_gray = self._ensure_grayscale_uint8(frame)
            frame_bgr = self._preprocess_for_tracking(frame_gray)
            frame_h, frame_w = frame_gray.shape[:2]

            # Adjust ROI based on mode
            x, y, w, h = roi
            if self.roi_mode == "fixed_150x150":
                # Use fixed 150x150 size, centered on provided ROI center
                center_x = x + w // 2
                center_y = y + h // 2
                new_w = self.FIXED_ROI_SIZE
                new_h = self.FIXED_ROI_SIZE
                new_x = center_x - new_w // 2
                new_y = center_y - new_h // 2

                # Clamp to frame bounds
                new_x = max(0, min(new_x, frame_w - new_w))
                new_y = max(0, min(new_y, frame_h - new_h))
                roi = (new_x, new_y, new_w, new_h)
                logger.info(f"Using fixed 150x150 ROI mode, adjusted ROI: {roi}")
            else:
                # adaptive mode - use provided ROI as-is
                logger.info(f"Using adaptive ROI mode, ROI: {roi}")

            # Validate ROI
            x, y, w, h = roi
            if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                logger.error(f"ROI out of bounds: {roi}, frame: {frame_w}x{frame_h}")
                return False

            if w <= 0 or h <= 0:
                logger.error(f"Invalid ROI dimensions: {roi}")
                return False

            # Initialize tracker
            tracker, tracker_type = create_tracker()
            self._state.tracker = tracker
            self._state.tracker_type = tracker_type

            logger.info(f"Init frame: shape={frame_bgr.shape}, min={frame_bgr.min()}, max={frame_bgr.max()}")
            logger.info(f"Init ROI: {roi}")

            # OpenCV 4.x trackers may return None from init() (void function)
            # Only fail if explicitly returns False
            init_result = self._state.tracker.init(frame_bgr, roi)

            # Check tracking score after init (TrackerVit specific)
            if hasattr(self._state.tracker, 'getTrackingScore'):
                score = self._state.tracker.getTrackingScore()
                logger.info(f"Init score: {score}")

            if init_result is False:
                logger.error(f"Tracker ({tracker_type}) initialization failed")
                return False

            # Store state
            self._state.previous_frame = frame_gray
            self._state.previous_roi = roi
            self._state.initial_roi = roi
            self._state.confidence_history = [1.0]  # Initial confidence

            logger.info(f"Tracking initialized: tracker={tracker_type}, ROI={roi}, mode={self.roi_mode}")
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
                confidence=0.0, method=self._state.tracker_type,
                error_message="Tracker not initialized"
            )

        try:
            frame_gray = self._ensure_grayscale_uint8(frame)
            frame_bgr = self._preprocess_for_tracking(frame_gray)
            frame_h, frame_w = frame_gray.shape[:2]

            logger.info(f"Track: frame shape={frame_bgr.shape}, min={frame_bgr.min()}, max={frame_bgr.max()}, tracker={self._state.tracker_type}")

            # Tracker update
            success, bbox = self._state.tracker.update(frame_bgr)

            # Check tracking score (TrackerVit specific)
            score = None
            if hasattr(self._state.tracker, 'getTrackingScore'):
                score = self._state.tracker.getTrackingScore()

            logger.info(f"Track result: success={success}, bbox={bbox}, score={score}")

            if not success:
                logger.warning(f"Tracking failed: bbox={bbox}, previous_roi={self._state.previous_roi}")
                return TrackingResult(
                    success=False, roi=self._state.previous_roi,
                    confidence=0.0, method=self._state.tracker_type,
                    error_message="Tracking failed"
                )

            new_x, new_y, new_w, new_h = map(int, bbox)

            # Apply ROI mode
            if self._state.roi_mode == "fixed_150x150":
                # Keep fixed 150x150 size, only track center position
                center_x = new_x + new_w // 2
                center_y = new_y + new_h // 2
                fixed_w = self.FIXED_ROI_SIZE
                fixed_h = self.FIXED_ROI_SIZE
                roi_x = center_x - fixed_w // 2
                roi_y = center_y - fixed_h // 2

                # Clamp to frame bounds
                roi_x = max(0, min(roi_x, frame_w - fixed_w))
                roi_y = max(0, min(roi_y, frame_h - fixed_h))
                roi = (roi_x, roi_y, fixed_w, fixed_h)
            else:
                # adaptive mode - use CSRT's determined size
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

            logger.debug(f"Tracked: ROI={roi}, confidence={confidence:.3f}, mode={self._state.roi_mode}, tracker={self._state.tracker_type}")

            return TrackingResult(
                success=True, roi=roi,
                confidence=confidence, method=self._state.tracker_type
            )

        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return TrackingResult(
                success=False, roi=self._state.previous_roi,
                confidence=0.0, method=self._state.tracker_type,
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

    def _preprocess_for_tracking(self, frame_gray: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for tracking.

        Applies histogram equalization to improve contrast for TrackerVit.
        Medical images often have low contrast which causes tracking failures.
        """
        # Apply histogram equalization for better contrast
        frame_eq = cv2.equalizeHist(frame_gray)
        # Convert to BGR for tracker
        frame_bgr = cv2.cvtColor(frame_eq, cv2.COLOR_GRAY2BGR)
        return frame_bgr

    def reset(self):
        """Reset tracking state."""
        self._state = TrackingState()
        logger.info("Tracking engine reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current tracking state for debugging."""
        return {
            "is_initialized": self.is_initialized,
            "tracker_type": self._state.tracker_type,
            "previous_roi": self._state.previous_roi,
            "initial_roi": self._state.initial_roi,
            "confidence_history": self._state.confidence_history[-10:],
            "avg_confidence": np.mean(self._state.confidence_history[-5:]) if self._state.confidence_history else 0.0,
            "roi_mode": self._state.roi_mode
        }


# Singleton instance
_engine_instance: Optional[TrackingEngine] = None


def get_engine() -> TrackingEngine:
    """Get singleton TrackingEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TrackingEngine()
    return _engine_instance
