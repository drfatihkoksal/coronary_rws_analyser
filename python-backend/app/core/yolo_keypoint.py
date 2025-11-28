"""
YOLO Keypoint Detection Engine for Vessel Seed Points

Detects 5 keypoints along the vessel centerline for AngioPy seed point generation.
Keypoints: start → quarter → center → three_quarter → end

Model: YOLOv8-pose trained on coronary vessel dataset (150x150)
Output: 5 ordered (x, y) coordinates for AngioPy segmentation

References:
- Ultralytics YOLOv8: https://docs.ultralytics.com/tasks/pose/
- AngioPy: Petersen et al., Int J Cardiol 2024
"""

from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Lazy import for ultralytics
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


@dataclass
class KeypointResult:
    """Result of YOLO keypoint detection."""
    keypoints: List[Tuple[float, float]]  # 5 keypoints as (x, y) in ROI coordinates
    confidence: float  # Detection confidence
    bbox: Optional[Tuple[int, int, int, int]]  # Detected bbox (x, y, w, h) in ROI
    success: bool
    error_message: Optional[str] = None

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def to_image_coords(
        self,
        roi_x: int,
        roi_y: int
    ) -> List[Tuple[float, float]]:
        """
        Convert keypoints from ROI coordinates to full image coordinates.

        Args:
            roi_x: ROI top-left x in image
            roi_y: ROI top-left y in image

        Returns:
            List of (x, y) in image coordinates
        """
        return [(kp[0] + roi_x, kp[1] + roi_y) for kp in self.keypoints]


class YOLOKeypointEngine:
    """
    YOLO-based keypoint detection for vessel seed points.

    Detects 5 keypoints along the vessel centerline within a 150x150 ROI.
    These keypoints are used as seed points for AngioPy segmentation.

    Usage:
        engine = YOLOKeypointEngine()
        result = engine.detect(roi_crop)  # 150x150 grayscale

        if result.success:
            seed_points = result.to_image_coords(roi_x, roi_y)
            # Use with AngioPy
    """

    # Model configuration
    INPUT_SIZE = 150
    NUM_KEYPOINTS = 5
    KEYPOINT_NAMES = ['start', 'quarter', 'center', 'three_quarter', 'end']

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.15,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO keypoint engine.

        Args:
            model_path: Path to trained .pt model. Falls back to default location.
            confidence_threshold: Minimum detection confidence
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics is required for YOLO keypoint detection. "
                "Install with: pip install ultralytics"
            )

        # Resolve model path
        if model_path is None:
            backend_dir = Path(__file__).parent.parent.parent
            model_path = (
                backend_dir / "models" / "Dataset506_YOLO_Keypoint" /
                "runs" / "vessel_endpoint_v1" / "weights" / "best.pt"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO keypoint model not found: {self.model_path}\n"
                "Train with: python models/Dataset506_YOLO_Keypoint/train_yolo_keypoint.py"
            )

        self.confidence_threshold = confidence_threshold
        self.device = device

        # Lazy-loaded model
        self._model = None
        self._model_loaded = False

        logger.info(f"YOLOKeypointEngine initialized (model={self.model_path.name})")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def _load_model(self):
        """Load YOLO model on first use (lazy loading)."""
        if self._model_loaded:
            return

        try:
            logger.info(f"Loading YOLO keypoint model from {self.model_path}")

            self._model = YOLO(str(self.model_path))

            # Set device if specified
            if self.device:
                self._model.to(self.device)

            self._model_loaded = True
            logger.info(f"YOLO keypoint model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for YOLO inference.

        Args:
            image: ROI crop, grayscale or BGR

        Returns:
            BGR image ready for YOLO
        """
        import cv2

        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def detect(self, roi_crop: np.ndarray) -> KeypointResult:
        """
        Detect vessel keypoints in ROI crop.

        Args:
            roi_crop: ROI image

        Returns:
            KeypointResult with detected keypoints in ROI coordinates
        """
        try:
            # Lazy load model
            self._load_model()

            # Prepare input
            image = self._prepare_input(roi_crop)
            h, w = roi_crop.shape[:2]

            # Run inference
            results = self._model.predict(
                source=image,
                conf=self.confidence_threshold,
                verbose=False,
                imgsz=160
            )

            # Parse results
            if len(results) == 0 or results[0].keypoints is None:
                return KeypointResult(
                    keypoints=[],
                    confidence=0.0,
                    bbox=None,
                    success=False,
                    error_message="No vessel detected in ROI"
                )

            result = results[0]

            # Check if any detections
            if len(result.boxes) == 0:
                return KeypointResult(
                    keypoints=[],
                    confidence=0.0,
                    bbox=None,
                    success=False,
                    error_message="No vessel detected in ROI"
                )

            # Get best detection (highest confidence)
            best_idx = int(result.boxes.conf.argmax())
            confidence = float(result.boxes.conf[best_idx])

            # Get bounding box
            bbox_xyxy = result.boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = bbox_xyxy
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # Get keypoints
            kps = result.keypoints.xy[best_idx].cpu().numpy()  # (N, 2)

            # Clamp keypoints to ROI bounds
            keypoints = []
            for i, (x, y) in enumerate(kps):
                x_clamped = max(0, min(float(x), w - 1))
                y_clamped = max(0, min(float(y), h - 1))
                keypoints.append((x_clamped, y_clamped))

            if len(keypoints) < 2:
                return KeypointResult(
                    keypoints=keypoints,
                    confidence=confidence,
                    bbox=bbox,
                    success=False,
                    error_message=f"Insufficient keypoints detected: {len(keypoints)}"
                )

            logger.info(f"[YOLO] Detected {len(keypoints)} keypoints, confidence={confidence:.3f}")
            logger.info(f"[YOLO] Keypoints: {[(f'{x:.1f}', f'{y:.1f}') for x, y in keypoints]}")

            return KeypointResult(
                keypoints=keypoints,
                confidence=confidence,
                bbox=bbox,
                success=True
            )

        except Exception as e:
            logger.error(f"YOLO keypoint detection failed: {e}")
            return KeypointResult(
                keypoints=[],
                confidence=0.0,
                bbox=None,
                success=False,
                error_message=str(e)
            )

    def detect_with_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> KeypointResult:
        """
        Detect keypoints from full frame with ROI specification.

        Args:
            frame: Full frame image
            roi: ROI as (x, y, w, h)

        Returns:
            KeypointResult with keypoints in IMAGE coordinates (not ROI)
        """
        x, y, w, h = roi

        # Validate ROI
        frame_h, frame_w = frame.shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return KeypointResult(
                keypoints=[],
                confidence=0.0,
                bbox=None,
                success=False,
                error_message=f"ROI out of bounds: {roi}, frame: {frame_w}x{frame_h}"
            )

        # Crop ROI
        roi_crop = frame[y:y+h, x:x+w]

        # Detect in ROI
        result = self.detect(roi_crop)

        if result.success:
            # Convert to image coordinates
            image_keypoints = result.to_image_coords(x, y)

            return KeypointResult(
                keypoints=image_keypoints,
                confidence=result.confidence,
                bbox=(x + result.bbox[0], y + result.bbox[1], result.bbox[2], result.bbox[3]) if result.bbox else None,
                success=True
            )

        return result

    def unload_model(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("YOLO keypoint model unloaded")


# Singleton instance
_yolo_instance: Optional[YOLOKeypointEngine] = None


def get_yolo_keypoint(model_path: Optional[str] = None) -> YOLOKeypointEngine:
    """Get singleton YOLOKeypointEngine instance."""
    global _yolo_instance

    if _yolo_instance is None:
        _yolo_instance = YOLOKeypointEngine(model_path=model_path)

    return _yolo_instance
