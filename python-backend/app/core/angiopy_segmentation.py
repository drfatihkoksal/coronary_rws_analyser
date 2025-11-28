"""
AngioPy Vessel Segmentation Engine

Seed-guided deep learning segmentation for coronary arteries.
Based on: https://gitlab.com/epfl-center-for-imaging/angiopy/angiopy-segmentation
Paper: https://doi.org/10.1016/j.ijcard.2024.132598

Architecture:
- U-Net with InceptionResNetV2 encoder (ImageNet pretrained)
- Input: 3-channel RGB (512x512)
  * Channel 0: Grayscale angiography image
  * Channel 1: Start/End seed points (first/last points as 4x4 squares)
  * Channel 2: Middle seed points (intermediate points as 4x4 squares)
- Output: Binary segmentation mask (2 classes: background, artery)

Requirements:
- 2-10 seed points along vessel
- Points ordered from proximal to distal
- Automatically resizes to 512x512 for inference
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
TORCH_AVAILABLE = False
SMP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AngioPyResult:
    """Result of AngioPy segmentation."""
    mask: np.ndarray  # Binary mask (H, W), values 0 or 1
    probability_map: Optional[np.ndarray]  # Probability [0, 1] if requested
    original_shape: Tuple[int, int]  # Original image shape (H, W)
    num_seed_points: int  # Number of seed points used


class AngioPySegmentation:
    """
    Seed-guided vessel segmentation using U-Net + InceptionResNetV2.

    Usage:
        model = AngioPySegmentation(model_path="/path/to/weights.pth")
        result = model.segment(image, seed_points=[(x1,y1), (x2,y2), ...])
        mask = result.mask

    Ref: Petersen et al., "AngioPy Segmentation: An open-source, user-guided
         deep learning tool for coronary artery segmentation",
         Int J Cardiol 2024
    """

    # Model configuration
    INPUT_SIZE = 512
    N_CLASSES = 2  # Background, Artery
    ENCODER_NAME = 'inceptionresnetv2'
    ENCODER_WEIGHTS = 'imagenet'

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize AngioPy segmentation model.

        Args:
            model_path: Path to model weights (.pth file).
                        Falls back to default location in models/ directory.
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AngioPy. Install with: pip install torch")

        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation_models_pytorch is required for AngioPy. "
                "Install with: pip install segmentation-models-pytorch timm"
            )

        # Resolve model path
        if model_path is None:
            # Default: python-backend/models/angiopy/
            backend_dir = Path(__file__).parent.parent.parent
            model_path = backend_dir / "models" / "angiopy" / "modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"AngioPy model not found: {self.model_path}\n"
                f"Download from: https://gitlab.com/epfl-center-for-imaging/angiopy/angiopy-segmentation"
            )

        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Lazy-loaded model
        self._model = None
        self._model_loaded = False

        logger.info(f"AngioPySegmentation initialized (device={self.device})")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def _load_model(self):
        """Load model weights on first use (lazy loading)."""
        if self._model_loaded:
            return

        try:
            import torch
            import torch.nn as nn
            import segmentation_models_pytorch as smp

            logger.info(f"Loading AngioPy model from {self.model_path}")

            # Create U-Net with InceptionResNetV2 encoder
            self._model = smp.Unet(
                encoder_name=self.ENCODER_NAME,
                encoder_weights=self.ENCODER_WEIGHTS,
                in_channels=3,
                classes=self.N_CLASSES
            )

            # Wrap with DataParallel for compatibility with saved weights
            self._model = nn.DataParallel(self._model)
            self._model.to(device=self.device)

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self._model.load_state_dict(checkpoint)
            self._model.eval()

            self._model_loaded = True
            logger.info(f"AngioPy model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load AngioPy model: {e}")
            raise

    def _prepare_input(
        self,
        image: np.ndarray,
        seed_points: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Prepare 3-channel input with seed point encoding.

        Args:
            image: Grayscale image (H, W) or (H, W, 1)
            seed_points: List of (x, y) coordinates (ordered start→end)

        Returns:
            Tuple of (rgb_array, scale_x, scale_y)
            - rgb_array: (512, 512, 3) uint8 array
            - scale_x, scale_y: Scale factors for coordinate mapping
        """
        import cv2
        import scipy.ndimage

        # Ensure grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]

        original_shape = image.shape[:2]
        h, w = original_shape

        # Resize to 512x512 if needed
        if image.shape != (self.INPUT_SIZE, self.INPUT_SIZE):
            scale_y = self.INPUT_SIZE / h
            scale_x = self.INPUT_SIZE / w
            image = scipy.ndimage.zoom(image, (scale_y, scale_x), order=1)
        else:
            scale_x, scale_y = 1.0, 1.0

        # Normalize to 0-255 uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        # Create RGB array (512, 512, 3)
        rgb_array = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        rgb_array[:, :, 0] = image  # Channel 0: Grayscale image

        # Scale seed points
        scaled_points = [(x * scale_x, y * scale_y) for x, y in seed_points]

        # Encode seed points as 4x4 white squares
        if len(scaled_points) >= 2:
            # Channel 1: Start and End points (first and last)
            start_point = scaled_points[0]
            end_point = scaled_points[-1]

            for px, py in [start_point, end_point]:
                ix, iy = int(round(px)), int(round(py))
                y_min = max(0, iy - 2)
                y_max = min(self.INPUT_SIZE, iy + 2)
                x_min = max(0, ix - 2)
                x_max = min(self.INPUT_SIZE, ix + 2)
                rgb_array[y_min:y_max, x_min:x_max, 1] = 255

            # Channel 2: Middle points (all except first and last)
            if len(scaled_points) > 2:
                for px, py in scaled_points[1:-1]:
                    ix, iy = int(round(px)), int(round(py))
                    y_min = max(0, iy - 2)
                    y_max = min(self.INPUT_SIZE, iy + 2)
                    x_min = max(0, ix - 2)
                    x_max = min(self.INPUT_SIZE, ix + 2)
                    rgb_array[y_min:y_max, x_min:x_max, 2] = 255

        return rgb_array, scale_x, scale_y

    def _preprocess(self, rgb_array: np.ndarray) -> 'torch.Tensor':
        """
        Convert RGB array to model input tensor.

        Args:
            rgb_array: (512, 512, 3) uint8 array

        Returns:
            Tensor (1, 3, 512, 512) float32 normalized [0, 1]
        """
        import torch

        # Normalize to [0, 1]
        img_array = rgb_array.astype(np.float32) / 255.0

        # Transpose to (C, H, W)
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        return tensor

    def _postprocess(
        self,
        output: 'torch.Tensor',
        original_shape: Tuple[int, int],
        return_probability: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert model output to binary mask.

        Args:
            output: Model output tensor (1, 2, 512, 512)
            original_shape: Target output shape (H, W)
            return_probability: Also return probability map

        Returns:
            Tuple of (binary_mask, probability_map)
        """
        import torch
        import cv2

        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)  # (1, 2, H, W)

        # Take channel 1 (artery class)
        artery_prob = probs[0, 1].cpu().numpy()  # (H, W)

        # Threshold to binary mask
        binary_mask = (artery_prob > 0.5).astype(np.uint8)

        # Resize to original shape if needed
        if binary_mask.shape != original_shape:
            binary_mask = cv2.resize(
                binary_mask,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            if return_probability:
                artery_prob = cv2.resize(
                    artery_prob,
                    (original_shape[1], original_shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

        prob_map = artery_prob if return_probability else None
        return binary_mask, prob_map

    def segment(
        self,
        image: np.ndarray,
        seed_points: List[Tuple[float, float]],
        return_probability: bool = False
    ) -> AngioPyResult:
        """
        Segment vessel with seed point guidance.

        Args:
            image: Grayscale image (H, W) or (H, W, 1)
            seed_points: List of (x, y) coordinates (2-10 points, ordered start→end)
            return_probability: Also return probability map

        Returns:
            AngioPyResult with mask and optional probability map

        Raises:
            ValueError: If less than 2 seed points provided
        """
        import torch

        # Validate seed points
        if not seed_points or len(seed_points) < 2:
            raise ValueError(f"At least 2 seed points required, got {len(seed_points) if seed_points else 0}")

        # Handle edge cases
        if len(seed_points) == 2:
            # Duplicate last point for better results (AngioPy works better with 3+)
            seed_points = list(seed_points) + [seed_points[-1]]
            logger.debug("Duplicated last seed point (2→3 points)")

        if len(seed_points) > 10:
            logger.warning(f"Using first 10 of {len(seed_points)} seed points")
            seed_points = seed_points[:10]

        # Lazy load model
        self._load_model()

        # Store original shape
        original_shape = image.shape[:2]

        # Prepare input
        rgb_input, scale_x, scale_y = self._prepare_input(image, seed_points)

        # Preprocess
        input_tensor = self._preprocess(rgb_input)
        input_tensor = input_tensor.to(device=self.device)

        # Inference
        with torch.no_grad():
            output = self._model(input_tensor)

        # Postprocess
        binary_mask, prob_map = self._postprocess(
            output, original_shape, return_probability
        )

        return AngioPyResult(
            mask=binary_mask,
            probability_map=prob_map,
            original_shape=original_shape,
            num_seed_points=len(seed_points)
        )

    def unload_model(self):
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False

            if TORCH_AVAILABLE:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info("AngioPy model unloaded")


# Singleton instance
_angiopy_instance: Optional[AngioPySegmentation] = None


def get_angiopy(model_path: Optional[str] = None) -> AngioPySegmentation:
    """Get singleton AngioPySegmentation instance."""
    global _angiopy_instance

    if _angiopy_instance is None:
        _angiopy_instance = AngioPySegmentation(model_path=model_path)

    return _angiopy_instance
