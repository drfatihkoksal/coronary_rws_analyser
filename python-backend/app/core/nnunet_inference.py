"""
nnU-Net Inference Engine

Custom nnU-Net model for coronary artery segmentation.
Supports both single-channel and dual-channel (image + Gaussian spatial attention) modes.

Model Architecture:
- Input: 1-channel (grayscale) or 2-channel (grayscale + Gaussian spatial map)
- Output: Binary segmentation mask
- Post-processing: CenterComponentKeeper (bifurcation suppression)
- Current default: Dataset506 with CLDice loss (single-channel)

References:
- Isensee et al., "nnU-Net: A Self-Configuring Method for Deep Learning-based
  Biomedical Image Segmentation", Nature Methods 2021
- CLDice: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
  for Tubular Structure Segmentation", CVPR 2021
- Custom training on coronary angiography dataset
"""

import os
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
TORCH_AVAILABLE = False
NNUNET_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    NNUNET_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SegmentationResult:
    """Result of segmentation inference."""
    mask: np.ndarray  # Binary mask
    probability_map: Optional[np.ndarray]  # Probability values [0, 1]
    original_shape: Tuple[int, int]  # Original image shape
    roi_used: Optional[Tuple[int, int, int, int]]  # ROI if cropping was applied


class CenterComponentKeeper:
    """
    Post-processor to keep only the center vessel component.

    Removes bifurcations and side branches by keeping the connected
    component closest to ROI center.
    """

    def __init__(self, center_tolerance_radius: int = 10):
        """
        Args:
            center_tolerance_radius: Radius around center to consider (pixels)
        """
        self.center_tolerance_radius = center_tolerance_radius

    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Keep only the center connected component.

        Args:
            mask: Binary segmentation mask

        Returns:
            Filtered mask with only center component
        """
        try:
            from scipy import ndimage

            # Find connected components
            labeled, num_features = ndimage.label(mask)

            if num_features <= 1:
                return mask

            # Get image center
            h, w = mask.shape
            center_y, center_x = h // 2, w // 2

            # Find which component contains the center point
            # Check within tolerance radius
            best_component = 0
            min_distance = float('inf')

            for label_id in range(1, num_features + 1):
                component = (labeled == label_id)
                coords = np.column_stack(np.where(component))

                if len(coords) == 0:
                    continue

                # Find closest point to center
                distances = np.sqrt(
                    (coords[:, 0] - center_y)**2 +
                    (coords[:, 1] - center_x)**2
                )
                closest_distance = np.min(distances)

                if closest_distance < min_distance:
                    min_distance = closest_distance
                    best_component = label_id

            # Keep only the center component
            if best_component > 0:
                return (labeled == best_component).astype(np.uint8)

            return mask

        except ImportError:
            logger.warning("scipy not available for connected components")
            return mask
        except Exception as e:
            logger.warning(f"Center component filtering failed: {e}")
            return mask


class NNUNetInference:
    """
    nnU-Net inference wrapper for coronary artery segmentation.

    Usage:
        model = NNUNetInference(model_dir="/path/to/nnUNet_results")
        result = model.segment(image, roi=(x, y, w, h))
        mask = result.mask
    """

    # Default model configuration
    DEFAULT_DATASET_ID = 506  # Dataset506_CoronarySingleChannel with CLDice
    DEFAULT_CONFIGURATION = '2d'
    DEFAULT_TRAINER = 'nnUNetTrainerCLDice'  # CLDice loss for better tubular segmentation
    DEFAULT_FOLD = 0
    ALL_FOLDS = (0, 1, 2, 3, 4)  # For ensemble inference

    # Default model path (Windows path for local development)
    DEFAULT_MODEL_DIR = r"C:\Users\fatih\programim\Coronary_Clear_Vision_V2.2\python-backend\training_data\nnUnet_results"

    # Dataset configurations
    DATASET_CONFIG = {
        506: {
            'name': 'Dataset506_CoronarySingleChannel',
            'trainer': 'nnUNetTrainerCLDice',
            'dual_channel': False,
            'sigma': None,  # Not used for single-channel
            'size': 150,
        },
        503: {
            'name': 'Dataset503_Coronary150GaussianDualChannel',
            'trainer': 'nnUNetTrainer',
            'dual_channel': True,
            'sigma': 26,  # 150x150 model
            'size': 150,
        },
        502: {
            'name': 'Dataset502_CoronaryGaussianDualChannel',
            'trainer': 'nnUNetTrainer',
            'dual_channel': True,
            'sigma': 35,  # 200x200 model
            'size': 200,
        },
        501: {
            'name': 'Dataset501_CoronaryROI',
            'trainer': 'nnUNetTrainer',
            'dual_channel': True,
            'sigma': 35,
            'size': 200,
        },
    }

    # Legacy maps for backward compatibility
    SIGMA_MAP = {
        506: None, # Single-channel, no Gaussian
        503: 26,   # 150x150 model (scaled from 35 * 150/200)
        502: 35,   # 200x200 model
        501: 35,   # Original model
    }

    SIZE_MAP = {
        506: 150,
        503: 150,
        502: 200,
        501: 200,
    }

    def __init__(
        self,
        model_dir: Optional[str] = None,
        dataset_id: int = 506,
        use_dual_channel: Optional[bool] = None,
        enable_bifurcation_suppression: bool = True,
        use_ensemble: bool = True
    ):
        """
        Initialize nnU-Net inference.

        Args:
            model_dir: Path to nnUNet_results directory.
                       Falls back to DEFAULT_MODEL_DIR or nnUNet_results env variable.
            dataset_id: Dataset ID (501, 502, 503, 506)
            use_dual_channel: Use dual-channel input (image + spatial map).
                             If None, automatically determined from dataset config.
            enable_bifurcation_suppression: Remove side branches via post-processing
            use_ensemble: Use all 5 folds for ensemble inference (recommended)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for nnU-Net inference")

        if not NNUNET_AVAILABLE:
            raise ImportError("nnunetv2 is required for inference. Install with: pip install nnunetv2")

        # Get dataset configuration
        self.dataset_id = dataset_id
        self._dataset_config = self.DATASET_CONFIG.get(dataset_id, {})

        # Determine model directory (priority: param > env > default)
        self.model_dir = model_dir or os.environ.get('nnUNet_results') or self.DEFAULT_MODEL_DIR

        # Auto-detect dual_channel from dataset config if not specified
        if use_dual_channel is None:
            self.use_dual_channel = self._dataset_config.get('dual_channel', False)
        else:
            self.use_dual_channel = use_dual_channel

        self.enable_bifurcation_suppression = enable_bifurcation_suppression
        self.use_ensemble = use_ensemble
        self.configuration = self.DEFAULT_CONFIGURATION
        self.folds = self.ALL_FOLDS if use_ensemble else (self.DEFAULT_FOLD,)

        # Get trainer name from config
        self._trainer = self._dataset_config.get('trainer', self.DEFAULT_TRAINER)

        # Lazy-loaded model
        self._predictor = None

        # Post-processor
        if enable_bifurcation_suppression:
            self._post_processor = CenterComponentKeeper(center_tolerance_radius=10)
        else:
            self._post_processor = None

        # Model parameters
        self._sigma = self._dataset_config.get('sigma') or self.SIGMA_MAP.get(dataset_id)
        self._target_size = self._dataset_config.get('size') or self.SIZE_MAP.get(dataset_id, 150)

        logger.info(f"NNUNetInference initialized: dataset={dataset_id}, "
                   f"trainer={self._trainer}, dual_channel={self.use_dual_channel}, "
                   f"ensemble={use_ensemble}, sigma={self._sigma}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._predictor is not None

    def _load_model(self):
        """Load nnU-Net model on first use."""
        if self._predictor is not None:
            return

        if not self.model_dir:
            raise ValueError(
                "Model directory not provided. Set nnUNet_results environment "
                "variable or pass model_dir parameter."
            )

        try:
            # Get dataset name from config or construct default
            dataset_name = self._dataset_config.get('name')
            if not dataset_name:
                # Fallback to legacy names
                dataset_names = {
                    501: "Dataset501_CoronaryROI",
                    502: "Dataset502_CoronaryGaussianDualChannel",
                    503: "Dataset503_Coronary150GaussianDualChannel",
                    506: "Dataset506_CoronarySingleChannel",
                }
                dataset_name = dataset_names.get(
                    self.dataset_id,
                    f"Dataset{self.dataset_id:03d}_CoronaryROI"
                )

            # Get trainer from config (e.g., nnUNetTrainerCLDice for Dataset506)
            trainer_name = f"{self._trainer}__nnUNetPlans__{self.configuration}"
            model_folder = os.path.join(self.model_dir, dataset_name, trainer_name)

            if not os.path.exists(model_folder):
                raise FileNotFoundError(f"Model folder not found: {model_folder}")

            # Initialize predictor
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Loading nnU-Net model on {device}")

            self._predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )

            # Check available folds and determine checkpoint name
            available_folds = []
            checkpoint_name = 'checkpoint_best.pth'

            for fold in self.folds:
                fold_folder = os.path.join(model_folder, f"fold_{fold}")
                if os.path.exists(fold_folder):
                    # Check for checkpoint
                    if os.path.exists(os.path.join(fold_folder, 'checkpoint_best.pth')):
                        available_folds.append(fold)
                    elif os.path.exists(os.path.join(fold_folder, 'checkpoint_latest.pth')):
                        available_folds.append(fold)
                        checkpoint_name = 'checkpoint_latest.pth'

            if not available_folds:
                raise FileNotFoundError(f"No valid folds found in {model_folder}")

            logger.info(f"Using folds: {available_folds} with checkpoint: {checkpoint_name}")

            self._predictor.initialize_from_trained_model_folder(
                model_folder,
                use_folds=tuple(available_folds),
                checkpoint_name=checkpoint_name
            )

            logger.info(f"Model loaded successfully from {model_folder} (ensemble={self.use_ensemble}, folds={available_folds})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_gaussian_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate Gaussian spatial attention map.

        The Gaussian is centered on the image and normalized to [0, 1].

        Args:
            shape: (height, width) of output map

        Returns:
            Normalized Gaussian map
        """
        h, w = shape
        center_y, center_x = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * self._sigma**2))

        # Normalize to [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

        return gaussian.astype(np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns:
            Preprocessed image (C, H, W) where C=2 for dual-channel
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = image[:, :, 0]

        # Convert to float32 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Generate dual-channel input
        if self.use_dual_channel:
            spatial_map = self.generate_gaussian_map(image.shape)
            return np.stack([image, spatial_map], axis=0)

        return image[np.newaxis, :, :]

    def segment(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        return_probability: bool = True
    ) -> SegmentationResult:
        """
        Segment coronary vessel in image.

        Args:
            image: Input image (H, W) grayscale
            roi: Optional ROI (x, y, w, h) to crop before inference
            return_probability: Return probability map

        Returns:
            SegmentationResult with mask and optional probability
        """
        self._load_model()

        original_shape = image.shape[:2]

        try:
            # Model expects 150x150 input
            target_size = self._target_size  # 150

            # Crop to ROI if provided
            if roi is not None:
                x, y, w, h = roi
                # Calculate center of ROI
                center_x = x + w // 2
                center_y = y + h // 2

                # Crop 150x150 from center
                half_size = target_size // 2
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(image.shape[1], x1 + target_size)
                y2 = min(image.shape[0], y1 + target_size)

                # Adjust x1, y1 if we hit the boundary
                if x2 - x1 < target_size:
                    x1 = max(0, x2 - target_size)
                if y2 - y1 < target_size:
                    y1 = max(0, y2 - target_size)

                crop = image[y1:y2, x1:x2]
                logger.info(f"ROI crop: center=({center_x}, {center_y}), crop_region=({x1}, {y1}, {x2}, {y2}), crop_shape={crop.shape}")
            else:
                # No ROI - crop center of image
                center_x = image.shape[1] // 2
                center_y = image.shape[0] // 2
                half_size = target_size // 2
                x1 = center_x - half_size
                y1 = center_y - half_size
                x2 = x1 + target_size
                y2 = y1 + target_size
                crop = image[y1:y2, x1:x2]

            # Preprocess (crop should be ~150x150)
            preprocessed = self._preprocess(crop)

            # Add singleton Z dimension for 2D model: (C, H, W) -> (C, 1, H, W)
            input_array = preprocessed[:, np.newaxis, :, :].astype(np.float32)

            # Properties for nnUNet
            properties = {
                'spacing': [999.0, 1.0, 1.0],
                'shape_after_cropping_and_before_resampling': input_array.shape[1:]
            }

            # Run inference
            result = self._predictor.predict_single_npy_array(
                input_array,
                properties,
                None, None,
                save_or_return_probabilities=return_probability
            )

            # Parse result
            prediction = result[0] if isinstance(result, tuple) else result

            # Remove Z dimension if present: (C, 1, H, W) -> (C, H, W)
            if len(prediction.shape) == 4:
                prediction = prediction[:, 0, :, :]

            # Extract vessel class (class 1)
            if len(prediction.shape) == 3 and prediction.shape[0] > 1:
                probability_map = prediction[1].astype(np.float32) if return_probability else None
                binary_mask = (prediction[1] > 0.5).astype(np.uint8)
            else:
                prob = prediction[0] if len(prediction.shape) == 3 else prediction
                probability_map = prob.astype(np.float32) if return_probability else None
                binary_mask = (prob > 0.5).astype(np.uint8)

            # Post-processing (bifurcation suppression)
            if self._post_processor is not None:
                binary_mask = self._post_processor.process(binary_mask)

            logger.info(f"Mask from model: shape={binary_mask.shape}, sum={binary_mask.sum()}")

            # Map mask back to full image at original position
            full_mask = np.zeros(original_shape, dtype=np.uint8)
            full_prob = np.zeros(original_shape, dtype=np.float32) if return_probability else None

            # Get actual mask dimensions (should be ~150x150)
            mask_h, mask_w = binary_mask.shape

            # Place mask at crop position
            paste_y2 = min(y1 + mask_h, original_shape[0])
            paste_x2 = min(x1 + mask_w, original_shape[1])
            mask_h_actual = paste_y2 - y1
            mask_w_actual = paste_x2 - x1

            full_mask[y1:paste_y2, x1:paste_x2] = binary_mask[:mask_h_actual, :mask_w_actual]

            if full_prob is not None and probability_map is not None:
                full_prob[y1:paste_y2, x1:paste_x2] = probability_map[:mask_h_actual, :mask_w_actual]

            binary_mask = full_mask
            probability_map = full_prob

            return SegmentationResult(
                mask=binary_mask,
                probability_map=probability_map,
                original_shape=original_shape,
                roi_used=roi
            )

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise

    def segment_batch(
        self,
        images: List[np.ndarray],
        rois: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[SegmentationResult]:
        """
        Segment multiple images.

        Args:
            images: List of input images
            rois: Optional list of ROIs (one per image)

        Returns:
            List of SegmentationResults
        """
        results = []
        for i, image in enumerate(images):
            roi = rois[i] if rois else None
            results.append(self.segment(image, roi=roi))
        return results


# Singleton instance
_inference_instance: Optional[NNUNetInference] = None


def get_inference(
    model_dir: Optional[str] = None,
    dataset_id: int = 506,
    use_ensemble: bool = True
) -> NNUNetInference:
    """Get singleton NNUNetInference instance."""
    global _inference_instance

    if _inference_instance is None:
        _inference_instance = NNUNetInference(
            model_dir=model_dir,
            dataset_id=dataset_id,
            use_ensemble=use_ensemble
        )

    return _inference_instance
