"""
DICOM Handler for Coronary Angiography

Handles DICOM file loading, frame extraction, and metadata parsing.
Supports multi-frame angiography with ECG metadata extraction.

References:
- DICOM Standard PS3.3 (Information Object Definitions)
- Siemens Angiography DICOM Private Tags Documentation
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
import numpy as np

# Import DICOM handlers before pydicom for proper registration
# GDCM (recommended for compressed DICOM)
try:
    import gdcm
    GDCM_AVAILABLE = True
except ImportError:
    GDCM_AVAILABLE = False

# pylibjpeg (alternative for JPEG compressed)
try:
    import pylibjpeg
    PYLIBJPEG_AVAILABLE = True
except ImportError:
    PYLIBJPEG_AVAILABLE = False

# pylibjpeg-libjpeg (for JPEG baseline)
try:
    import libjpeg
    LIBJPEG_AVAILABLE = True
except ImportError:
    LIBJPEG_AVAILABLE = False

# pylibjpeg-openjpeg (for JPEG 2000)
try:
    import openjpeg
    OPENJPEG_AVAILABLE = True
except ImportError:
    OPENJPEG_AVAILABLE = False

try:
    import pydicom
    from pydicom.filereader import dcmread
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DicomMetadata:
    """
    DICOM metadata container for coronary angiography.

    Contains patient/study info, image parameters, and calibration data.
    Critical for QCA: pixel_spacing determines real-world measurements.
    """
    # Patient info (anonymizable)
    patient_id: str = "Anonymous"
    patient_name: str = "Anonymous"

    # Study info
    study_date: str = ""
    study_time: str = ""
    study_description: str = ""
    series_description: str = ""

    # Image parameters
    modality: str = "XA"  # X-ray Angiography
    manufacturer: str = ""
    rows: int = 0
    columns: int = 0
    num_frames: int = 1
    bits_stored: int = 12

    # CRITICAL: Pixel spacing for QCA calculations
    # Format: [row_spacing, col_spacing] in mm/pixel
    pixel_spacing: Optional[List[float]] = None
    pixel_spacing_source: Optional[str] = None  # "PixelSpacing", "ImagerPixelSpacing", etc.

    # Frame rate
    frame_rate: float = 15.0  # FPS
    frame_time: Optional[float] = None  # ms per frame

    # Angiography angles
    primary_angle: Optional[float] = None    # LAO/RAO angle
    secondary_angle: Optional[float] = None  # CRA/CAU angle
    angle_label: Optional[str] = None

    # ECG info
    has_ecg: bool = False
    ecg_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


@dataclass
class DicomData:
    """Container for loaded DICOM data."""
    metadata: DicomMetadata
    pixel_array: np.ndarray  # Shape: (num_frames, rows, cols)
    raw_dicom: Any = None  # Original pydicom dataset


class DicomHandler:
    """
    Handler for DICOM coronary angiography files.

    Responsibilities:
    - Load DICOM files (with compression support)
    - Extract and normalize frames
    - Parse metadata including pixel spacing
    - Detect ECG availability

    Usage:
        handler = DicomHandler()
        success, error = handler.load("/path/to/dicom")
        if success:
            frame = handler.get_frame(0)
            metadata = handler.get_metadata()
    """

    def __init__(self):
        """Initialize DICOM handler."""
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM processing")

        self._dicom_data: Optional[DicomData] = None
        self._frame_cache: Dict[int, np.ndarray] = {}

        # Log available handlers
        handlers = []
        if GDCM_AVAILABLE:
            handlers.append("gdcm")
        if PYLIBJPEG_AVAILABLE:
            handlers.append("pylibjpeg")
        if LIBJPEG_AVAILABLE:
            handlers.append("libjpeg")
        if OPENJPEG_AVAILABLE:
            handlers.append("openjpeg")
        logger.info(f"DicomHandler initialized. Decompression handlers: {handlers or 'none (uncompressed only)'}")

    @property
    def is_loaded(self) -> bool:
        """Check if a DICOM file is currently loaded."""
        return self._dicom_data is not None

    def load(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Load a DICOM file.

        Args:
            file_path: Path to DICOM file

        Returns:
            Tuple of (success, error_message)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False, f"File not found: {file_path}"

        # Clear previous data
        self._dicom_data = None
        self._frame_cache.clear()

        try:
            # Load DICOM with force=True to handle files without proper headers
            dcm = pydicom.dcmread(str(file_path), force=True)

            # Extract pixel array
            pixel_array = self._extract_pixel_array(dcm)
            if pixel_array is None:
                return False, "Failed to extract pixel data from DICOM"

            # Extract metadata
            metadata = self._extract_metadata(dcm, pixel_array.shape)

            # Store data
            self._dicom_data = DicomData(
                metadata=metadata,
                pixel_array=pixel_array,
                raw_dicom=dcm
            )

            logger.info(
                f"Loaded DICOM: {file_path.name}, "
                f"Shape: {pixel_array.shape}, "
                f"Frames: {metadata.num_frames}, "
                f"FPS: {metadata.frame_rate}"
            )

            return True, None

        except Exception as e:
            error_msg = f"Failed to load DICOM: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def _extract_pixel_array(self, dcm) -> Optional[np.ndarray]:
        """
        Extract and normalize pixel array from DICOM.

        Handles various DICOM formats:
        - Single frame (2D)
        - Multi-frame grayscale (3D)
        - Multi-frame RGB (4D) -> converted to grayscale

        Returns:
            Normalized pixel array with shape (num_frames, rows, cols)
        """
        try:
            pixel_array = dcm.pixel_array

            # Handle different array dimensions
            if len(pixel_array.shape) == 4:
                # RGB multi-frame: (frames, rows, cols, channels)
                # Take first channel or convert to grayscale
                pixel_array = pixel_array[:, :, :, 0]
                logger.info("Converted RGB to grayscale")

            elif len(pixel_array.shape) == 3:
                # Already multi-frame grayscale: (frames, rows, cols)
                pass

            elif len(pixel_array.shape) == 2:
                # Single frame: (rows, cols) -> (1, rows, cols)
                pixel_array = np.expand_dims(pixel_array, axis=0)

            return pixel_array

        except Exception as e:
            logger.error(f"Pixel array extraction failed: {e}")
            return None

    def _extract_metadata(self, dcm, pixel_shape: Tuple) -> DicomMetadata:
        """
        Extract comprehensive metadata from DICOM dataset.

        Critical extractions:
        - Pixel spacing (for QCA measurements)
        - Frame rate (for timing)
        - ECG availability (for R-peak sync)
        """

        def safe_get(attr: str, default: Any = "") -> Any:
            """Safely get DICOM attribute."""
            value = getattr(dcm, attr, default)
            return str(value) if value and value != default else default

        metadata = DicomMetadata(
            # Patient/Study
            patient_id=safe_get("PatientID", "Anonymous"),
            patient_name=safe_get("PatientName", "Anonymous"),
            study_date=safe_get("StudyDate"),
            study_time=safe_get("StudyTime"),
            study_description=safe_get("StudyDescription"),
            series_description=safe_get("SeriesDescription"),

            # Image params
            modality=safe_get("Modality", "XA"),
            manufacturer=safe_get("Manufacturer"),
            rows=pixel_shape[1] if len(pixel_shape) > 1 else 0,
            columns=pixel_shape[2] if len(pixel_shape) > 2 else 0,
            num_frames=pixel_shape[0],
            bits_stored=int(getattr(dcm, "BitsStored", 12)),
        )

        # Extract pixel spacing (CRITICAL for QCA)
        metadata.pixel_spacing, metadata.pixel_spacing_source = self._extract_pixel_spacing(dcm)

        # Extract frame rate
        metadata.frame_rate, metadata.frame_time = self._extract_frame_rate(dcm)

        # Extract angiography angles
        self._extract_angles(dcm, metadata)

        # Check ECG availability
        self._check_ecg_availability(dcm, metadata)

        return metadata

    def _extract_pixel_spacing(self, dcm) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Extract pixel spacing with fallback chain.

        Priority:
        1. PixelSpacing (0028,0030)
        2. ImagerPixelSpacing (0018,1164)

        Returns:
            Tuple of (spacing_list, source_name)
        """
        # Try PixelSpacing first
        if hasattr(dcm, "PixelSpacing") and dcm.PixelSpacing:
            try:
                spacing = [float(dcm.PixelSpacing[0])]
                if len(dcm.PixelSpacing) > 1:
                    spacing.append(float(dcm.PixelSpacing[1]))
                else:
                    spacing.append(spacing[0])
                logger.info(f"PixelSpacing: {spacing[0]:.5f} x {spacing[1]:.5f} mm/pixel")
                return spacing, "PixelSpacing"
            except (ValueError, TypeError, IndexError):
                pass

        # Fallback to ImagerPixelSpacing
        if hasattr(dcm, "ImagerPixelSpacing") and dcm.ImagerPixelSpacing:
            try:
                spacing = [float(dcm.ImagerPixelSpacing[0])]
                if len(dcm.ImagerPixelSpacing) > 1:
                    spacing.append(float(dcm.ImagerPixelSpacing[1]))
                else:
                    spacing.append(spacing[0])
                logger.info(f"ImagerPixelSpacing: {spacing[0]:.5f} x {spacing[1]:.5f} mm/pixel")
                return spacing, "ImagerPixelSpacing"
            except (ValueError, TypeError, IndexError):
                pass

        logger.warning("No pixel spacing found in DICOM - calibration required")
        return None, None

    def _extract_frame_rate(self, dcm) -> Tuple[float, Optional[float]]:
        """
        Extract frame rate with multiple fallbacks.

        Priority:
        1. CineRate
        2. RecommendedDisplayFrameRate
        3. FrameTime (convert from ms)
        4. FrameDelay (convert from ms)
        5. Default: 15 FPS

        Returns:
            Tuple of (frame_rate_fps, frame_time_ms)
        """
        # CineRate (most common for angiography)
        if hasattr(dcm, "CineRate") and dcm.CineRate:
            fps = float(dcm.CineRate)
            return fps, 1000.0 / fps

        # RecommendedDisplayFrameRate
        if hasattr(dcm, "RecommendedDisplayFrameRate") and dcm.RecommendedDisplayFrameRate:
            fps = float(dcm.RecommendedDisplayFrameRate)
            return fps, 1000.0 / fps

        # FrameTime (milliseconds per frame)
        if hasattr(dcm, "FrameTime") and dcm.FrameTime:
            frame_time = float(dcm.FrameTime)
            fps = 1000.0 / frame_time
            return fps, frame_time

        # FrameDelay (milliseconds)
        if hasattr(dcm, "FrameDelay") and dcm.FrameDelay:
            frame_time = float(dcm.FrameDelay)
            fps = 1000.0 / frame_time
            return fps, frame_time

        # Default fallback
        logger.warning("No frame rate found, using default: 15 FPS")
        return 15.0, 1000.0 / 15.0

    def _extract_angles(self, dcm, metadata: DicomMetadata) -> None:
        """Extract angiography projection angles."""
        if hasattr(dcm, "PositionerPrimaryAngle"):
            try:
                primary = float(dcm.PositionerPrimaryAngle)
                secondary = float(getattr(dcm, "PositionerSecondaryAngle", 0))

                metadata.primary_angle = primary
                metadata.secondary_angle = secondary

                # Generate human-readable label
                # LAO = positive, RAO = negative
                # CRA = positive, CAU = negative
                lr = "LAO" if primary > 0 else "RAO"
                cc = "CRA" if secondary > 0 else "CAU"
                metadata.angle_label = f"{lr} {abs(primary):.0f}° / {cc} {abs(secondary):.0f}°"

            except (ValueError, TypeError):
                pass

    def _check_ecg_availability(self, dcm, metadata: DicomMetadata) -> None:
        """
        Check for ECG data in DICOM.

        Supports:
        - WaveformSequence (modern DICOM)
        - Siemens curved ECG (legacy, group 50xx)
        - Private tags
        """
        # Modern waveform sequence
        if hasattr(dcm, "WaveformSequence"):
            metadata.has_ecg = True
            metadata.ecg_type = "WaveformSequence"
            return

        # Legacy Siemens curve data (50xx group)
        for group in range(0x5000, 0x5020, 0x0002):
            curve_data_tag = (group, 0x3000)
            if curve_data_tag in dcm:
                metadata.has_ecg = True
                metadata.ecg_type = f"SiemensCurve_0x{group:04X}"
                return

    def get_frame(
        self,
        frame_index: int,
        normalize: bool = True,
        target_dtype: np.dtype = np.uint8
    ) -> Optional[np.ndarray]:
        """
        Get a specific frame from the loaded DICOM.

        Args:
            frame_index: Zero-based frame index
            normalize: If True, normalize to target_dtype range
            target_dtype: Target data type (default: uint8)

        Returns:
            Frame as numpy array, or None if invalid
        """
        if not self.is_loaded:
            logger.warning("No DICOM loaded")
            return None

        num_frames = self._dicom_data.pixel_array.shape[0]
        if frame_index < 0 or frame_index >= num_frames:
            logger.warning(f"Frame index {frame_index} out of range [0, {num_frames})")
            return None

        # Check cache
        cache_key = (frame_index, normalize, target_dtype)
        if frame_index in self._frame_cache:
            return self._frame_cache[frame_index]

        # Get raw frame
        frame = self._dicom_data.pixel_array[frame_index].copy()

        # Normalize if requested
        if normalize:
            frame = self._normalize_frame(frame, target_dtype)

        # Cache the result
        self._frame_cache[frame_index] = frame

        return frame

    def _normalize_frame(self, frame: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
        """
        Normalize frame to target data type range.

        Uses min-max normalization to preserve contrast.
        """
        if frame.dtype == target_dtype:
            return frame

        frame_min = np.min(frame)
        frame_max = np.max(frame)

        if frame_max > frame_min:
            # Get target range
            if np.issubdtype(target_dtype, np.integer):
                info = np.iinfo(target_dtype)
                target_max = info.max
            else:
                target_max = 1.0

            normalized = ((frame - frame_min) / (frame_max - frame_min) * target_max)
            return normalized.astype(target_dtype)
        else:
            return np.zeros_like(frame, dtype=target_dtype)

    def get_all_frames(self, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Get all frames as a numpy array.

        Returns:
            Array of shape (num_frames, rows, cols)
        """
        if not self.is_loaded:
            return None

        if normalize:
            frames = []
            for i in range(self._dicom_data.pixel_array.shape[0]):
                frames.append(self.get_frame(i, normalize=True))
            return np.array(frames)
        else:
            return self._dicom_data.pixel_array.copy()

    def get_metadata(self) -> Optional[DicomMetadata]:
        """Get metadata of loaded DICOM."""
        if not self.is_loaded:
            return None
        return self._dicom_data.metadata

    def get_num_frames(self) -> int:
        """Get number of frames in loaded DICOM."""
        if not self.is_loaded:
            return 0
        return self._dicom_data.pixel_array.shape[0]

    def get_raw_dicom(self) -> Optional[Any]:
        """Get raw pydicom dataset (for ECG extraction, etc.)."""
        if not self.is_loaded:
            return None
        return self._dicom_data.raw_dicom

    def clear(self) -> None:
        """Clear loaded data and cache."""
        self._dicom_data = None
        self._frame_cache.clear()
        logger.info("DicomHandler cleared")


# Singleton instance for API usage
_handler_instance: Optional[DicomHandler] = None


def get_handler() -> DicomHandler:
    """Get singleton DicomHandler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = DicomHandler()
    return _handler_instance
