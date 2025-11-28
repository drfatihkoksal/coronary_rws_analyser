"""
Pydantic Schemas for API Request/Response Models

Contains all data models for API communication.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


# ============================================================================
# DICOM Schemas
# ============================================================================

class DicomMetadataResponse(BaseModel):
    """DICOM file metadata response."""
    patient_id: str = "Anonymous"
    patient_name: str = "Anonymous"
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    modality: str = "XA"
    manufacturer: Optional[str] = None
    rows: int
    columns: int
    num_frames: int
    bits_stored: int = 12
    pixel_spacing: Optional[List[float]] = None
    pixel_spacing_source: Optional[str] = None
    frame_rate: float = 15.0
    frame_time: Optional[float] = None
    primary_angle: Optional[float] = None
    secondary_angle: Optional[float] = None
    angle_label: Optional[str] = None
    has_ecg: bool = False
    ecg_type: Optional[str] = None


class FrameResponse(BaseModel):
    """Single frame response."""
    frame_index: int
    data: str  # Base64 encoded PNG
    width: int
    height: int


class ECGResponse(BaseModel):
    """ECG data response."""
    signal: List[float]
    time: List[float]
    r_peaks: Optional[List[int]] = None  # Sample indices
    sampling_rate: float
    heart_rate: Optional[float] = None
    duration: float
    metadata: Dict[str, Any] = {}


# ============================================================================
# Segmentation Schemas
# ============================================================================

class SegmentationEngine(str, Enum):
    """Available segmentation engines."""
    NNUNET = "nnunet"    # Custom nnU-Net (ROI-based, no seed points)
    ANGIOPY = "angiopy"  # AngioPy (seed point-guided)


class SeedPoint(BaseModel):
    """Seed point for vessel segmentation (AngioPy)."""
    x: float = Field(..., description="X coordinate (pixel)")
    y: float = Field(..., description="Y coordinate (pixel)")


class SegmentationRequest(BaseModel):
    """Segmentation request with engine selection."""
    model_config = {"protected_namespaces": ()}

    frame_index: int = Field(..., ge=0, description="Frame index to segment")
    engine: SegmentationEngine = Field(
        default=SegmentationEngine.NNUNET,
        description="Segmentation engine: 'nnunet' (ROI-based) or 'angiopy' (seed-guided)"
    )

    # nnU-Net parameters (required when engine='nnunet')
    roi: Optional[Tuple[int, int, int, int]] = Field(
        None,
        description="ROI bounding box (x, y, width, height) - required for nnU-Net"
    )

    # AngioPy parameters (required when engine='angiopy')
    seed_points: Optional[List[SeedPoint]] = Field(
        None,
        description="Seed points for AngioPy (2-10 points, ordered proximal→distal)"
    )

    return_probability: bool = Field(
        True,
        description="Return probability map in addition to binary mask"
    )


class SegmentationResponse(BaseModel):
    """Segmentation result."""
    frame_index: int
    mask: str  # Base64 encoded mask
    probability_map: Optional[str] = None  # Base64 encoded
    width: int
    height: int
    roi_used: Optional[List[int]] = None
    engine_used: str = "nnunet"  # Which engine was used
    num_seed_points: Optional[int] = None  # For AngioPy


# ============================================================================
# Tracking Schemas
# ============================================================================

class Point(BaseModel):
    """2D point."""
    x: float
    y: float


class BoundingBox(BaseModel):
    """Bounding box."""
    x: int
    y: int
    width: int
    height: int


class TrackingInitRequest(BaseModel):
    """Initialize tracking request (CSRT-only, ROI tracking)."""
    frame_index: int
    roi: BoundingBox
    roi_mode: str = "fixed_150x150"  # "fixed_150x150" or "adaptive"
    # seed_points removed - using CSRT-only for ROI tracking


class TrackingRequest(BaseModel):
    """Track to next frame request."""
    frame_index: int


class TrackingResponse(BaseModel):
    """Tracking result (CSRT ROI tracking)."""
    success: bool
    frame_index: int
    roi: Optional[BoundingBox] = None
    confidence: float
    method: str = "csrt"
    error_message: Optional[str] = None


class PropagateRequest(BaseModel):
    """Auto-propagate tracking request."""
    start_frame: int
    end_frame: int
    direction: str = "forward"  # "forward", "backward", "both"


class PropagateStatusResponse(BaseModel):
    """Propagation status."""
    is_running: bool
    current_frame: int
    total_frames: int
    progress: float  # 0-100
    status: str  # "running", "completed", "stopped", "failed"


# ============================================================================
# QCA Schemas
# ============================================================================

class QCAMethod(str, Enum):
    """Diameter calculation method for sub-pixel accuracy."""
    GAUSSIAN = "gaussian"     # Gaussian fitting (default, best accuracy)
    PARABOLIC = "parabolic"   # Parabolic fitting (faster)
    THRESHOLD = "threshold"   # Simple thresholding (fastest)


class QCARequest(BaseModel):
    """QCA calculation request."""
    frame_index: int
    num_points: int = 50  # N-point analysis
    method: QCAMethod = Field(
        default=QCAMethod.GAUSSIAN,
        description="Diameter calculation method (gaussian/parabolic/threshold)"
    )


class QCAResponse(BaseModel):
    """QCA metrics response."""
    frame_index: int
    diameter_profile: List[float]  # mm
    mld: float  # Minimum Lumen Diameter (mm)
    mld_index: int
    mld_position: List[float]  # [y, x]
    proximal_rd: float  # Proximal Reference Diameter (mm)
    proximal_rd_index: int
    proximal_rd_position: List[float]
    distal_rd: float  # Distal Reference Diameter (mm)
    distal_rd_index: int
    distal_rd_position: List[float]
    interpolated_rd: float  # Interpolated Reference at MLD
    diameter_stenosis: float  # DS%
    lesion_length: Optional[float] = None  # mm
    pixel_spacing: float
    num_points: int


# ============================================================================
# RWS Schemas (Primary Feature)
# ============================================================================

class OutlierMethod(str, Enum):
    """Outlier detection methods for RWS calculation."""
    NONE = "none"             # No outlier filtering
    HAMPEL = "hampel"         # Hampel filter (default, recommended)
    DOUBLE_HAMPEL = "double_hampel"  # Double-pass Hampel
    IQR = "iqr"               # Interquartile range
    TEMPORAL = "temporal"     # Temporal consistency check


class RWSInterpretation(str, Enum):
    """Clinical interpretation of RWS."""
    NORMAL = "normal"
    INTERMEDIATE = "intermediate"
    ELEVATED = "elevated"


class RWSMeasurementResponse(BaseModel):
    """Single RWS measurement."""
    position: str  # "mld", "proximal", "distal"
    dmax: float  # mm
    dmax_frame: int
    dmin: float  # mm
    dmin_frame: int
    rws: float  # Percentage
    interpretation: RWSInterpretation


class RWSRequest(BaseModel):
    """RWS calculation request."""
    start_frame: int
    end_frame: int
    beat_number: Optional[int] = None
    outlier_method: OutlierMethod = Field(
        default=OutlierMethod.HAMPEL,
        description="Outlier detection method for diameter filtering"
    )


class RWSResponse(BaseModel):
    """RWS calculation response."""
    mld_rws: RWSMeasurementResponse
    proximal_rws: RWSMeasurementResponse
    distal_rws: RWSMeasurementResponse
    beat_number: Optional[int] = None
    start_frame: int
    end_frame: int
    num_frames: int
    average_rws: float


class RWSSummaryResponse(BaseModel):
    """RWS summary across multiple beats."""
    num_beats: int
    mld_rws_mean: float
    mld_rws_std: float
    mld_rws_min: float
    mld_rws_max: float
    proximal_rws_mean: float
    distal_rws_mean: float


# ============================================================================
# Calibration Schemas
# ============================================================================

class CalibrationRequest(BaseModel):
    """Manual calibration request."""
    catheter_size_fr: int  # French size (5, 6, 7, 8)
    measured_diameter_pixels: float


class CalibrationResponse(BaseModel):
    """Calibration result."""
    pixel_spacing: List[float]  # [row, col] mm/pixel
    source: str  # "manual_catheter", "dicom"
    catheter_size_fr: Optional[int] = None


# ============================================================================
# Export Schemas
# ============================================================================

class ExportFormat(str, Enum):
    """Export format options."""
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"


class ExportRequest(BaseModel):
    """Export request."""
    format: ExportFormat
    include_qca: bool = True
    include_rws: bool = True
    include_metadata: bool = True


class ExportResponse(BaseModel):
    """Export result."""
    filename: str
    path: str
    format: str
    size_bytes: int


# ============================================================================
# General Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: Optional[str] = None
