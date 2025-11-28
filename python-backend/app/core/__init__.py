"""
Core Engines Package

Contains the main processing engines for coronary artery analysis:

- DicomHandler: DICOM file loading and frame extraction
- ECGParser: ECG waveform extraction and R-peak detection
- NNUNetInference: Custom nnU-Net segmentation model
- AngioPySegmentation: Seed-guided U-Net segmentation
- YOLOKeypointEngine: Vessel keypoint detection for auto seed points
- CenterlineExtractor: Skeleton-based centerline extraction
- TrackingEngine: Hybrid CSRT + Template + Optical Flow
- QCAEngine: Quantitative Coronary Analysis (diameter profiling)
- RWSEngine: Radial Wall Strain calculation (PRIMARY FEATURE)
"""

# Singleton getters for convenience
from app.core.dicom_handler import get_handler, DicomHandler, DicomMetadata
from app.core.ecg_parser import get_parser, ECGParser
from app.core.centerline_extractor import get_extractor, CenterlineExtractor
from app.core.tracking_engine import get_engine as get_tracking_engine, TrackingEngine, TrackingResult
from app.core.qca_engine import get_engine as get_qca_engine, QCAEngine, QCAMetrics
from app.core.rws_engine import get_engine as get_rws_engine, RWSEngine, RWSResult, RWSMeasurement

# Optional: nnU-Net (requires torch and nnunetv2)
try:
    from app.core.nnunet_inference import get_inference, NNUNetInference, SegmentationResult
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    NNUNetInference = None
    get_inference = None
    SegmentationResult = None

# Optional: AngioPy (requires torch and segmentation-models-pytorch)
try:
    from app.core.angiopy_segmentation import (
        get_angiopy, AngioPySegmentation, AngioPyResult,
        segment_with_yolo, AutoSegmentationResult
    )
    ANGIOPY_AVAILABLE = True
except ImportError:
    ANGIOPY_AVAILABLE = False
    AngioPySegmentation = None
    get_angiopy = None
    AngioPyResult = None
    segment_with_yolo = None
    AutoSegmentationResult = None

# Optional: YOLO Keypoint (requires ultralytics)
try:
    from app.core.yolo_keypoint import get_yolo_keypoint, YOLOKeypointEngine, KeypointResult
    YOLO_KEYPOINT_AVAILABLE = True
except ImportError:
    YOLO_KEYPOINT_AVAILABLE = False
    YOLOKeypointEngine = None
    get_yolo_keypoint = None
    KeypointResult = None

__all__ = [
    # DICOM
    "DicomHandler",
    "DicomMetadata",
    "get_handler",

    # ECG
    "ECGParser",
    "get_parser",

    # Segmentation - nnU-Net
    "NNUNetInference",
    "SegmentationResult",
    "get_inference",
    "NNUNET_AVAILABLE",

    # Segmentation - AngioPy
    "AngioPySegmentation",
    "AngioPyResult",
    "get_angiopy",
    "segment_with_yolo",
    "AutoSegmentationResult",
    "ANGIOPY_AVAILABLE",

    # YOLO Keypoint
    "YOLOKeypointEngine",
    "KeypointResult",
    "get_yolo_keypoint",
    "YOLO_KEYPOINT_AVAILABLE",

    # Centerline
    "CenterlineExtractor",
    "get_extractor",

    # Tracking
    "TrackingEngine",
    "TrackingResult",
    "get_tracking_engine",

    # QCA
    "QCAEngine",
    "QCAMetrics",
    "get_qca_engine",

    # RWS (Primary Feature)
    "RWSEngine",
    "RWSResult",
    "RWSMeasurement",
    "get_rws_engine",
]
