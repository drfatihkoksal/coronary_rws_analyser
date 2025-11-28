"""
Calibration API

Endpoints for pixel-to-millimeter calibration using catheter reference.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import numpy as np
import base64

from app.core.calibration_engine import get_engine as get_calibration_engine, CalibrationEngine
from app.core.dicom_handler import get_handler as get_dicom_handler
from app.core.angiopy_segmentation import get_angiopy as get_angiopy_engine
from app.core.centerline_extractor import get_extractor as get_centerline_extractor

logger = logging.getLogger(__name__)

router = APIRouter()


class CalibrationInfo(BaseModel):
    """Current calibration information"""
    pixel_spacing: float  # mm per pixel
    source: str  # "dicom", "manual", or "none"
    confidence: str  # "high", "medium", "low"
    catheter_size: Optional[str] = None


class ManualCalibrationRequest(BaseModel):
    """Manual catheter-based calibration request"""
    catheter_size: str  # "5F", "6F", "7F", "8F", or "custom"
    custom_size_mm: Optional[float] = None  # Required if catheter_size == "custom"
    seed_points: List[List[float]]  # [[x1, y1], [x2, y2], ...] - min 2 points
    frame_index: int
    method: str = "gaussian"  # "gaussian", "parabolic", "threshold"
    n_points: int = 50


class CalibrationResponse(BaseModel):
    """Calibration response"""
    success: bool
    new_pixel_spacing: Optional[float] = None
    old_pixel_spacing: Optional[float] = None
    catheter_size: Optional[str] = None
    known_diameter_mm: Optional[float] = None
    measured_diameter_px: Optional[float] = None
    quality_score: Optional[float] = None
    quality_notes: Optional[List[str]] = None
    error_message: Optional[str] = None


# Catheter sizes in mm (diameter)
CATHETER_SIZES = CalibrationEngine.CATHETER_SIZES


@router.get("/catheter-sizes")
async def get_catheter_sizes() -> Dict[str, float]:
    """Get available catheter sizes"""
    return CalibrationEngine.get_catheter_sizes()


@router.get("/current", response_model=CalibrationInfo)
async def get_current():
    """Get current calibration settings"""
    dicom_handler = get_dicom_handler()
    calibration_engine = get_calibration_engine()

    # Check for manual calibration first
    if calibration_engine.last_calibration and calibration_engine.last_calibration.success:
        return CalibrationInfo(
            pixel_spacing=calibration_engine.last_calibration.new_pixel_spacing,
            source="manual",
            confidence="high" if calibration_engine.last_calibration.quality_score > 0.7 else "medium",
            catheter_size=calibration_engine.last_calibration.catheter_size
        )

    # Check DICOM metadata
    if dicom_handler.is_loaded():
        metadata = dicom_handler.get_metadata()
        if metadata and metadata.pixel_spacing:
            return CalibrationInfo(
                pixel_spacing=metadata.pixel_spacing,
                source="dicom",
                confidence="medium"  # DICOM pixel spacing may not be accurate
            )

    # No calibration available
    return CalibrationInfo(
        pixel_spacing=1.0,
        source="none",
        confidence="low"
    )


@router.post("/from-dicom", response_model=CalibrationInfo)
async def calibrate_from_dicom():
    """
    Extract calibration from DICOM ImagerPixelSpacing tag.

    DICOM tag (0018, 1164) contains [row_spacing, column_spacing] in mm.
    """
    logger.info("Extracting calibration from DICOM")

    dicom_handler = get_dicom_handler()

    if not dicom_handler.is_loaded():
        raise HTTPException(status_code=400, detail="No DICOM file loaded")

    metadata = dicom_handler.get_metadata()

    if not metadata or not metadata.pixel_spacing:
        raise HTTPException(
            status_code=400,
            detail="DICOM file does not contain ImagerPixelSpacing tag"
        )

    return CalibrationInfo(
        pixel_spacing=metadata.pixel_spacing,
        source="dicom",
        confidence="medium"
    )


@router.post("/manual", response_model=CalibrationResponse)
async def manual_calibration(request: ManualCalibrationRequest):
    """
    Perform manual catheter-based calibration.

    Process:
    1. User selects catheter size (5F/6F/7F/8F or custom)
    2. User places seed points on catheter
    3. AngioPy segments catheter
    4. Centerline extracted
    5. Diameter measured at N points
    6. Calculate: pixel_spacing = known_diameter / measured_pixels
    """
    logger.info(f"Manual calibration with {request.catheter_size} catheter")

    # Validate catheter size
    is_valid, error_msg = CalibrationEngine.validate_catheter_size(
        request.catheter_size,
        request.custom_size_mm
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Validate seed points
    if len(request.seed_points) < 2:
        raise HTTPException(status_code=400, detail="At least 2 seed points required")

    # Get frame from DICOM
    dicom_handler = get_dicom_handler()
    if not dicom_handler.is_loaded():
        raise HTTPException(status_code=400, detail="No DICOM file loaded")

    frame = dicom_handler.get_frame(request.frame_index)
    if frame is None:
        raise HTTPException(status_code=400, detail=f"Failed to get frame {request.frame_index}")

    # Get old pixel spacing
    metadata = dicom_handler.get_metadata()
    old_pixel_spacing = metadata.pixel_spacing if metadata else None

    try:
        # Convert seed points to tuples
        seed_points = [(float(p[0]), float(p[1])) for p in request.seed_points]

        # Segment catheter using AngioPy
        angiopy = get_angiopy_engine()
        seg_result = angiopy.segment(frame, seed_points, return_probability=True)

        if seg_result is None or seg_result.mask is None:
            raise HTTPException(status_code=500, detail="Catheter segmentation failed")

        mask = seg_result.mask
        probability_map = seg_result.probability_map

        # Extract centerline
        centerline_extractor = get_centerline_extractor()
        centerline_result = centerline_extractor.extract(mask, method='skeleton')

        if centerline_result is None or len(centerline_result.centerline) < 3:
            raise HTTPException(status_code=500, detail="Failed to extract catheter centerline")

        centerline = np.array(centerline_result.centerline)

        # Calculate calibration
        calibration_engine = get_calibration_engine()
        result = calibration_engine.calculate(
            mask=mask,
            centerline=centerline,
            catheter_size=request.catheter_size,
            custom_size_mm=request.custom_size_mm,
            probability_map=probability_map,
            n_points=request.n_points,
            method=request.method,
            old_pixel_spacing=old_pixel_spacing
        )

        return CalibrationResponse(
            success=result.success,
            new_pixel_spacing=result.new_pixel_spacing if result.success else None,
            old_pixel_spacing=result.old_pixel_spacing,
            catheter_size=result.catheter_size,
            known_diameter_mm=result.known_diameter_mm if result.success else None,
            measured_diameter_px=result.measured_diameter_px if result.success else None,
            quality_score=result.quality_score if result.success else None,
            quality_notes=result.quality_notes if result.success else None,
            error_message=result.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return CalibrationResponse(
            success=False,
            error_message=str(e)
        )


@router.get("/health")
async def health():
    """Calibration service health check"""
    return {"status": "healthy", "service": "calibration"}
