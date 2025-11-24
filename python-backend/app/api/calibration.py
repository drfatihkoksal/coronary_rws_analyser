"""
Calibration API

Endpoints for pixel-to-millimeter calibration.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger

router = APIRouter()


class CalibrationInfo(BaseModel):
    """Current calibration information"""
    pixel_spacing: float  # mm per pixel
    source: str  # "dicom" or "manual"
    confidence: str  # "high", "medium", "low"


class ManualCalibrationRequest(BaseModel):
    """Manual catheter-based calibration request"""
    catheter_size: str  # "5F", "6F", "7F", "8F"
    point1: List[int]  # [x, y]
    point2: List[int]  # [x, y]
    frame_index: int


# Catheter sizes in mm (diameter)
CATHETER_SIZES = {
    "5F": 1.67,
    "6F": 2.00,
    "7F": 2.33,
    "8F": 2.67
}


@router.get("/current", response_model=CalibrationInfo)
async def get_current():
    """Get current calibration settings"""
    # TODO: Return current calibration
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/from-dicom")
async def calibrate_from_dicom():
    """
    Extract calibration from DICOM ImagerPixelSpacing tag.

    DICOM tag (0018, 1164) contains [row_spacing, column_spacing] in mm.
    """
    logger.info("Extracting calibration from DICOM")

    # TODO: Extract ImagerPixelSpacing from loaded DICOM
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/manual", response_model=CalibrationInfo)
async def manual_calibration(request: ManualCalibrationRequest):
    """
    Perform manual catheter-based calibration.

    Process:
    1. User selects catheter size (5F/6F/7F/8F)
    2. User places two points on catheter
    3. Measure pixel distance between points
    4. Calculate: pixel_spacing = known_diameter / measured_pixels

    This recalculates ALL QCA metrics with new calibration.
    """
    logger.info(f"Manual calibration with {request.catheter_size} catheter")

    if request.catheter_size not in CATHETER_SIZES:
        raise HTTPException(status_code=400, detail=f"Invalid catheter size. Valid: {list(CATHETER_SIZES.keys())}")

    # TODO: Implement manual calibration
    # - Segment catheter around provided points
    # - Measure diameter in pixels
    # - Calculate new pixel_spacing
    # - Trigger QCA/RWS recalculation

    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/health")
async def health():
    """Calibration service health check"""
    return {"status": "healthy", "service": "calibration"}
