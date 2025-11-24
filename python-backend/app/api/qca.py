"""
Quantitative Coronary Analysis (QCA) API

Endpoints for diameter measurement and stenosis calculation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

router = APIRouter()


class QCARequest(BaseModel):
    """QCA calculation request"""
    frame_index: int
    centerline: List[List[float]]  # List of [x, y] coordinates
    mask: str  # Base64 encoded mask
    pixel_spacing: float  # mm per pixel


class QCAMetrics(BaseModel):
    """QCA measurement results"""
    frame_index: int

    # Diameter measurements (mm)
    mld: float  # Minimum Lumen Diameter
    mld_position: int  # Index along centerline
    proximal_rd: float  # Proximal Reference Diameter
    proximal_position: int
    distal_rd: float  # Distal Reference Diameter
    distal_position: int

    # Stenosis metrics
    interpolated_reference: float  # Linear interpolation between Prox/Distal RD
    diameter_stenosis: float  # DS% = (1 - MLD/IntRef) * 100
    lesion_length: Optional[float] = None  # mm

    # Full diameter profile
    diameter_profile: List[float]  # N-point diameter array
    n_points: int


@router.post("/calculate", response_model=QCAMetrics)
async def calculate_qca(request: QCARequest):
    """
    Calculate QCA metrics from centerline and mask.

    Diameter measurement methods (in priority order):
    1. Gaussian FWHM (sub-pixel)
    2. Parabolic fitting
    3. Threshold-based (fallback)

    Args:
        request: Frame index, centerline, mask, and calibration

    Returns:
        Complete QCA metrics including MLD, reference diameters, and DS%
    """
    logger.info(f"Calculating QCA for frame {request.frame_index}")

    # TODO: Implement QCA calculation
    # - Sample N points along centerline (default 50)
    # - Calculate perpendicular diameter at each point
    # - Find MLD (minimum)
    # - Find Proximal RD (max in first N/5)
    # - Find Distal RD (max in last N/5)
    # - Calculate interpolated reference
    # - Calculate DS%

    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/metrics/{frame_index}")
async def get_metrics(frame_index: int):
    """Get cached QCA metrics for a frame"""
    # TODO: Return cached metrics
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/health")
async def health():
    """QCA service health check"""
    return {"status": "healthy", "service": "qca", "accuracy_target": "<3%"}
