"""
Quantitative Coronary Analysis (QCA) API

Endpoints for diameter measurement and stenosis calculation.
Uses Distance Transform for diameter profiling along centerline.
"""

import base64
import io
from fastapi import APIRouter, HTTPException
from loguru import logger
import numpy as np
from PIL import Image

from app.models.schemas import (
    QCARequest,
    QCAResponse,
    HealthResponse,
)
from app.core.qca_engine import get_engine as get_qca_engine, QCAMetrics
from app.core.centerline_extractor import get_extractor
from app.core.dicom_handler import get_handler

# Import RWS cache helper
from app.api.rws import set_frame_qca

router = APIRouter()


def decode_mask(mask_base64: str) -> np.ndarray:
    """Decode base64 mask to numpy array."""
    # Remove data URL prefix if present
    if "base64," in mask_base64:
        mask_base64 = mask_base64.split("base64,")[1]

    image_data = base64.b64decode(mask_base64)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


@router.post("/calculate", response_model=QCAResponse)
async def calculate_qca(request: QCARequest):
    """
    Calculate QCA metrics from segmentation mask.

    Uses Distance Transform to measure vessel diameter at N points
    along the centerline.

    Measurement positions:
    - MLD: Minimum Lumen Diameter (narrowest point)
    - Proximal RD: Maximum in first N/5 points
    - Distal RD: Maximum in last N/5 points

    Stenosis calculation:
    - Interpolated RD: Linear interpolation between Prox/Distal at MLD position
    - DS% = (1 - MLD / Interpolated_RD) × 100

    Args:
        request: Frame index and number of measurement points

    Returns:
        Complete QCA metrics
    """
    logger.info(f"Calculating QCA for frame {request.frame_index}")

    # Get DICOM handler
    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    # Get pixel spacing for calibration
    metadata = handler.get_metadata()
    pixel_spacing = 1.0  # Default: 1 pixel = 1 unit

    if metadata.pixel_spacing:
        pixel_spacing = metadata.pixel_spacing[0]  # Use row spacing
        logger.info(f"Using pixel spacing: {pixel_spacing} mm/pixel")
    else:
        logger.warning("No pixel spacing in DICOM, using 1.0 (uncalibrated)")

    # Get frame for mask extraction (if mask not provided)
    # In a real workflow, mask comes from segmentation API
    frame = handler.get_frame(request.frame_index)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    # Get or extract centerline
    extractor = get_extractor()

    # For now, use last extracted centerline or extract from session state
    # In production, centerline would be provided from segmentation/tracking
    if extractor.last_centerline is None or len(extractor.last_centerline) == 0:
        raise HTTPException(
            status_code=400,
            detail="No centerline available. Run segmentation first to extract centerline."
        )

    centerline = extractor.last_centerline

    # Get last mask from segmentation (or use last_diameter_map)
    if extractor.last_diameter_map is None:
        raise HTTPException(
            status_code=400,
            detail="No segmentation mask available. Run segmentation first."
        )

    # Create a binary mask from diameter map (non-zero = vessel)
    mask = (extractor.last_diameter_map > 0).astype(np.uint8)

    # Calculate QCA
    try:
        qca_engine = get_qca_engine(num_points=request.num_points, method=request.method.value)
        metrics = qca_engine.calculate(
            mask=mask,
            centerline=centerline,
            pixel_spacing=pixel_spacing
        )

        if metrics is None:
            raise HTTPException(status_code=500, detail="QCA calculation failed")

        # Cache for RWS calculation
        set_frame_qca(request.frame_index, metrics)

        return QCAResponse(
            frame_index=request.frame_index,
            diameter_profile=metrics.diameter_profile,
            mld=metrics.mld,
            mld_index=metrics.mld_index,
            mld_position=list(metrics.mld_position),
            proximal_rd=metrics.proximal_rd,
            proximal_rd_index=metrics.proximal_rd_index,
            proximal_rd_position=list(metrics.proximal_rd_position),
            distal_rd=metrics.distal_rd,
            distal_rd_index=metrics.distal_rd_index,
            distal_rd_position=list(metrics.distal_rd_position),
            interpolated_rd=metrics.interpolated_rd,
            diameter_stenosis=metrics.diameter_stenosis,
            lesion_length=metrics.lesion_length,
            pixel_spacing=metrics.pixel_spacing,
            num_points=metrics.num_points
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QCA calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-with-mask", response_model=QCAResponse)
async def calculate_qca_with_mask(
    frame_index: int,
    mask_base64: str,
    centerline: list[list[float]],
    pixel_spacing: float = 1.0,
    num_points: int = 50
):
    """
    Calculate QCA with provided mask and centerline.

    Alternative endpoint when mask is provided directly.

    Args:
        frame_index: Frame index
        mask_base64: Base64 encoded binary mask
        centerline: List of [y, x] coordinates
        pixel_spacing: mm/pixel calibration
        num_points: Number of measurement points
    """
    logger.info(f"Calculating QCA with provided mask for frame {frame_index}")

    try:
        # Decode mask
        mask = decode_mask(mask_base64)

        # Convert centerline to numpy array
        centerline_array = np.array(centerline)

        if len(centerline_array) < 3:
            raise HTTPException(
                status_code=400,
                detail="Centerline must have at least 3 points"
            )

        # Calculate QCA
        qca_engine = get_qca_engine(num_points=num_points)
        metrics = qca_engine.calculate(
            mask=mask,
            centerline=centerline_array,
            pixel_spacing=pixel_spacing
        )

        if metrics is None:
            raise HTTPException(status_code=500, detail="QCA calculation failed")

        # Cache for RWS
        set_frame_qca(frame_index, metrics)

        return QCAResponse(
            frame_index=frame_index,
            diameter_profile=metrics.diameter_profile,
            mld=metrics.mld,
            mld_index=metrics.mld_index,
            mld_position=list(metrics.mld_position),
            proximal_rd=metrics.proximal_rd,
            proximal_rd_index=metrics.proximal_rd_index,
            proximal_rd_position=list(metrics.proximal_rd_position),
            distal_rd=metrics.distal_rd,
            distal_rd_index=metrics.distal_rd_index,
            distal_rd_position=list(metrics.distal_rd_position),
            interpolated_rd=metrics.interpolated_rd,
            diameter_stenosis=metrics.diameter_stenosis,
            lesion_length=metrics.lesion_length,
            pixel_spacing=metrics.pixel_spacing,
            num_points=metrics.num_points
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QCA calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{frame_index}")
async def get_metrics(frame_index: int):
    """
    Get cached QCA metrics for a frame.

    Returns previously calculated metrics if available.
    """
    from app.api.rws import get_frame_qca

    metrics = get_frame_qca(frame_index)

    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No QCA metrics for frame {frame_index}. Calculate QCA first."
        )

    # Handle both QCAMetrics objects and dicts
    if hasattr(metrics, 'to_dict'):
        return metrics.to_dict()
    return metrics


@router.get("/diameter-at-position")
async def get_diameter_at_position(
    frame_index: int,
    y: float,
    x: float
):
    """
    Get vessel diameter at a specific position.

    Args:
        frame_index: Frame index
        y, x: Position coordinates
    """
    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    # Get pixel spacing
    metadata = handler.get_metadata()
    pixel_spacing = metadata.pixel_spacing[0] if metadata.pixel_spacing else 1.0

    # Get extractor
    extractor = get_extractor()

    if extractor.last_diameter_map is None:
        raise HTTPException(
            status_code=400,
            detail="No segmentation available"
        )

    # Create mask from diameter map
    mask = (extractor.last_diameter_map > 0).astype(np.uint8)

    # Get diameter at position
    qca_engine = get_qca_engine()
    diameter = qca_engine.get_diameter_at_position(
        mask=mask,
        position=(y, x),
        pixel_spacing=pixel_spacing
    )

    return {
        "frame_index": frame_index,
        "position": {"y": y, "x": x},
        "diameter_mm": diameter,
        "pixel_spacing": pixel_spacing
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    """QCA service health check."""
    return HealthResponse(
        status="healthy",
        service="qca",
        version="1.0.0"
    )
