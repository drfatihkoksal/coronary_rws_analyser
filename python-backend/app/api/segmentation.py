"""
Vessel Segmentation API

Endpoints for coronary vessel segmentation with multiple engine support:
- nnU-Net: Custom dual-channel model (ROI-based, no seed points)
- AngioPy: Seed-guided U-Net + InceptionResNetV2

Engine selection:
- Global default via SEGMENTATION_ENGINE environment variable
- Per-request override via 'engine' parameter
"""

import os
import base64
import io
from fastapi import APIRouter, HTTPException
from loguru import logger
import numpy as np
from PIL import Image

from app.models.schemas import (
    SegmentationRequest,
    SegmentationResponse,
    SegmentationEngine,
    HealthResponse,
)
from app.core.dicom_handler import get_handler
from app.core.centerline_extractor import get_extractor
from app.core.nnunet_inference import get_inference, NNUNET_AVAILABLE

# Check AngioPy availability
ANGIOPY_AVAILABLE = False
try:
    from app.core.angiopy_segmentation import get_angiopy, SMP_AVAILABLE, segment_with_yolo
    ANGIOPY_AVAILABLE = SMP_AVAILABLE
except ImportError:
    segment_with_yolo = None

# Check YOLO Keypoint availability
YOLO_KEYPOINT_AVAILABLE = False
try:
    from app.core.yolo_keypoint import get_yolo_keypoint
    YOLO_KEYPOINT_AVAILABLE = True
except ImportError:
    get_yolo_keypoint = None

router = APIRouter()

# Default segmentation engine (can be overridden by environment variable)
DEFAULT_ENGINE = os.environ.get("SEGMENTATION_ENGINE", "nnunet")

# Log availability at module load
logger.info(f"Segmentation engines - nnU-Net: {NNUNET_AVAILABLE}, AngioPy: {ANGIOPY_AVAILABLE}, YOLO: {YOLO_KEYPOINT_AVAILABLE}, Default: {DEFAULT_ENGINE}")


def encode_mask(mask: np.ndarray) -> str:
    """Encode numpy mask to base64 PNG."""
    # Convert binary mask (0/1) to full range (0/255)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    img = Image.fromarray(mask)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


def decode_mask(mask_base64: str) -> np.ndarray:
    """Decode base64 mask to numpy array."""
    if "base64," in mask_base64:
        mask_base64 = mask_base64.split("base64,")[1]

    image_data = base64.b64decode(mask_base64)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


@router.post("/segment", response_model=SegmentationResponse)
async def segment_vessel(request: SegmentationRequest):
    """
    Segment coronary vessel using selected engine.

    Engines:
    - **nnunet**: Custom dual-channel nnU-Net (ROI-based)
      - Requires: roi (x, y, w, h)
      - No seed points needed
      - Pipeline: ROI crop → Gaussian spatial map → Inference → CenterComponentKeeper

    - **angiopy**: Seed-guided U-Net + InceptionResNetV2
      - Requires: seed_points (2-10 points, ordered proximal→distal)
      - Points encoded as 4x4 squares in RGB channels
      - Ref: Petersen et al., Int J Cardiol 2024

    Args:
        request: SegmentationRequest with engine selection and parameters

    Returns:
        Binary mask and optional probability map (base64 encoded)
    """
    engine = request.engine.value
    logger.info(f"Segmenting frame {request.frame_index} with engine={engine}")

    # Get DICOM handler
    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    # Get frame
    frame = handler.get_frame(request.frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    try:
        if engine == "nnunet":
            return await _segment_nnunet(request, frame)
        elif engine == "angiopy":
            return await _segment_angiopy(request, frame)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown engine: {engine}. Use 'nnunet' or 'angiopy'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _segment_nnunet(request: SegmentationRequest, frame: np.ndarray) -> SegmentationResponse:
    """Segment using nnU-Net engine."""
    if not NNUNET_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="nnU-Net not available. Install nnunetv2 and PyTorch."
        )

    # Convert ROI if provided
    roi = None
    if request.roi:
        roi = (request.roi[0], request.roi[1], request.roi[2], request.roi[3])

    # Run segmentation
    inference = get_inference()
    result = inference.segment(
        image=frame,
        roi=roi,
        return_probability=request.return_probability
    )

    # Encode mask
    mask_base64 = encode_mask(result.mask)

    # Encode probability map if available
    prob_base64 = None
    if result.probability_map is not None:
        prob_uint8 = (result.probability_map * 255).astype(np.uint8)
        prob_base64 = encode_mask(prob_uint8)

    return SegmentationResponse(
        frame_index=request.frame_index,
        mask=mask_base64,
        probability_map=prob_base64,
        width=result.mask.shape[1],
        height=result.mask.shape[0],
        roi_used=list(result.roi_used) if result.roi_used else None,
        engine_used="nnunet"
    )


async def _segment_angiopy(request: SegmentationRequest, frame: np.ndarray) -> SegmentationResponse:
    """Segment using AngioPy engine."""
    if not ANGIOPY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AngioPy not available. Install segmentation-models-pytorch and timm."
        )

    # Validate seed points
    if not request.seed_points or len(request.seed_points) < 2:
        raise HTTPException(
            status_code=400,
            detail="AngioPy requires at least 2 seed points (ordered proximal→distal)"
        )

    # Convert seed points to list of tuples
    seed_points = [(p.x, p.y) for p in request.seed_points]

    logger.info(f"AngioPy segmentation with {len(seed_points)} seed points")

    # Run segmentation
    angiopy = get_angiopy()
    result = angiopy.segment(
        image=frame,
        seed_points=seed_points,
        return_probability=request.return_probability
    )

    # Encode mask
    mask_base64 = encode_mask(result.mask)

    # Encode probability map if available
    prob_base64 = None
    if result.probability_map is not None:
        prob_uint8 = (result.probability_map * 255).astype(np.uint8)
        prob_base64 = encode_mask(prob_uint8)

    return SegmentationResponse(
        frame_index=request.frame_index,
        mask=mask_base64,
        probability_map=prob_base64,
        width=result.mask.shape[1],
        height=result.mask.shape[0],
        engine_used="angiopy",
        num_seed_points=result.num_seed_points
    )


@router.post("/segment-simple")
async def segment_simple(
    frame_index: int,
    engine: str = "nnunet",
    x: int = 0,
    y: int = 0,
    w: int = 0,
    h: int = 0
):
    """
    Simple segmentation endpoint with query parameters.

    For quick testing without complex request body.
    Only supports nnU-Net (ROI-based) - use POST /segment for AngioPy.
    """
    roi = (x, y, w, h) if w > 0 and h > 0 else None

    # Get handler
    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    frame = handler.get_frame(frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    if engine == "nnunet":
        if not NNUNET_AVAILABLE:
            empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            return {
                "frame_index": frame_index,
                "mask": encode_mask(empty_mask),
                "engine": "nnunet",
                "message": "nnU-Net not available, returning empty mask"
            }

        try:
            inference = get_inference()
            result = inference.segment(image=frame, roi=roi, return_probability=False)

            return {
                "frame_index": frame_index,
                "mask": encode_mask(result.mask),
                "roi_used": list(result.roi_used) if result.roi_used else None,
                "engine": "nnunet"
            }

        except Exception as e:
            logger.error(f"nnU-Net segmentation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Engine '{engine}' not supported in simple endpoint. Use POST /segment for AngioPy."
        )


@router.post("/centerline")
async def extract_centerline(
    frame_index: int,
    mask_base64: str = None,
    method: str = "skeleton",
    num_seeds: int = 3
):
    """
    Extract centerline from segmentation mask.

    Methods:
    - "skeleton": Fast skeleton-based (recommended)
    - "distance": Sub-pixel accurate via distance transform
    - "mcp": Minimum cost path through seed points

    Args:
        frame_index: Frame index
        mask_base64: Base64 encoded mask (optional, uses last segmentation if not provided)
        method: Extraction method
        num_seeds: Number of seed points to generate from centerline

    Returns:
        Centerline coordinates and seed points
    """
    logger.info(f"Extracting centerline for frame {frame_index}, method={method}")

    extractor = get_extractor()

    # Get mask
    if mask_base64:
        mask = decode_mask(mask_base64)
    else:
        # Use last segmentation result
        if extractor.last_diameter_map is not None:
            mask = (extractor.last_diameter_map > 0).astype(np.uint8)
        else:
            raise HTTPException(
                status_code=400,
                detail="No mask provided and no previous segmentation available"
            )

    try:
        # Extract centerline
        centerline = extractor.extract(mask, method=method)

        if len(centerline) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract centerline - empty mask or no vessel found"
            )

        # Generate seed points from centerline
        seed_points = extractor.generate_seed_points(centerline, num_seeds=num_seeds)

        # Smooth centerline with B-spline
        smoothed = extractor.smooth_bspline(centerline)

        # Convert to {x, y} format for frontend
        centerline_xy = [{"x": float(x), "y": float(y)} for y, x in smoothed]
        seeds_xy = [{"x": float(x), "y": float(y)} for x, y in seed_points]

        return {
            "frame_index": frame_index,
            "centerline": centerline_xy,
            "seed_points": seeds_xy,
            "num_points": len(centerline_xy),
            "method": method
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Centerline extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-and-extract")
async def segment_and_extract_centerline(
    frame_index: int,
    engine: str = "nnunet",
    x: int = 0,
    y: int = 0,
    w: int = 0,
    h: int = 0,
    centerline_method: str = "skeleton",
    num_seeds: int = 3
):
    """
    Combined segmentation and centerline extraction.

    Convenience endpoint for complete vessel analysis in one call.
    Only supports nnU-Net (ROI-based) - use separate endpoints for AngioPy.

    Returns:
        mask, probability_map, centerline, seed_points
    """
    logger.info(f"Segment and extract for frame {frame_index}, engine={engine}")

    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    frame = handler.get_frame(frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    roi = (x, y, w, h) if w > 0 and h > 0 else None

    # Segmentation
    mask = None
    probability_map = None

    error_detail = None
    if engine == "nnunet":
        if not NNUNET_AVAILABLE:
            error_detail = "nnU-Net not available - check if nnunetv2 is installed"
        else:
            try:
                inference = get_inference()
                logger.info(f"Running nnU-Net inference on frame {frame_index}, ROI: {roi}, frame shape: {frame.shape}")
                result = inference.segment(image=frame, roi=roi, return_probability=True)
                mask = result.mask
                probability_map = result.probability_map

                # Debug: check mask content
                mask_sum = mask.sum() if mask is not None else 0
                mask_max = mask.max() if mask is not None else 0
                logger.info(f"Segmentation result - mask shape: {mask.shape if mask is not None else None}, "
                           f"sum: {mask_sum}, max: {mask_max}, unique values: {np.unique(mask) if mask is not None else None}")
            except Exception as e:
                error_detail = f"nnU-Net inference error: {str(e)}"
                logger.error(f"nnU-Net segmentation failed: {e}", exc_info=True)

    if mask is None:
        raise HTTPException(
            status_code=503,
            detail=error_detail or "Segmentation failed - unknown error"
        )

    # Centerline extraction
    extractor = get_extractor()

    try:
        centerline = extractor.extract(mask, method=centerline_method)
        seed_points = extractor.generate_seed_points(centerline, num_seeds=num_seeds)
        smoothed = extractor.smooth_bspline(centerline)

        # Convert coordinates to {x, y} format for frontend
        centerline_xy = [{"x": float(x), "y": float(y)} for y, x in smoothed]
        seeds_xy = [{"x": float(x), "y": float(y)} for x, y in seed_points]

    except Exception as e:
        logger.warning(f"Centerline extraction failed: {e}")
        centerline_xy = []
        seeds_xy = []

    # Encode outputs
    mask_base64 = encode_mask(mask)
    prob_base64 = None
    if probability_map is not None:
        prob_uint8 = (probability_map * 255).astype(np.uint8)
        prob_base64 = encode_mask(prob_uint8)

    return {
        "frame_index": frame_index,
        "mask": mask_base64,
        "probability_map": prob_base64,
        "centerline": centerline_xy,
        "seed_points": seeds_xy,
        "roi_used": list(roi) if roi else None,
        "mask_shape": list(mask.shape),
        "engine": engine
    }


@router.get("/engines")
async def get_available_engines():
    """Get list of available segmentation engines and their status."""
    engines = {
        "nnunet": {
            "available": NNUNET_AVAILABLE,
            "name": "Custom nnU-Net",
            "description": "Dual-channel (image + Gaussian spatial map), ROI-based",
            "requires": "roi (x, y, w, h)",
            "seed_points": False
        },
        "angiopy": {
            "available": ANGIOPY_AVAILABLE,
            "name": "AngioPy",
            "description": "U-Net + InceptionResNetV2, seed point-guided",
            "requires": "seed_points (2-10, ordered proximal→distal)",
            "seed_points": True,
            "reference": "Petersen et al., Int J Cardiol 2024"
        },
        "yolo+angiopy": {
            "available": YOLO_KEYPOINT_AVAILABLE and ANGIOPY_AVAILABLE,
            "name": "YOLO + AngioPy (Auto)",
            "description": "Automatic: YOLO detects 5 keypoints → AngioPy segments",
            "requires": "roi (150x150)",
            "seed_points": "auto-detected",
            "endpoint": "/segmentation/segment-auto"
        }
    }

    # Add model info if available
    if NNUNET_AVAILABLE:
        try:
            inference = get_inference()
            engines["nnunet"]["model_info"] = {
                "dataset_id": inference.dataset_id,
                "dual_channel": inference.use_dual_channel,
                "is_loaded": inference.is_loaded
            }
        except Exception:
            pass

    if ANGIOPY_AVAILABLE:
        try:
            angiopy = get_angiopy()
            engines["angiopy"]["model_info"] = {
                "encoder": "inceptionresnetv2",
                "input_size": 512,
                "is_loaded": angiopy.is_loaded
            }
        except Exception:
            pass

    if YOLO_KEYPOINT_AVAILABLE:
        try:
            yolo = get_yolo_keypoint()
            engines["yolo+angiopy"]["model_info"] = {
                "yolo_model": "YOLOv8-pose",
                "input_size": 150,
                "num_keypoints": 5,
                "is_loaded": yolo.is_loaded
            }
        except Exception:
            pass

    return {
        "default_engine": DEFAULT_ENGINE,
        "engines": engines
    }


@router.get("/model-info")
async def get_model_info():
    """Get detailed information about all segmentation models."""
    info = {
        "nnunet": {
            "available": NNUNET_AVAILABLE,
            "message": "nnU-Net not installed" if not NNUNET_AVAILABLE else "Ready"
        },
        "angiopy": {
            "available": ANGIOPY_AVAILABLE,
            "message": "segmentation-models-pytorch not installed" if not ANGIOPY_AVAILABLE else "Ready"
        }
    }

    if NNUNET_AVAILABLE:
        try:
            inference = get_inference()
            info["nnunet"].update({
                "model_type": "nnU-Net v2",
                "dataset_id": inference.dataset_id,
                "dual_channel": inference.use_dual_channel,
                "sigma": inference._sigma,
                "target_size": inference._target_size,
                "bifurcation_suppression": inference.enable_bifurcation_suppression,
                "is_loaded": inference.is_loaded
            })
        except Exception as e:
            info["nnunet"]["error"] = str(e)

    if ANGIOPY_AVAILABLE:
        try:
            angiopy = get_angiopy()
            info["angiopy"].update({
                "model_type": "U-Net + InceptionResNetV2",
                "encoder": angiopy.ENCODER_NAME,
                "input_size": angiopy.INPUT_SIZE,
                "n_classes": angiopy.N_CLASSES,
                "is_loaded": angiopy.is_loaded,
                "model_path": str(angiopy.model_path)
            })
        except Exception as e:
            info["angiopy"]["error"] = str(e)

    return info


@router.get("/health", response_model=HealthResponse)
async def health():
    """Segmentation service health check."""
    # Service is healthy if at least one engine is available
    any_available = NNUNET_AVAILABLE or ANGIOPY_AVAILABLE

    if any_available:
        status = "healthy"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        service="segmentation",
        version="1.2.0"  # Bumped for YOLO keypoint support
    )


# =============================================================================
# YOLO Keypoint Detection Endpoints
# =============================================================================

@router.post("/detect-keypoints")
async def detect_keypoints(
    frame_index: int,
    roi_x: int,
    roi_y: int,
    roi_w: int = 150,
    roi_h: int = 150
):
    """
    Detect vessel keypoints using YOLO within ROI.

    Detects 5 keypoints along the vessel centerline:
    start → quarter → center → three_quarter → end

    These keypoints can be used as seed points for AngioPy segmentation.

    Args:
        frame_index: Frame index in DICOM
        roi_x: ROI top-left x coordinate
        roi_y: ROI top-left y coordinate
        roi_w: ROI width (default 150)
        roi_h: ROI height (default 150)

    Returns:
        Detected keypoints in IMAGE coordinates (not ROI)
    """
    if not YOLO_KEYPOINT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="YOLO keypoint detection not available. Install ultralytics."
        )

    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    frame = handler.get_frame(frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    roi = (roi_x, roi_y, roi_w, roi_h)

    # Validate ROI bounds
    frame_h, frame_w = frame.shape[:2]
    if roi_x < 0 or roi_y < 0 or roi_x + roi_w > frame_w or roi_y + roi_h > frame_h:
        raise HTTPException(
            status_code=400,
            detail=f"ROI out of bounds: {roi}, frame: {frame_w}x{frame_h}"
        )

    try:
        yolo = get_yolo_keypoint()
        result = yolo.detect_with_roi(frame, roi)

        if not result.success:
            return {
                "frame_index": frame_index,
                "roi": list(roi),
                "success": False,
                "error": result.error_message,
                "keypoints": []
            }

        # Convert keypoints to {x, y} format
        keypoints_xy = [{"x": float(x), "y": float(y)} for x, y in result.keypoints]

        return {
            "frame_index": frame_index,
            "roi": list(roi),
            "success": True,
            "confidence": result.confidence,
            "keypoints": keypoints_xy,
            "num_keypoints": len(keypoints_xy),
            "keypoint_names": ["start", "quarter", "center", "three_quarter", "end"]
        }

    except Exception as e:
        logger.error(f"YOLO keypoint detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-auto")
async def segment_auto(
    frame_index: int,
    roi_x: int,
    roi_y: int,
    roi_w: int = 150,
    roi_h: int = 150,
    return_probability: bool = False
):
    """
    Automatic vessel segmentation using YOLO keypoint detection + AngioPy.

    Pipeline:
    1. Crop ROI (150x150) from frame
    2. Run YOLO keypoint detection → 5 seed points
    3. Run AngioPy segmentation with detected seed points
    4. Return mask in full image coordinates

    This is the recommended endpoint for automatic segmentation during tracking.

    Args:
        frame_index: Frame index in DICOM
        roi_x: ROI top-left x coordinate
        roi_y: ROI top-left y coordinate
        roi_w: ROI width (default 150)
        roi_h: ROI height (default 150)
        return_probability: Also return probability map

    Returns:
        Segmentation mask, detected seed points, and confidence
    """
    if not YOLO_KEYPOINT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="YOLO keypoint detection not available. Install ultralytics."
        )

    if not ANGIOPY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AngioPy not available. Install segmentation-models-pytorch."
        )

    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    frame = handler.get_frame(frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    roi = (roi_x, roi_y, roi_w, roi_h)

    # Validate ROI bounds
    frame_h, frame_w = frame.shape[:2]
    if roi_x < 0 or roi_y < 0 or roi_x + roi_w > frame_w or roi_y + roi_h > frame_h:
        raise HTTPException(
            status_code=400,
            detail=f"ROI out of bounds: {roi}, frame: {frame_w}x{frame_h}"
        )

    logger.info(f"Auto segmentation: frame={frame_index}, ROI={roi}")

    try:
        result = segment_with_yolo(
            image=frame,
            roi=roi,
            return_probability=return_probability
        )

        if not result.success:
            return {
                "frame_index": frame_index,
                "roi": list(roi),
                "success": False,
                "error": result.error_message,
                "mask": encode_mask(np.zeros((frame_h, frame_w), dtype=np.uint8)),
                "seed_points": [],
                "centerline": []
            }

        # Encode mask
        mask_base64 = encode_mask(result.mask)

        # Convert seed points to {x, y} format
        seed_points_xy = [{"x": float(x), "y": float(y)} for x, y in result.seed_points]

        # Convert centerline to {x, y} format
        centerline_xy = [{"x": float(x), "y": float(y)} for x, y in result.centerline]

        response = {
            "frame_index": frame_index,
            "roi": list(roi),
            "success": True,
            "mask": mask_base64,
            "width": result.mask.shape[1],
            "height": result.mask.shape[0],
            "seed_points": seed_points_xy,
            "centerline": centerline_xy,
            "yolo_confidence": result.yolo_confidence,
            "engine_used": "yolo+angiopy"
        }

        return response

    except Exception as e:
        logger.error(f"Auto segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-auto-with-centerline")
async def segment_auto_with_centerline(
    frame_index: int,
    roi_x: int,
    roi_y: int,
    roi_w: int = 150,
    roi_h: int = 150
):
    """
    Complete automatic analysis: YOLO → AngioPy → MCP Centerline.

    Pipeline:
    1. YOLO keypoint detection (5 keypoints: start, quarter, center, 3/4, end)
    2. AngioPy segmentation with detected keypoints
    3. MCP Centerline extraction (Dijkstra from start to end keypoint)

    The centerline is extracted using Minimum Cost Path algorithm between
    YOLO's start and end keypoints, ensuring the path follows the vessel center.

    Args:
        frame_index: Frame index
        roi_x, roi_y, roi_w, roi_h: ROI parameters (150x150 recommended)

    Returns:
        mask, seed_points (YOLO keypoints), centerline (MCP path)
    """
    if not YOLO_KEYPOINT_AVAILABLE or not ANGIOPY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="YOLO and AngioPy required for auto segmentation"
        )

    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    frame = handler.get_frame(frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    roi = (roi_x, roi_y, roi_w, roi_h)

    try:
        # YOLO + AngioPy + MCP Centerline (all in one call)
        seg_result = segment_with_yolo(
            image=frame,
            roi=roi,
            extract_centerline=True  # Uses MCP from start to end keypoint
        )

        if not seg_result.success:
            return {
                "frame_index": frame_index,
                "roi": list(roi),
                "success": False,
                "error": seg_result.error_message,
                "seed_points": [],
                "centerline": []
            }

        # Convert seed points and centerline to {x, y} format
        seed_points_xy = [{"x": float(x), "y": float(y)} for x, y in seg_result.seed_points]
        centerline_xy = [{"x": float(x), "y": float(y)} for x, y in seg_result.centerline]

        return {
            "frame_index": frame_index,
            "roi": list(roi),
            "success": True,
            "mask": encode_mask(seg_result.mask),
            "seed_points": seed_points_xy,
            "centerline": centerline_xy,
            "centerline_method": "mcp",  # Minimum Cost Path
            "yolo_confidence": seg_result.yolo_confidence,
            "engine_used": "yolo+angiopy"
        }

    except Exception as e:
        logger.error(f"Auto segmentation with centerline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
