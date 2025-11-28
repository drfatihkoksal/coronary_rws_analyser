"""
Radial Wall Strain (RWS) API

PRIMARY FEATURE - Endpoints for RWS calculation.

RWS = (Dmax - Dmin) / Dmax × 100%

Clinical thresholds (Hong et al., 2023):
- Normal: < 8%
- Intermediate: 8-12%
- High risk (vulnerable plaque): > 12-14%
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import Dict, Any

from app.models.schemas import (
    RWSRequest,
    RWSResponse,
    RWSMeasurementResponse,
    RWSSummaryResponse,
    RWSInterpretation,
    HealthResponse,
)
from app.core.rws_engine import get_engine as get_rws_engine, OutlierMethod as RWSOutlierMethod
from app.core.qca_engine import get_engine as get_qca_engine

router = APIRouter()

# In-memory storage for frame QCA metrics (populated by QCA API)
# In production, this would be a proper state manager
_frame_qca_cache: Dict[int, Any] = {}


def set_frame_qca(frame_index: int, metrics: Any):
    """Store QCA metrics for a frame (called by QCA API)."""
    _frame_qca_cache[frame_index] = metrics


def get_frame_qca(frame_index: int) -> Any:
    """Get QCA metrics for a frame."""
    return _frame_qca_cache.get(frame_index)


def clear_qca_cache():
    """Clear QCA cache."""
    _frame_qca_cache.clear()


@router.post("/calculate", response_model=RWSResponse)
async def calculate_rws(request: RWSRequest):
    """
    Calculate Radial Wall Strain for a frame range.

    **PRIMARY FEATURE** of the application.

    Process:
    1. For each measurement position (MLD, Proximal RD, Distal RD):
       a. Find frame with maximum diameter (Dmax)
       b. Find frame with minimum diameter (Dmin)
       c. Calculate RWS = (Dmax - Dmin) / Dmax × 100%
    2. Interpret results based on clinical thresholds

    Args:
        request: Frame range and optional beat number

    Returns:
        RWS measurements at all positions with clinical interpretation
    """
    logger.info(f"Calculating RWS for frames {request.start_frame}-{request.end_frame}")

    # Validate frame range
    if request.start_frame >= request.end_frame:
        raise HTTPException(
            status_code=400,
            detail="start_frame must be less than end_frame"
        )

    # Check if we have QCA data for the frames
    frame_metrics = {}
    for frame_idx in range(request.start_frame, request.end_frame + 1):
        metrics = get_frame_qca(frame_idx)
        if metrics is not None:
            frame_metrics[frame_idx] = metrics

    if len(frame_metrics) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need QCA data for at least 2 frames. Found: {len(frame_metrics)}. "
                   "Run QCA calculation on frames first via /qca/calculate endpoint."
        )

    # Calculate RWS
    try:
        # Map schema OutlierMethod to engine OutlierMethod
        engine_outlier_method = RWSOutlierMethod(request.outlier_method.value)
        rws_engine = get_rws_engine(outlier_method=engine_outlier_method)
        result = rws_engine.calculate(
            frame_metrics=frame_metrics,
            beat_number=request.beat_number
        )

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="RWS calculation failed"
            )

        # Convert to response format
        return RWSResponse(
            mld_rws=RWSMeasurementResponse(
                position="mld",
                dmax=result.mld_rws.dmax,
                dmax_frame=result.mld_rws.dmax_frame,
                dmin=result.mld_rws.dmin,
                dmin_frame=result.mld_rws.dmin_frame,
                rws=result.mld_rws.rws,
                interpretation=RWSInterpretation(result.mld_rws.interpretation)
            ),
            proximal_rws=RWSMeasurementResponse(
                position="proximal",
                dmax=result.proximal_rws.dmax,
                dmax_frame=result.proximal_rws.dmax_frame,
                dmin=result.proximal_rws.dmin,
                dmin_frame=result.proximal_rws.dmin_frame,
                rws=result.proximal_rws.rws,
                interpretation=RWSInterpretation(result.proximal_rws.interpretation)
            ),
            distal_rws=RWSMeasurementResponse(
                position="distal",
                dmax=result.distal_rws.dmax,
                dmax_frame=result.distal_rws.dmax_frame,
                dmin=result.distal_rws.dmin,
                dmin_frame=result.distal_rws.dmin_frame,
                rws=result.distal_rws.rws,
                interpretation=RWSInterpretation(result.distal_rws.interpretation)
            ),
            beat_number=result.beat_number,
            start_frame=result.start_frame,
            end_frame=result.end_frame,
            num_frames=result.num_frames,
            average_rws=result.average_rws
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RWS calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-from-diameters", response_model=RWSResponse)
async def calculate_from_diameters(
    mld_diameters: list[float],
    proximal_diameters: list[float],
    distal_diameters: list[float],
    frame_indices: list[int] = None,
    beat_number: int = None
):
    """
    Calculate RWS directly from diameter arrays.

    Convenience endpoint when QCA metrics are provided directly.

    Args:
        mld_diameters: MLD values for each frame (mm)
        proximal_diameters: Proximal RD values (mm)
        distal_diameters: Distal RD values (mm)
        frame_indices: Frame indices (optional)
        beat_number: Beat number (optional)
    """
    if len(mld_diameters) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 diameter values"
        )

    if len(mld_diameters) != len(proximal_diameters) or len(mld_diameters) != len(distal_diameters):
        raise HTTPException(
            status_code=400,
            detail="All diameter arrays must have same length"
        )

    try:
        rws_engine = get_rws_engine()
        result = rws_engine.calculate_from_diameters(
            mld_diameters=mld_diameters,
            proximal_diameters=proximal_diameters,
            distal_diameters=distal_diameters,
            frame_indices=frame_indices,
            beat_number=beat_number
        )

        if result is None:
            raise HTTPException(status_code=500, detail="RWS calculation failed")

        return RWSResponse(
            mld_rws=RWSMeasurementResponse(
                position="mld",
                dmax=result.mld_rws.dmax,
                dmax_frame=result.mld_rws.dmax_frame,
                dmin=result.mld_rws.dmin,
                dmin_frame=result.mld_rws.dmin_frame,
                rws=result.mld_rws.rws,
                interpretation=RWSInterpretation(result.mld_rws.interpretation)
            ),
            proximal_rws=RWSMeasurementResponse(
                position="proximal",
                dmax=result.proximal_rws.dmax,
                dmax_frame=result.proximal_rws.dmax_frame,
                dmin=result.proximal_rws.dmin,
                dmin_frame=result.proximal_rws.dmin_frame,
                rws=result.proximal_rws.rws,
                interpretation=RWSInterpretation(result.proximal_rws.interpretation)
            ),
            distal_rws=RWSMeasurementResponse(
                position="distal",
                dmax=result.distal_rws.dmax,
                dmax_frame=result.distal_rws.dmax_frame,
                dmin=result.distal_rws.dmin,
                dmin_frame=result.distal_rws.dmin_frame,
                rws=result.distal_rws.rws,
                interpretation=RWSInterpretation(result.distal_rws.interpretation)
            ),
            beat_number=result.beat_number,
            start_frame=result.start_frame,
            end_frame=result.end_frame,
            num_frames=result.num_frames,
            average_rws=result.average_rws
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RWS calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_results():
    """
    Get all calculated RWS results.

    Returns history of RWS calculations performed in this session.
    """
    rws_engine = get_rws_engine()
    results = rws_engine.get_all_results()

    return {
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }


@router.get("/summary", response_model=RWSSummaryResponse)
async def get_summary():
    """
    Get summary statistics across all analyzed beats.

    Provides mean, std, min, max for each measurement position.
    """
    rws_engine = get_rws_engine()
    summary = rws_engine.get_beat_summary()

    if not summary or summary.get("num_beats", 0) == 0:
        raise HTTPException(
            status_code=404,
            detail="No RWS results available. Calculate RWS first."
        )

    return RWSSummaryResponse(
        num_beats=summary["num_beats"],
        mld_rws_mean=summary["mld_rws"]["mean"],
        mld_rws_std=summary["mld_rws"]["std"],
        mld_rws_min=summary["mld_rws"]["min"],
        mld_rws_max=summary["mld_rws"]["max"],
        proximal_rws_mean=summary["proximal_rws"]["mean"],
        distal_rws_mean=summary["distal_rws"]["mean"],
    )


@router.get("/export")
async def export_results():
    """
    Export RWS results for statistical analysis.

    Returns data in format suitable for external analysis tools.
    """
    rws_engine = get_rws_engine()
    export_data = rws_engine.export_for_analysis()

    if not export_data:
        raise HTTPException(
            status_code=404,
            detail="No RWS results to export"
        )

    return {
        "data": export_data,
        "format": "tabular",
        "columns": [
            "beat_index", "beat_number", "start_frame", "end_frame", "num_frames",
            "mld_rws", "mld_dmax", "mld_dmin", "mld_interpretation",
            "proximal_rws", "proximal_dmax", "proximal_dmin",
            "distal_rws", "distal_dmax", "distal_dmin",
            "average_rws"
        ]
    }


@router.post("/clear")
async def clear_results():
    """Clear all stored RWS results."""
    rws_engine = get_rws_engine()
    rws_engine.clear_results()
    clear_qca_cache()

    return {"status": "cleared"}


@router.get("/interpret/{rws_value}")
async def interpret_rws(rws_value: float):
    """
    Interpret a single RWS value.

    Returns clinical interpretation based on Hong et al. 2023 thresholds.

    Args:
        rws_value: RWS percentage to interpret

    Returns:
        Interpretation with clinical description
    """
    if rws_value < 8:
        interpretation = "normal"
        description = "Normal vessel compliance, no significant plaque burden"
        risk_level = "low"
    elif rws_value < 12:
        interpretation = "intermediate"
        description = "Intermediate compliance, further evaluation may be warranted"
        risk_level = "moderate"
    else:
        interpretation = "elevated"
        description = "Elevated RWS, possible vulnerable plaque characteristics"
        risk_level = "high"

    return {
        "rws": rws_value,
        "interpretation": interpretation,
        "description": description,
        "risk_level": risk_level,
        "thresholds": {
            "normal": "<8%",
            "intermediate": "8-12%",
            "elevated": ">12%"
        },
        "reference": "Hong et al., EuroIntervention 2023"
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    """RWS service health check."""
    rws_engine = get_rws_engine()
    num_results = len(rws_engine.get_all_results())

    return HealthResponse(
        status="healthy",
        service="rws",
        version="1.0.0"
    )
