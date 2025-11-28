"""
DICOM Processing API

Endpoints for loading and processing DICOM angiography files.
Supports multi-frame DICOM with ECG extraction.
"""

import base64
import io
import tempfile
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger
from PIL import Image
import numpy as np

from app.models.schemas import (
    DicomMetadataResponse,
    FrameResponse,
    ECGResponse,
    HealthResponse,
)
from app.core.dicom_handler import get_handler
from app.core.ecg_parser import get_parser

router = APIRouter()


@router.post("/load", response_model=DicomMetadataResponse)
async def load_dicom(file: UploadFile = File(...)):
    """
    Load a DICOM file and extract metadata.

    Accepts DICOM file upload, parses it, and prepares frames for extraction.
    The file is temporarily saved to disk for pydicom processing.

    Returns:
        DicomMetadataResponse with file metadata
    """
    logger.info(f"Loading DICOM file: {file.filename}")

    # Save uploaded file to temp location
    temp_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Load DICOM using handler
        handler = get_handler()
        success, error = handler.load(temp_path)

        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to load DICOM: {error}")

        # Get metadata
        metadata = handler.get_metadata()

        if metadata is None:
            raise HTTPException(status_code=500, detail="Failed to extract metadata")

        # Extract ECG if available
        if metadata.has_ecg:
            try:
                parser = get_parser()
                raw_dicom = handler.get_raw_dicom()
                if raw_dicom:
                    parser.extract_from_dicom(raw_dicom)
                    logger.info("ECG data extracted successfully")
            except Exception as e:
                logger.warning(f"ECG extraction failed: {e}")

        logger.info(f"DICOM loaded: {metadata.num_frames} frames, {metadata.rows}x{metadata.columns}")

        return DicomMetadataResponse(
            patient_id=metadata.patient_id,
            patient_name=metadata.patient_name,
            study_date=metadata.study_date,
            study_time=metadata.study_time,
            study_description=metadata.study_description,
            modality=metadata.modality,
            manufacturer=metadata.manufacturer,
            rows=metadata.rows,
            columns=metadata.columns,
            num_frames=metadata.num_frames,
            bits_stored=metadata.bits_stored,
            pixel_spacing=metadata.pixel_spacing,
            pixel_spacing_source=metadata.pixel_spacing_source,
            frame_rate=metadata.frame_rate,
            frame_time=metadata.frame_time,
            primary_angle=metadata.primary_angle,
            secondary_angle=metadata.secondary_angle,
            angle_label=metadata.angle_label,
            has_ecg=metadata.has_ecg,
            ecg_type=metadata.ecg_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DICOM loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@router.get("/metadata", response_model=DicomMetadataResponse)
async def get_metadata():
    """
    Get metadata of currently loaded DICOM file.

    Returns cached metadata from the last loaded file.
    """
    handler = get_handler()

    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    metadata = handler.get_metadata()

    return DicomMetadataResponse(
        patient_id=metadata.patient_id,
        patient_name=metadata.patient_name,
        study_date=metadata.study_date,
        study_time=metadata.study_time,
        study_description=metadata.study_description,
        modality=metadata.modality,
        manufacturer=metadata.manufacturer,
        rows=metadata.rows,
        columns=metadata.columns,
        num_frames=metadata.num_frames,
        bits_stored=metadata.bits_stored,
        pixel_spacing=metadata.pixel_spacing,
        pixel_spacing_source=metadata.pixel_spacing_source,
        frame_rate=metadata.frame_rate,
        frame_time=metadata.frame_time,
        primary_angle=metadata.primary_angle,
        secondary_angle=metadata.secondary_angle,
        angle_label=metadata.angle_label,
        has_ecg=metadata.has_ecg,
        ecg_type=metadata.ecg_type,
    )


@router.get("/frame/{frame_index}", response_model=FrameResponse)
async def get_frame(frame_index: int):
    """
    Get a specific frame as base64 encoded PNG.

    Args:
        frame_index: Zero-based frame index

    Returns:
        FrameResponse with base64 encoded image data
    """
    handler = get_handler()

    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    num_frames = handler.get_num_frames()
    if frame_index < 0 or frame_index >= num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Frame index {frame_index} out of range [0, {num_frames})"
        )

    # Get frame as uint8 numpy array
    frame = handler.get_frame(frame_index, normalize=True, target_dtype=np.uint8)

    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to extract frame")

    # Convert to PNG and base64 encode
    try:
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return FrameResponse(
            frame_index=frame_index,
            data=f"data:image/png;base64,{base64_data}",
            width=frame.shape[1],
            height=frame.shape[0],
        )

    except Exception as e:
        logger.error(f"Frame encoding failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to encode frame")


@router.get("/frames")
async def get_frames_range(start: int = 0, end: int = -1):
    """
    Get multiple frames as base64 encoded PNGs.

    Args:
        start: Start frame index (inclusive)
        end: End frame index (exclusive, -1 for all remaining)

    Returns:
        List of FrameResponse objects
    """
    handler = get_handler()

    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    num_frames = handler.get_num_frames()

    if end == -1:
        end = num_frames

    if start < 0 or start >= num_frames:
        raise HTTPException(status_code=400, detail=f"Invalid start index: {start}")

    if end > num_frames:
        end = num_frames

    frames = []
    for idx in range(start, end):
        frame = handler.get_frame(idx, normalize=True, target_dtype=np.uint8)
        if frame is not None:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            frames.append({
                "frame_index": idx,
                "data": f"data:image/png;base64,{base64_data}",
                "width": frame.shape[1],
                "height": frame.shape[0],
            })

    return {"frames": frames, "total": len(frames)}


@router.get("/ecg", response_model=ECGResponse)
async def get_ecg():
    """
    Get ECG data from loaded DICOM.

    Returns ECG waveform and detected R-peaks for cardiac sync.

    Returns:
        ECGResponse with signal, r_peaks, and metadata
    """
    handler = get_handler()
    parser = get_parser()

    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    # Check if ECG was already extracted
    if parser.ecg_data is None:
        # Try to extract
        raw_dicom = handler.get_raw_dicom()
        if raw_dicom is None:
            raise HTTPException(status_code=404, detail="Raw DICOM data not available")

        success = parser.extract_from_dicom(raw_dicom)
        if not success:
            raise HTTPException(status_code=404, detail="No ECG data found in DICOM")

    # Get display data
    display_data = parser.get_display_data()

    if not display_data:
        raise HTTPException(status_code=404, detail="Failed to format ECG data")

    return ECGResponse(
        signal=display_data.get("signal", []),
        time=display_data.get("time", []),
        r_peaks=display_data.get("r_peaks"),
        sampling_rate=display_data.get("sampling_rate", 1000.0),
        heart_rate=display_data.get("heart_rate"),
        duration=display_data.get("duration", 0.0),
        metadata=display_data.get("metadata", {}),
    )


@router.get("/num_frames")
async def get_num_frames():
    """Get the number of frames in loaded DICOM."""
    handler = get_handler()

    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    return {"num_frames": handler.get_num_frames()}


@router.post("/clear")
async def clear_dicom():
    """Clear loaded DICOM data and free memory."""
    handler = get_handler()
    handler.clear()

    parser = get_parser()
    parser._reset()

    return {"status": "cleared"}


@router.get("/health", response_model=HealthResponse)
async def health():
    """DICOM service health check."""
    handler = get_handler()

    return HealthResponse(
        status="healthy",
        service="dicom",
        version="1.0.0"
    )
