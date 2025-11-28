"""
Export API

Endpoints for exporting analysis results to CSV, JSON, and PDF formats.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import csv
import json
import os
import tempfile
from datetime import datetime

from app.core.qca_engine import get_engine as get_qca_engine
from app.core.rws_engine import get_engine as get_rws_engine
from app.core.dicom_handler import get_handler as get_dicom_handler
from app.core.calibration_engine import get_engine as get_calibration_engine

logger = logging.getLogger(__name__)

router = APIRouter()

# Temporary export directory
EXPORT_DIR = tempfile.gettempdir()


class ExportRequest(BaseModel):
    """Export request parameters"""
    format: str = "csv"  # "csv", "json", "pdf"
    include_qca: bool = True
    include_rws: bool = True
    include_metadata: bool = True
    include_diameter_profiles: bool = False


class ExportResponse(BaseModel):
    """Export result"""
    success: bool
    filename: Optional[str] = None
    path: Optional[str] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class QCAExportRequest(BaseModel):
    """QCA-specific export request"""
    frame_indices: Optional[List[int]] = None  # None = all frames
    include_diameter_profiles: bool = True
    format: str = "csv"  # "csv" or "json"


class RWSExportRequest(BaseModel):
    """RWS-specific export request"""
    beat_indices: Optional[List[int]] = None  # None = all beats
    include_temporal_data: bool = True
    format: str = "csv"  # "csv" or "json"


@router.post("/qca", response_model=ExportResponse)
async def export_qca(request: QCAExportRequest):
    """
    Export QCA metrics to CSV or JSON.

    Generates:
    - Frame-by-frame QCA metrics (MLD, Proximal RD, Distal RD, DS%, LL)
    - Optional diameter profiles per frame
    """
    logger.info(f"Exporting QCA to {request.format}")

    try:
        qca_engine = get_qca_engine()
        last_metrics = qca_engine.get_last_metrics()

        if last_metrics is None:
            return ExportResponse(
                success=False,
                error_message="No QCA data available. Run QCA calculation first."
            )

        # Get DICOM metadata
        dicom_handler = get_dicom_handler()
        metadata = dicom_handler.get_metadata() if dicom_handler.is_loaded() else None

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "csv":
            filename = f"qca_export_{timestamp}.csv"
            filepath = os.path.join(EXPORT_DIR, filename)

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                headers = [
                    "frame_index", "mld_mm", "proximal_rd_mm", "distal_rd_mm",
                    "interpolated_rd_mm", "diameter_stenosis_percent", "lesion_length_mm",
                    "mean_diameter_mm", "std_diameter_mm", "n_points", "method", "pixel_spacing"
                ]
                if request.include_diameter_profiles:
                    max_points = last_metrics.num_points
                    headers.extend([f"diameter_{i}" for i in range(max_points)])

                writer.writerow(headers)

                # Data row (single frame for now)
                metrics_dict = last_metrics.to_dict()
                row = [
                    0,  # frame_index
                    metrics_dict['mld'],
                    metrics_dict['proximal_rd'],
                    metrics_dict['distal_rd'],
                    metrics_dict['interpolated_rd'],
                    metrics_dict['diameter_stenosis'],
                    metrics_dict.get('lesion_length', 0) or 0,
                    metrics_dict.get('mean_diameter_mm', 0),
                    metrics_dict.get('std_diameter_mm', 0),
                    metrics_dict['num_points'],
                    metrics_dict.get('method', 'gaussian'),
                    metrics_dict['pixel_spacing']
                ]

                if request.include_diameter_profiles:
                    row.extend(metrics_dict['diameter_profile'])

                writer.writerow(row)

            file_size = os.path.getsize(filepath)

            return ExportResponse(
                success=True,
                filename=filename,
                path=filepath,
                format="csv",
                size_bytes=file_size
            )

        elif request.format == "json":
            filename = f"qca_export_{timestamp}.json"
            filepath = os.path.join(EXPORT_DIR, filename)

            export_data = {
                "export_timestamp": timestamp,
                "metadata": {
                    "patient_id": metadata.patient_id if metadata else None,
                    "study_date": metadata.study_date if metadata else None,
                },
                "qca_results": [last_metrics.to_dict()]
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            file_size = os.path.getsize(filepath)

            return ExportResponse(
                success=True,
                filename=filename,
                path=filepath,
                format="json",
                size_bytes=file_size
            )

        else:
            return ExportResponse(
                success=False,
                error_message=f"Unsupported format: {request.format}"
            )

    except Exception as e:
        logger.error(f"QCA export failed: {e}")
        return ExportResponse(
            success=False,
            error_message=str(e)
        )


@router.post("/rws", response_model=ExportResponse)
async def export_rws(request: RWSExportRequest):
    """
    Export RWS results to CSV or JSON.

    Generates:
    - Beat-by-beat RWS values (MLD RWS, Proximal RWS, Distal RWS)
    - Dmax/Dmin frames
    - Quality scores
    - Optional temporal diameter data
    """
    logger.info(f"Exporting RWS to {request.format}")

    try:
        rws_engine = get_rws_engine()
        results = rws_engine.get_all_results()

        if not results:
            return ExportResponse(
                success=False,
                error_message="No RWS data available. Run RWS calculation first."
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "csv":
            filename = f"rws_export_{timestamp}.csv"
            filepath = os.path.join(EXPORT_DIR, filename)

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                headers = [
                    "beat_index", "beat_number", "start_frame", "end_frame", "num_frames",
                    "mld_rws_percent", "proximal_rws_percent", "distal_rws_percent", "average_rws_percent",
                    "mld_dmax_mm", "mld_dmin_mm", "mld_dmax_frame", "mld_dmin_frame",
                    "mld_interpretation", "mld_quality_score",
                    "proximal_dmax_mm", "proximal_dmin_mm",
                    "distal_dmax_mm", "distal_dmin_mm",
                    "overall_quality_score", "outlier_method"
                ]
                writer.writerow(headers)

                # Data rows
                export_data = rws_engine.export_for_analysis()
                for row_data in export_data:
                    row = [
                        row_data.get('beat_index', 0),
                        row_data.get('beat_number', ''),
                        row_data.get('start_frame', 0),
                        row_data.get('end_frame', 0),
                        row_data.get('num_frames', 0),
                        row_data.get('mld_rws', 0),
                        row_data.get('proximal_rws', 0),
                        row_data.get('distal_rws', 0),
                        row_data.get('average_rws', 0),
                        row_data.get('mld_dmax', 0),
                        row_data.get('mld_dmin', 0),
                        row_data.get('mld_dmax_frame', -1),
                        row_data.get('mld_dmin_frame', -1),
                        row_data.get('mld_interpretation', ''),
                        row_data.get('mld_quality_score', 1.0),
                        row_data.get('proximal_dmax', 0),
                        row_data.get('proximal_dmin', 0),
                        row_data.get('distal_dmax', 0),
                        row_data.get('distal_dmin', 0),
                        row_data.get('overall_quality_score', 1.0),
                        row_data.get('outlier_method', 'hampel')
                    ]
                    writer.writerow(row)

            file_size = os.path.getsize(filepath)

            return ExportResponse(
                success=True,
                filename=filename,
                path=filepath,
                format="csv",
                size_bytes=file_size
            )

        elif request.format == "json":
            filename = f"rws_export_{timestamp}.json"
            filepath = os.path.join(EXPORT_DIR, filename)

            # Get metadata
            dicom_handler = get_dicom_handler()
            metadata = dicom_handler.get_metadata() if dicom_handler.is_loaded() else None

            export_data = {
                "export_timestamp": timestamp,
                "metadata": {
                    "patient_id": metadata.patient_id if metadata else None,
                    "study_date": metadata.study_date if metadata else None,
                },
                "rws_results": [r.to_dict() for r in results],
                "summary": rws_engine.get_beat_summary()
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            file_size = os.path.getsize(filepath)

            return ExportResponse(
                success=True,
                filename=filename,
                path=filepath,
                format="json",
                size_bytes=file_size
            )

        else:
            return ExportResponse(
                success=False,
                error_message=f"Unsupported format: {request.format}"
            )

    except Exception as e:
        logger.error(f"RWS export failed: {e}")
        return ExportResponse(
            success=False,
            error_message=str(e)
        )


@router.post("/all", response_model=ExportResponse)
async def export_all(request: ExportRequest):
    """
    Export all analysis data (QCA + RWS + metadata).

    Generates comprehensive export file.
    """
    logger.info(f"Exporting all data to {request.format}")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Collect all data
        export_data: Dict[str, Any] = {
            "export_timestamp": timestamp,
            "export_version": "1.0"
        }

        # DICOM metadata
        if request.include_metadata:
            dicom_handler = get_dicom_handler()
            if dicom_handler.is_loaded():
                metadata = dicom_handler.get_metadata()
                if metadata:
                    export_data["metadata"] = {
                        "patient_id": metadata.patient_id,
                        "study_date": metadata.study_date,
                        "study_description": metadata.study_description,
                        "modality": metadata.modality,
                        "manufacturer": metadata.manufacturer,
                        "rows": metadata.rows,
                        "columns": metadata.columns,
                        "num_frames": metadata.num_frames,
                        "pixel_spacing": metadata.pixel_spacing,
                        "frame_rate": metadata.frame_rate,
                    }

        # Calibration
        calibration_engine = get_calibration_engine()
        if calibration_engine.last_calibration:
            cal = calibration_engine.last_calibration
            export_data["calibration"] = cal.to_dict()

        # QCA data
        if request.include_qca:
            qca_engine = get_qca_engine()
            last_metrics = qca_engine.get_last_metrics()
            if last_metrics:
                export_data["qca"] = last_metrics.to_dict()

        # RWS data
        if request.include_rws:
            rws_engine = get_rws_engine()
            results = rws_engine.get_all_results()
            if results:
                export_data["rws"] = {
                    "results": [r.to_dict() for r in results],
                    "summary": rws_engine.get_beat_summary()
                }

        if request.format == "json":
            filename = f"coronary_analysis_{timestamp}.json"
            filepath = os.path.join(EXPORT_DIR, filename)

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            file_size = os.path.getsize(filepath)

            return ExportResponse(
                success=True,
                filename=filename,
                path=filepath,
                format="json",
                size_bytes=file_size
            )

        elif request.format == "csv":
            # For CSV, we create a summary file
            filename = f"coronary_analysis_{timestamp}.csv"
            filepath = os.path.join(EXPORT_DIR, filename)

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Summary section
                writer.writerow(["=== CORONARY RWS ANALYSIS EXPORT ==="])
                writer.writerow(["Export Time", timestamp])
                writer.writerow([])

                # Metadata
                if "metadata" in export_data:
                    writer.writerow(["=== METADATA ==="])
                    for key, value in export_data["metadata"].items():
                        writer.writerow([key, value])
                    writer.writerow([])

                # QCA Summary
                if "qca" in export_data:
                    writer.writerow(["=== QCA METRICS ==="])
                    qca = export_data["qca"]
                    writer.writerow(["MLD (mm)", qca.get('mld', 0)])
                    writer.writerow(["Proximal RD (mm)", qca.get('proximal_rd', 0)])
                    writer.writerow(["Distal RD (mm)", qca.get('distal_rd', 0)])
                    writer.writerow(["Diameter Stenosis (%)", qca.get('diameter_stenosis', 0)])
                    writer.writerow(["Lesion Length (mm)", qca.get('lesion_length', 0) or 0])
                    writer.writerow([])

                # RWS Summary
                if "rws" in export_data and export_data["rws"].get("summary"):
                    writer.writerow(["=== RWS RESULTS ==="])
                    summary = export_data["rws"]["summary"]
                    if "mld_rws" in summary:
                        writer.writerow(["MLD RWS Mean (%)", summary["mld_rws"].get("mean", 0)])
                        writer.writerow(["MLD RWS Std (%)", summary["mld_rws"].get("std", 0)])
                    if "proximal_rws" in summary:
                        writer.writerow(["Proximal RWS Mean (%)", summary["proximal_rws"].get("mean", 0)])
                    if "distal_rws" in summary:
                        writer.writerow(["Distal RWS Mean (%)", summary["distal_rws"].get("mean", 0)])
                    writer.writerow(["Number of Beats", summary.get("num_beats", 0)])

            file_size = os.path.getsize(filepath)

            return ExportResponse(
                success=True,
                filename=filename,
                path=filepath,
                format="csv",
                size_bytes=file_size
            )

        else:
            return ExportResponse(
                success=False,
                error_message=f"Unsupported format: {request.format}. Use 'csv' or 'json'."
            )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return ExportResponse(
            success=False,
            error_message=str(e)
        )


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download an exported file"""
    filepath = os.path.join(EXPORT_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Determine media type
    if filename.endswith('.csv'):
        media_type = 'text/csv'
    elif filename.endswith('.json'):
        media_type = 'application/json'
    elif filename.endswith('.pdf'):
        media_type = 'application/pdf'
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        filepath,
        media_type=media_type,
        filename=filename
    )


@router.get("/health")
async def health():
    """Export service health check"""
    return {
        "status": "healthy",
        "service": "export",
        "formats": ["csv", "json"],
        "export_dir": EXPORT_DIR
    }
