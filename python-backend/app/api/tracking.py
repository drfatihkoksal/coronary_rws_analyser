"""
Vessel Tracking API

Endpoints for ROI tracking across frames using hybrid approach:
- CSRT: ROI bounding box tracking
- Template Matching: Precise seed point localization
- Optical Flow: Fallback for challenging frames

WebSocket endpoint for real-time tracking updates during propagation.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger
from typing import List, Optional, Set
import numpy as np
import asyncio
import json

from app.models.schemas import (
    TrackingInitRequest,
    TrackingRequest,
    TrackingResponse,
    PropagateRequest,
    PropagateStatusResponse,
    BoundingBox,
    HealthResponse,
)
from app.core.tracking_engine import get_engine as get_tracking_engine
from app.core.dicom_handler import get_handler

router = APIRouter()

# Propagation state (for progress tracking)
_propagation_state = {
    "is_running": False,
    "current_frame": 0,
    "total_frames": 0,
    "status": "idle"
}

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for tracking updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        self.active_connections -= disconnected


ws_manager = ConnectionManager()


@router.post("/initialize")
async def initialize_tracker(request: TrackingInitRequest):
    """
    Initialize CSRT tracker for ROI tracking.

    Sets up CSRT tracker for ROI position tracking across frames.

    Args:
        request: Initial frame, ROI, and roi_mode ("fixed_150x150" or "adaptive")
    """
    logger.info(f"Initializing CSRT tracker at frame {request.frame_index}, mode={request.roi_mode}")

    # Get DICOM handler
    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    # Get frame
    frame = handler.get_frame(request.frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    # Convert request to engine format
    roi = (
        request.roi.x,
        request.roi.y,
        request.roi.width,
        request.roi.height
    )

    # Validate roi_mode
    if request.roi_mode not in ("fixed_150x150", "adaptive"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid roi_mode: {request.roi_mode}. Must be 'fixed_150x150' or 'adaptive'"
        )

    # Initialize tracker (CSRT only)
    engine = get_tracking_engine()
    success = engine.initialize(frame=frame, roi=roi, roi_mode=request.roi_mode)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize CSRT tracker"
        )

    # Get actual ROI used (may be adjusted by roi_mode)
    state = engine.get_state()
    actual_roi = state.get("initial_roi", roi)

    return {
        "status": "initialized",
        "frame_index": request.frame_index,
        "roi": {
            "x": actual_roi[0],
            "y": actual_roi[1],
            "width": actual_roi[2],
            "height": actual_roi[3]
        },
        "roi_mode": request.roi_mode
    }


@router.post("/track", response_model=TrackingResponse)
async def track_single_frame(request: TrackingRequest):
    """
    Track ROI to a single frame using CSRT.

    Args:
        request: Target frame index
    """
    logger.debug(f"Tracking to frame {request.frame_index}")

    # Get tracker
    engine = get_tracking_engine()
    if not engine.is_initialized:
        raise HTTPException(
            status_code=400,
            detail="Tracker not initialized. Call /initialize first."
        )

    # Get frame
    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    frame = handler.get_frame(request.frame_index, normalize=True, target_dtype=np.uint8)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to get frame")

    # Track with CSRT
    result = engine.track(frame)

    # Convert to response
    roi = None
    if result.roi:
        roi = BoundingBox(
            x=result.roi[0],
            y=result.roi[1],
            width=result.roi[2],
            height=result.roi[3]
        )

    return TrackingResponse(
        success=result.success,
        frame_index=request.frame_index,
        roi=roi,
        confidence=result.confidence,
        method=result.method,
        error_message=result.error_message
    )


@router.post("/propagate")
async def propagate_tracking(request: PropagateRequest):
    """
    Automatically propagate tracking across frame range.

    Tracks each frame in sequence, updating ROI and seed points.
    Stops automatically when confidence drops below threshold (0.6).

    Args:
        request: Start frame, end frame, direction

    Returns:
        List of tracking results and stop reason
    """
    logger.info(f"Propagating from frame {request.start_frame} to {request.end_frame}")

    global _propagation_state

    # Get tracker
    engine = get_tracking_engine()
    if not engine.is_initialized:
        raise HTTPException(
            status_code=400,
            detail="Tracker not initialized. Call /initialize first."
        )

    # Get handler
    handler = get_handler()
    if not handler.is_loaded:
        raise HTTPException(status_code=404, detail="No DICOM file loaded")

    num_frames = handler.get_num_frames()

    # Validate range
    if request.start_frame < 0 or request.start_frame >= num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Start frame out of bounds [0, {num_frames})"
        )
    if request.end_frame < 0 or request.end_frame >= num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"End frame out of bounds [0, {num_frames})"
        )

    # Determine direction and create frame range
    if request.direction == "forward":
        if request.end_frame < request.start_frame:
            raise HTTPException(
                status_code=400,
                detail="Forward: end_frame must be >= start_frame"
            )
        frames = range(request.start_frame, request.end_frame + 1)
    elif request.direction == "backward":
        if request.end_frame > request.start_frame:
            raise HTTPException(
                status_code=400,
                detail="Backward: end_frame must be <= start_frame"
            )
        frames = range(request.start_frame, request.end_frame - 1, -1)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid direction: {request.direction}"
        )

    # Update state
    _propagation_state = {
        "is_running": True,
        "current_frame": request.start_frame,
        "total_frames": abs(request.end_frame - request.start_frame) + 1,
        "status": "running"
    }

    results = []
    stop_reason = None

    try:
        for frame_idx in frames:
            _propagation_state["current_frame"] = frame_idx

            # Get frame
            frame = handler.get_frame(frame_idx, normalize=True, target_dtype=np.uint8)
            if frame is None:
                stop_reason = f"Failed to get frame {frame_idx}"
                break

            # Track
            result = engine.track(frame)

            # Store result
            roi = None
            if result.roi:
                roi = {
                    "x": result.roi[0],
                    "y": result.roi[1],
                    "width": result.roi[2],
                    "height": result.roi[3]
                }

            results.append({
                "frame_index": frame_idx,
                "success": result.success,
                "roi": roi,
                "confidence": result.confidence,
                "method": result.method
            })

            # Check if tracking failed
            if not result.success:
                stop_reason = f"Tracking failed at frame {frame_idx}: {result.error_message}"
                break

            # Check confidence threshold
            if engine.should_stop():
                stop_reason = f"Confidence dropped below threshold at frame {frame_idx}"
                break

    finally:
        _propagation_state["is_running"] = False
        _propagation_state["status"] = "completed" if stop_reason is None else "stopped"

    return {
        "results": results,
        "num_frames_tracked": len(results),
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason
    }


@router.get("/status", response_model=PropagateStatusResponse)
async def get_status():
    """Get current tracking/propagation status."""
    engine = get_tracking_engine()

    total = _propagation_state["total_frames"]
    current = _propagation_state["current_frame"]
    progress = (current / total * 100) if total > 0 else 0

    return PropagateStatusResponse(
        is_running=_propagation_state["is_running"],
        current_frame=current,
        total_frames=total,
        progress=progress,
        status=_propagation_state["status"]
    )


@router.get("/state")
async def get_tracker_state():
    """Get detailed tracker state for debugging."""
    engine = get_tracking_engine()
    return engine.get_state()


@router.post("/reset")
async def reset_tracker():
    """Reset tracker state."""
    logger.info("Resetting tracker")

    global _propagation_state
    _propagation_state = {
        "is_running": False,
        "current_frame": 0,
        "total_frames": 0,
        "status": "idle"
    }

    engine = get_tracking_engine()
    engine.reset()

    return {"status": "reset"}


@router.get("/health", response_model=HealthResponse)
async def health():
    """Tracking service health check."""
    return HealthResponse(
        status="healthy",
        service="tracking",
        version="1.0.0"
    )


@router.websocket("/ws")
async def websocket_tracking(websocket: WebSocket):
    """
    WebSocket endpoint for real-time tracking updates.

    Messages sent to client:
    - {"type": "connected"}: Connection established
    - {"type": "frame_update", "frame_index": N, "confidence": 0.X, ...}: Per-frame updates
    - {"type": "progress", "current": N, "total": M, "percent": X}: Progress updates
    - {"type": "completed", "results": [...], "stop_reason": "..."}: Propagation complete
    - {"type": "error", "message": "..."}: Error occurred

    Client can send:
    - {"action": "propagate", "start_frame": N, "end_frame": M, "direction": "forward"|"backward"}
    - {"action": "stop"}: Stop propagation
    """
    await ws_manager.connect(websocket)

    try:
        # Send connection confirmation
        await websocket.send_json({"type": "connected", "message": "WebSocket connected"})

        while True:
            # Wait for messages from client
            data = await websocket.receive_json()

            action = data.get("action")

            if action == "propagate":
                # Start propagation with WebSocket updates
                await _propagate_with_websocket(
                    websocket=websocket,
                    start_frame=data.get("start_frame", 0),
                    end_frame=data.get("end_frame", 0),
                    direction=data.get("direction", "forward")
                )

            elif action == "stop":
                # Stop current propagation
                global _propagation_state
                if _propagation_state["is_running"]:
                    _propagation_state["status"] = "stopping"
                    await websocket.send_json({
                        "type": "info",
                        "message": "Stop requested"
                    })

            elif action == "ping":
                # Heartbeat
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


async def _propagate_with_websocket(
    websocket: WebSocket,
    start_frame: int,
    end_frame: int,
    direction: str
):
    """
    Propagate tracking with real-time WebSocket updates.
    """
    global _propagation_state

    # Get tracker
    engine = get_tracking_engine()
    if not engine.is_initialized:
        await websocket.send_json({
            "type": "error",
            "message": "Tracker not initialized"
        })
        return

    # Get handler
    handler = get_handler()
    if not handler.is_loaded:
        await websocket.send_json({
            "type": "error",
            "message": "No DICOM file loaded"
        })
        return

    num_frames = handler.get_num_frames()

    # Validate range
    if start_frame < 0 or start_frame >= num_frames:
        await websocket.send_json({
            "type": "error",
            "message": f"Start frame out of bounds [0, {num_frames})"
        })
        return
    if end_frame < 0 or end_frame >= num_frames:
        await websocket.send_json({
            "type": "error",
            "message": f"End frame out of bounds [0, {num_frames})"
        })
        return

    # Determine direction and create frame range
    if direction == "forward":
        if end_frame < start_frame:
            await websocket.send_json({
                "type": "error",
                "message": "Forward: end_frame must be >= start_frame"
            })
            return
        frames = list(range(start_frame, end_frame + 1))
    elif direction == "backward":
        if end_frame > start_frame:
            await websocket.send_json({
                "type": "error",
                "message": "Backward: end_frame must be <= start_frame"
            })
            return
        frames = list(range(start_frame, end_frame - 1, -1))
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Invalid direction: {direction}"
        })
        return

    total_frames = len(frames)

    # Update state
    _propagation_state = {
        "is_running": True,
        "current_frame": start_frame,
        "total_frames": total_frames,
        "status": "running"
    }

    # Send start message
    await websocket.send_json({
        "type": "started",
        "start_frame": start_frame,
        "end_frame": end_frame,
        "direction": direction,
        "total_frames": total_frames
    })

    results = []
    stop_reason = None

    try:
        for i, frame_idx in enumerate(frames):
            # Check if stop requested
            if _propagation_state["status"] == "stopping":
                stop_reason = "User requested stop"
                break

            _propagation_state["current_frame"] = frame_idx

            # Get frame
            frame = handler.get_frame(frame_idx, normalize=True, target_dtype=np.uint8)
            if frame is None:
                stop_reason = f"Failed to get frame {frame_idx}"
                await websocket.send_json({
                    "type": "error",
                    "message": stop_reason
                })
                break

            # Track
            result = engine.track(frame)

            # Build result dict
            roi = None
            if result.roi:
                roi = {
                    "x": result.roi[0],
                    "y": result.roi[1],
                    "width": result.roi[2],
                    "height": result.roi[3]
                }

            frame_result = {
                "frame_index": frame_idx,
                "success": result.success,
                "roi": roi,
                "confidence": result.confidence,
                "method": result.method
            }

            results.append(frame_result)

            # Send frame update
            await websocket.send_json({
                "type": "frame_update",
                **frame_result
            })

            # Send progress update
            progress_percent = ((i + 1) / total_frames) * 100
            await websocket.send_json({
                "type": "progress",
                "current": i + 1,
                "total": total_frames,
                "percent": progress_percent,
                "current_frame": frame_idx
            })

            # Check if tracking failed
            if not result.success:
                stop_reason = f"Tracking failed at frame {frame_idx}: {result.error_message}"
                break

            # Check confidence threshold
            if engine.should_stop():
                stop_reason = f"Confidence dropped below threshold at frame {frame_idx}"
                break

            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"Propagation error: {e}")
        stop_reason = str(e)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

    finally:
        _propagation_state["is_running"] = False
        _propagation_state["status"] = "completed" if stop_reason is None else "stopped"

    # Send completion message
    await websocket.send_json({
        "type": "completed",
        "num_frames_tracked": len(results),
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "results": results
    })
