"""
Coronary RWS Analyser - Python Backend

FastAPI application for coronary angiography analysis.
Primary focus: Radial Wall Strain (RWS) calculation.

Endpoints:
- /dicom: DICOM file loading and frame extraction
- /segmentation: Vessel segmentation using custom nnU-Net
- /tracking: ROI tracking across frames
- /qca: Quantitative Coronary Analysis metrics
- /rws: Radial Wall Strain calculation
- /calibration: Pixel-to-mm calibration
- /export: Data export (CSV, PDF)
- /ws: WebSocket for real-time updates
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

# Create FastAPI app
app = FastAPI(
    title="Coronary RWS Analyser API",
    description="Backend API for Radial Wall Strain analysis of coronary angiography",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.api import dicom, segmentation, tracking, qca, rws, calibration, export

app.include_router(dicom.router, prefix="/dicom", tags=["DICOM"])
app.include_router(segmentation.router, prefix="/segmentation", tags=["Segmentation"])
app.include_router(tracking.router, prefix="/tracking", tags=["Tracking"])
app.include_router(qca.router, prefix="/qca", tags=["QCA"])
app.include_router(rws.router, prefix="/rws", tags=["RWS"])
app.include_router(calibration.router, prefix="/calibration", tags=["Calibration"])
app.include_router(export.router, prefix="/export", tags=["Export"])


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Coronary RWS Analyser API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "dicom": "ready",
            "segmentation": "ready",
            "tracking": "ready",
            "qca": "ready",
            "rws": "ready"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("=" * 50)
    logger.info("Coronary RWS Analyser API Starting...")
    logger.info("=" * 50)
    logger.info("Endpoints available at http://127.0.0.1:8000")
    logger.info("API documentation at http://127.0.0.1:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Coronary RWS Analyser API Shutting down...")
