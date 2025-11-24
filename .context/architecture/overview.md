# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CORONARY RWS ANALYSER                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    TAURI SHELL (Rust)                         │  │
│  │  - Window management                                          │  │
│  │  - File system access                                         │  │
│  │  - Native dialogs                                             │  │
│  │  - Python sidecar management                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 REACT FRONTEND (TypeScript)                   │  │
│  │                                                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │   Viewer    │  │  Controls   │  │    Data Panels      │   │  │
│  │  │             │  │             │  │                     │   │  │
│  │  │ VideoLayer  │  │ PlaybackBar │  │ ECGPanel            │   │  │
│  │  │ SegmentLyr  │  │ ToolPanel   │  │ QCAPanel            │   │  │
│  │  │ AnnotatLyr  │  │ TrackCtrl   │  │ RWSPanel            │   │  │
│  │  │ OverlayLyr  │  │ Settings    │  │ ExportPanel         │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │              ZUSTAND STORES (8 stores)                  │  │  │
│  │  │                                                         │  │  │
│  │  │  playerStore    │ dicomStore     │ ecgStore            │  │  │
│  │  │  segmentStore   │ trackingStore  │ qcaStore            │  │  │
│  │  │  rwsStore       │ calibrationStore                     │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              │ HTTP/WebSocket                       │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 PYTHON BACKEND (FastAPI)                      │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │                    API ROUTERS                           │ │  │
│  │  │  /dicom  │  /segmentation  │  /tracking  │  /qca  │ /rws │ │  │
│  │  │  /calibration  │  /export  │  /ws                        │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │                    CORE ENGINES                          │ │  │
│  │  │                                                          │ │  │
│  │  │  DicomHandler     │  SegmentationEngine  │  TrackingEngine│ │  │
│  │  │  ECGParser        │  CenterlineExtractor │  QCAEngine     │ │  │
│  │  │  RWSEngine        │  CalibrationEngine   │  PDFReporter   │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │               EXTERNAL LIBRARIES                         │ │  │
│  │  │  PyTorch │ OpenCV │ pydicom │ AngioPy │ numpy │ scipy   │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer Breakdown

### Layer 1: Tauri Shell (Rust)
**Responsibility:** Native OS integration
- Window creation and management
- File system operations (DICOM file access)
- Native file dialogs
- Python backend process management
- IPC between frontend and system

**Key Files:**
- `src-tauri/src/main.rs`
- `src-tauri/src/lib.rs`
- `src-tauri/tauri.conf.json`

### Layer 2: React Frontend (TypeScript)
**Responsibility:** User interface and state management

#### 2a. UI Components
Organized by feature domain:
```
src/components/
├── Viewer/           # Canvas layers, video display
├── Controls/         # Playback, tools, settings
├── Panels/           # ECG, QCA, RWS data panels
├── Annotation/       # Seed points, ROI tools
└── Common/           # Buttons, dialogs, shared UI
```

#### 2b. Zustand Stores (8 stores)
```
src/stores/
├── playerStore.ts      # Playback state, current frame
├── dicomStore.ts       # DICOM metadata, frames
├── ecgStore.ts         # ECG data, R-peaks, beats
├── segmentationStore.ts # Masks, centerlines (Map-based)
├── trackingStore.ts    # ROI, confidence, progress
├── qcaStore.ts         # QCA metrics (Map-based)
├── rwsStore.ts         # RWS results per beat
└── calibrationStore.ts # Pixel spacing, calibration
```

### Layer 3: Python Backend (FastAPI)
**Responsibility:** Heavy computation, ML inference

#### 3a. API Routers
```
python-backend/app/api/
├── dicom.py          # DICOM load, frame extraction
├── segmentation.py   # Model inference, centerline
├── tracking.py       # CSRT, template matching, OF
├── qca.py            # Diameter calculation
├── rws.py            # RWS computation
├── calibration.py    # Calibration management
├── export.py         # CSV, PDF generation
└── websocket.py      # Real-time updates
```

#### 3b. Core Engines
```
python-backend/app/core/
├── dicom_handler.py       # pydicom wrapper
├── ecg_parser.py          # Siemens ECG extraction
├── segmentation_engine.py # AngioPy integration
├── centerline_extractor.py # Skeleton-based extraction
├── tracking_engine.py     # Hybrid tracker
├── qca_engine.py          # Diameter analysis
├── rws_engine.py          # RWS calculation
├── calibration_engine.py  # Pixel-to-mm conversion
└── pdf_reporter.py        # Report generation
```

## Data Flow

### RWS Analysis Workflow
```
1. DICOM Load
   User selects file → Backend parses → Frontend receives frames + metadata

2. ECG Processing
   Backend extracts ECG → R-peak detection → Beat boundaries calculated

3. Initial Segmentation (Frame 0)
   User places seed points → Backend segments → Mask + Centerline returned

4. ROI Definition
   User draws ROI around vessel → Stored in trackingStore

5. Tracking Propagation
   For each frame in beat range:
     a. CSRT tracks ROI position
     b. Template matching refines seed points
     c. Optical flow fallback if needed
     d. Re-segment with new seeds
     e. Extract centerline
     f. Calculate QCA

6. RWS Calculation
   For each beat:
     a. Find frame with max diameter at MLD position (Dmax)
     b. Find frame with min diameter at MLD position (Dmin)
     c. RWS = (Dmax - Dmin) / Dmax × 100%

7. Export
   User exports → Backend generates CSV/PDF → File saved
```

## Communication Protocols

### HTTP REST
- Standard request/response for CRUD operations
- JSON payloads
- Used for: DICOM load, single frame requests, export triggers

### WebSocket
- Real-time bidirectional communication
- Used for: Tracking progress, live segmentation updates
- Endpoint: `/ws/tracking`

### Data Formats
- Frames: Base64 encoded PNG
- Masks: Base64 encoded binary
- Centerlines: Array of [x, y] coordinates
- Metrics: JSON objects

## Key Design Patterns

### 1. Map-Based Per-Frame Storage
```typescript
// Each frame-indexed data uses Map
masks: Map<number, Uint8Array>
centerlines: Map<number, Point[]>
qcaMetrics: Map<number, QCAResult>
```
**Why:** O(1) lookup, sparse storage, easy frame clearing

### 2. Transform Version Synchronization
```typescript
// All canvas layers sync via version number
viewTransformVersion: number
// Increment on zoom/pan → all layers re-render
```
**Why:** Avoids deep object comparison, guarantees sync

### 3. Callback Refs for Canvas
```typescript
// Use callback ref, not useRef
const canvasRef = useCallback((canvas) => {
  if (canvas) initCanvas(canvas);
}, []);
```
**Why:** Guarantees DOM ready before init

### 4. getState() for Async Updates
```typescript
// In async callbacks, use getState()
const frame = usePlayerStore.getState().currentFrame;
// NOT: setFrame(prev => prev + 1)
```
**Why:** Avoids stale closure issues

## Module Dependencies

```
                    ┌─────────────┐
                    │   rwsStore  │
                    └──────┬──────┘
                           │ depends on
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ qcaStore │ │ ecgStore │ │trackStore│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
                  ┌───────────────┐
                  │segmentStore   │
                  └───────┬───────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │dicomStore│ │playerStore│ │calibStore│
        └──────────┘ └──────────┘ └──────────┘
```

---

*Last Updated: 2024-11-24*
