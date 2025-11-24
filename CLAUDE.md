# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Coronary RWS Analyser** - Open-source Radial Wall Strain analysis platform for coronary angiography. Primary goal: Calculate RWS from angiography videos with academic-grade documentation for publication.

**Key Differentiator:** First open-source tool for RWS analysis.

## Critical Reading Order

1. **`.context/00-vision.md`** - Project goals and scope
2. **`.context/decision.log`** - All architectural decisions with rationale
3. **`.context/01-requirements.md`** - Functional/non-functional requirements
4. **`.context/architecture/overview.md`** - System architecture
5. **`.context/domain/rws-methodology.md`** - RWS clinical background

**After making decisions:** Update `decision.log` with rationale and references.

## Development Commands

### Start Development
```bash
npm run dev:full  # Backend + Tauri (WSL/Linux)
# OR manually:
# Terminal 1: npm run tauri:dev
# Terminal 2: cd python-backend && uvicorn app.main:app --reload --port 8000
```

### Build & Test
```bash
npm run build           # TypeScript check + Vite build
npm run tauri:build     # Production build
cd python-backend && pytest tests/  # Python tests
```

### Health Checks
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/qca/health
curl http://127.0.0.1:8000/rws/health
```

## Tech Stack

**Frontend:** Tauri 2.x, React 18, TypeScript (strict), Zustand (8 stores), Radix UI, Tailwind CSS

**Backend:** Python 3.10+, FastAPI, PyTorch, OpenCV 4.10+ (opencv-contrib-python), pydicom

**Segmentation:** Custom nnU-Net v2 (dual-channel: image + Gaussian spatial attention)

**Tracking:** Hybrid CSRT + Template Matching + Optical Flow

## Architecture Principles

### 1. Layer-by-Layer Development
```
Phase 1: Backend API stubs + core engines
Phase 2: Zustand stores
Phase 3: UI components
Phase 4: Integration
```

### 2. Map-Based Per-Frame Storage
```typescript
// All frame-indexed data uses Map<number, T>
masks: Map<number, Uint8Array>
centerlines: Map<number, Point[]>
qcaMetrics: Map<number, QCAResult>
```

### 3. Transform Version Sync (Canvas Layers)
```typescript
// Increment on zoom/pan to sync all layers
viewTransformVersion: number
```

### 4. getState() in Async Contexts
```typescript
// Always use getState() in async callbacks
const frame = usePlayerStore.getState().currentFrame;
```

## Zustand Stores (8 total)

| Store | Responsibility |
|-------|---------------|
| `playerStore` | Playback, current frame, speed |
| `dicomStore` | Metadata, frames (base64), pixel spacing |
| `ecgStore` | ECG data, R-peaks, beat boundaries |
| `segmentationStore` | Masks, centerlines (Map-based) |
| `trackingStore` | ROI, confidence, progress |
| `qcaStore` | QCA metrics per frame (Map-based) |
| `rwsStore` | RWS results per beat |
| `calibrationStore` | Pixel-to-mm calibration |

## Critical Bugs to Avoid

### 1. Coordinate Swap (Python ↔ TypeScript)
```python
# Python extractors return (y, x)
# Canvas needs (x, y)
centerline_canvas = [(float(x), float(y)) for y, x in coords]
```

### 2. Optical Flow Requirements
```python
# MUST be grayscale uint8
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = frame.astype(np.uint8)
```

### 3. CSRT Tracker API
```python
# OpenCV version-dependent
try:
    tracker = cv2.legacy.TrackerCSRT_create()
except:
    tracker = cv2.TrackerCSRT_create()
```

### 4. Canvas Callback Ref
```typescript
// Use callback ref, not useRef
const canvasRef = useCallback((canvas) => {
  if (canvas) init(canvas);
}, []);
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dicom/load` | POST | Load DICOM file |
| `/dicom/frame/{idx}` | GET | Get frame (base64) |
| `/dicom/ecg` | GET | ECG data + R-peaks |
| `/segmentation/segment` | POST | Run segmentation |
| `/segmentation/centerline` | POST | Extract centerline |
| `/tracking/initialize` | POST | Init CSRT tracker |
| `/tracking/track` | POST | Track single frame |
| `/tracking/propagate` | POST | Auto-propagate |
| `/qca/calculate` | POST | Calculate QCA |
| `/rws/calculate` | POST | Calculate RWS |
| `/export/csv` | POST | Export to CSV |
| `/export/pdf` | POST | Generate PDF report |
| `/ws/tracking` | WS | Real-time tracking updates |

## RWS Calculation (Primary Feature)

```python
# Per cardiac beat, at each measurement point (MLD, Proximal, Distal):
Dmax = max([diameter[frame] for frame in beat_frames])
Dmin = min([diameter[frame] for frame in beat_frames])
RWS = (Dmax - Dmin) / Dmax * 100  # percentage

# Clinical interpretation:
# < 8%: Normal
# 8-12%: Intermediate
# > 12-14%: Possible vulnerable plaque
```

## File Organization

```
coronary_rws_analyser/
├── .context/              # Meta: vision, decisions, requirements
├── akademik/              # Academic: literature, manuscript
├── .claude/commands/      # Custom slash commands
├── src/                   # React frontend
│   ├── components/        # UI components
│   ├── stores/            # Zustand stores
│   └── lib/               # API client, utilities
├── src-tauri/             # Tauri (Rust)
├── python-backend/        # FastAPI backend
│   ├── app/api/           # API routers
│   ├── app/core/          # Core engines
│   └── app/models/        # Pydantic schemas
└── docs/                  # User documentation
```

## Documentation Standards

### Code Comments
- Document **why**, not **what**
- Reference papers for algorithms: `# Ref: Hong et al., 2023, Eq. 3`
- Mark TODOs with context: `# TODO(academic): Add validation metrics`

### Decision Logging
When making architectural decisions:
1. Add entry to `.context/decision.log`
2. Include alternatives considered
3. Reference supporting literature
4. Document consequences

## Performance Targets

- Video: 30 FPS @ 4K, 60 FPS @ FHD
- Segmentation: <300ms (CPU), <100ms (GPU)
- Tracking: <350ms per frame
- QCA: <50ms per frame
- RWS: <10ms per beat

## Academic Requirements

This project will be published. Ensure:
1. All algorithms have literature references
2. Decision rationale is documented
3. Code is readable and well-commented
4. Validation metrics are tracked
5. Reproducibility is maintained

---

*Last Updated: 2024-11-24*
