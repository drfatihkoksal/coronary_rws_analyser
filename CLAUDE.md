# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Proje:** Coronary RWS Analyser - Koroner anjiyografi videolarından Radial Wall Strain (RWS) hesaplayan açık kaynaklı masaüstü uygulaması.
**Hedef:** JOSS (Journal of Open Source Software) yayını

---

## Geliştirme Öncesi

Karar almadan önce şu dosyaları oku:
- `.context/decision.log` - Tüm mimari kararlar + gerekçeler
- `.context/00-vision.md` - Proje vizyonu
- `.context/domain/rws-methodology.md` - RWS klinik arka planı

**Karar aldıktan sonra:** `decision.log` dosyasını güncelle.

---

## Komutlar

### Kurulum

```bash
# Frontend
npm install

# Python backend
cd python-backend
python -m venv venv
source venv/bin/activate          # Linux/WSL/Mac
# Windows CMD: venv\Scripts\activate.bat
# Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt

# GPU (CUDA 12.1) için:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### Geliştirme

```bash
# Önerilen - Backend + Tauri paralel başlat
npm run dev:full

# Manuel (iki terminal):
npm run tauri:dev           # Terminal 1: Tauri + React (localhost:1420)
npm run backend:dev         # Terminal 2: FastAPI (localhost:8000)

# Sadece React (Tauri olmadan):
npm run dev                 # Vite dev server (localhost:5173)
```

### Build, Lint, Test

```bash
npm run tauri:build         # Production build (.exe/.dmg/.AppImage)
npm run build               # TypeScript type-check + Vite build
npm run lint                # ESLint
npm run typecheck           # TypeScript only

# Python testleri
cd python-backend && pytest tests/
cd python-backend && pytest tests/test_qca.py -v  # Tek test dosyası
cd python-backend && pytest tests/ -k "test_name"  # Tek test

# Health check
curl http://127.0.0.1:8000/health
```

---

## Teknoloji Stack'i

- **Frontend:** Tauri 2.x, React 18, TypeScript (strict), Zustand 5, Radix UI, Tailwind CSS
- **Backend:** Python 3.10+, FastAPI, PyTorch 2.x, OpenCV (opencv-contrib-python for CSRT)
- **Segmentation:** Custom nnU-Net v2 (dual-channel: image + Gaussian spatial map)
- **Charts:** Recharts (ECG visualization, diameter profiles)

---

## Mimari Prensipler

### 1. Layer-by-Layer Geliştirme (DECISION-006)

```
Phase 1: Backend API + Core Engines  ✓
Phase 2: Zustand Stores              ✓
Phase 3: UI Components               ✓
Phase 4: Integration & Testing       ← ŞİMDİKİ AŞAMA
Phase 5: Academic Validation
```

Her faz önceki fazın üzerine inşa edilir. UI, altyapı tamamlanmadan yazılmaz.

### 2. Map-Based Per-Frame Storage (V2.2'den korundu)

```typescript
// Tüm frame-indexed data Map<number, T> kullanır
interface SegmentationStore {
  masks: Map<number, Uint8Array>;          // frame → mask
  centerlines: Map<number, Point[]>;       // frame → centerline
  probabilityMaps: Map<number, Float32Array>;

  getMask: (frameIndex: number) => Uint8Array | undefined;
  setMask: (frameIndex: number, mask: Uint8Array) => void;
}
```

**Neden Map?**
- Sparse storage (sadece işlenmiş frame'ler tutulur)
- O(1) erişim
- Kolay frame clearing
- LRU cache için uygun (ileride)

### 3. Transform Version Sync (Canvas Layers)

```typescript
// Zoom/pan'de tüm layer'ları senkronize et
const [viewTransform, setViewTransform] = useState({ scale: 1, x: 0, y: 0 });
const [viewTransformVersion, setViewTransformVersion] = useState(0);

// Transform değiştiğinde version'ı artır
const handleZoom = () => {
  setViewTransform(newTransform);
  setViewTransformVersion(v => v + 1);  // Tüm layer'lar bunu dinler
};
```

### 4. getState() in Async Contexts (V2.2'den öğrenilen)

```typescript
// ❌ YANLIŞ - async callback'te stale state
setCurrentFrame((prev) => prev + 1);

// ✅ DOĞRU - güncel state
const currentFrame = usePlayerStore.getState().currentFrame;
setCurrentFrame(currentFrame + 1);
```

### 5. Callback Ref Pattern (Canvas)

```typescript
// ❌ YANLIŞ - timing issues
const canvasRef = useRef<HTMLCanvasElement>(null);
useEffect(() => {
  if (canvasRef.current) init(canvasRef.current);
}, []);

// ✅ DOĞRU - DOM hazır garantisi
const canvasRef = useCallback((canvas: HTMLCanvasElement | null) => {
  if (canvas) init(canvas);
}, []);
```

---

## Zustand Store'ları (8 adet)

| Store | Sorumluluk | Map-based? |
|-------|------------|------------|
| `playerStore` | Playback, current frame, speed, loop | No |
| `dicomStore` | Metadata, frames (base64), pixel spacing | Frames: Map |
| `ecgStore` | ECG data, R-peaks, beat boundaries | No |
| `segmentationStore` | Masks, centerlines, probability maps | Yes |
| `trackingStore` | ROI, confidence, progress | Per-frame: Map |
| `qcaStore` | QCA metrics per frame | Yes |
| `rwsStore` | RWS results per beat | Yes |
| `calibrationStore` | Pixel-to-mm calibration | No |

---

## Core Engine'ler (Backend)

### 1. DicomHandler (`app/core/dicom_handler.py`)

```python
class DicomHandler:
    """DICOM dosya işleme"""

    def load(self, path: str) -> DicomData
    def get_frame(self, index: int) -> np.ndarray
    def get_metadata(self) -> DicomMetadata
    def extract_ecg(self) -> ECGData  # Siemens format
    def detect_r_peaks(self) -> List[int]
```

### 2. Segmentation Engines

**nnU-Net (`app/core/nnunet_inference.py`)** - Custom dual-channel model:
```python
class NNUNetInference:
    def segment(self, image: np.ndarray, roi: BoundingBox) -> SegmentationResult
    def generate_gaussian_map(self, shape: Tuple, sigma: float = 35) -> np.ndarray
    # Pipeline: ROI crop → Dual-channel → Inference → Post-process
```

**AngioPy (`app/core/angiopy_segmentation.py`)** - Alternative InceptionResNetV2+U-Net:
```python
class AngioPySegmentation:
    def segment(self, image: np.ndarray) -> np.ndarray  # Returns binary mask
```

### 3. CenterlineExtractor (`app/core/centerline_extractor.py`)

```python
class CenterlineExtractor:
    """Skeleton-based centerline extraction"""

    def extract(self, mask: np.ndarray) -> List[Tuple[float, float]]
    def smooth_bspline(self, points: List[Tuple]) -> List[Tuple]
```

### 4. TrackingEngine (`app/core/tracking_engine.py`)

```python
class TrackingEngine:
    """Hybrid CSRT + Template + Optical Flow"""

    def initialize(self, frame: np.ndarray, roi: BoundingBox, seed_points: List[Point])
    def track(self, frame: np.ndarray) -> TrackingResult
    def propagate(self, frames: List[np.ndarray], start: int, end: int) -> Generator

    # Confidence scoring: 5-frame rolling window
    # Auto-stop threshold: 0.6
```

### 5. QCAEngine (`app/core/qca_engine.py`)

```python
class QCAEngine:
    """Quantitative Coronary Analysis"""

    def calculate(self, mask: np.ndarray, centerline: List[Point], pixel_spacing: float) -> QCAResult
    def profile_diameters(self, n_points: int = 50) -> List[float]
    def find_mld(self) -> MLDResult
    def calculate_stenosis(self) -> float  # DS%
```

### 6. RWSEngine (`app/core/rws_engine.py`) - **ANA ÖZELLİK**

```python
class RWSEngine:
    """Radial Wall Strain calculation"""

    def calculate(self, diameters: Dict[int, QCAResult], beat_frames: List[int]) -> RWSResult

    # Formula: RWS = (Dmax - Dmin) / Dmax × 100%

    # Output positions:
    # - MLD RWS (en önemli)
    # - Proximal RD RWS
    # - Distal RD RWS
```

### 7. CalibrationEngine (`app/core/calibration_engine.py`)

```python
class CalibrationEngine:
    """Pixel-to-mm calibration from catheter or known objects"""

    def calibrate_from_catheter(self, image: np.ndarray, known_diameter_mm: float) -> float
    def calibrate_manual(self, pixel_distance: float, mm_distance: float) -> float
```

---

## API Endpoint'leri

API dokümantasyonu: http://127.0.0.1:8000/docs

Temel endpoint'ler: `/dicom`, `/segmentation`, `/tracking`, `/qca`, `/rws`, `/calibration`, `/export`

### Frontend → Backend İletişimi

API client (`src/lib/api.ts`) tüm backend çağrılarını yönetir:

```typescript
import { api } from '@/lib/api';

// DICOM yükleme
const metadata = await api.dicom.load(filePath);
const frame = await api.dicom.getFrame(frameIndex);

// Segmentation
const result = await api.segmentation.segment(frameIndex, roi);

// Tracking
await api.tracking.initialize(roi, seedPoints);
const trackResult = await api.tracking.propagate(startFrame, endFrame);

// QCA & RWS
const qca = await api.qca.calculate(frameIndex);
const rws = await api.rws.calculate(beatFrames);
```

---

## RWS Hesaplama (Ana Özellik)

```python
RWS = (Dmax - Dmin) / Dmax × 100%
# <8%: Normal, 8-12%: Intermediate, >12%: Vulnerable plak
# Ref: Hong et al., EuroIntervention 2023
```

Her kardiyak beat için 3 noktada hesaplanır: MLD RWS (en önemli), Proximal RD, Distal RD

---

## Kritik Hatalardan Kaçınma

### 1. Koordinat Sistemi (y,x) ↔ (x,y)

```python
# Python extractors (y, x) döndürür
# Canvas (x, y) bekler

# ❌ YANLIŞ - Görüntü yansır
centerline_canvas = [(float(y), float(x)) for y, x in coords]

# ✅ DOĞRU - Swap yap
centerline_canvas = [(float(x), float(y)) for y, x in coords]
```

### 2. Optical Flow Gereksinimleri

```python
# ❌ YANLIŞ - Sessizce başarısız olur
flow = cv2.calcOpticalFlowPyrLK(prev, next, ...)  # RGB veya float32

# ✅ DOĞRU - Grayscale uint8 zorunlu
if len(frame.shape) == 3:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
if frame.dtype != np.uint8:
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
```

### 3. CSRT Tracker API (OpenCV version-dependent)

```python
try:
    # OpenCV >= 4.5.1
    tracker = cv2.legacy.TrackerCSRT_create()
except AttributeError:
    # OpenCV < 4.5.1
    tracker = cv2.TrackerCSRT_create()
```

### 4. nnU-Net 2D Input Shape

```python
# nnU-Net 2D, (C, 1, H, W) bekler (singleton Z dimension)
input_array = dual_channel[:, np.newaxis, :, :]  # (2, 1, H, W)
```

---

## Canvas Layer Mimarisi (Frontend)

Video görüntüleme için katmanlı canvas sistemi (`src/lib/canvas/`):

```typescript
// LayerManager coordinates all canvas layers
class LayerManager {
  private layers: Layer[] = [];      // VideoLayer, SegmentationLayer, AnnotationLayer, OverlayLayer
  private transform: ViewTransform;  // Shared zoom/pan state

  render(frameIndex: number): void   // Render all layers in order
  setTransform(t: ViewTransform): void
}

// Each layer implements Layer interface
interface Layer {
  render(ctx: CanvasRenderingContext2D, frameIndex: number): void;
  setTransform(transform: ViewTransform): void;
}
```

**Katmanlar (alt→üst):**
1. `VideoLayer` - DICOM frame görüntüsü
2. `SegmentationLayer` - Mask overlay (yarı-şeffaf)
3. `AnnotationLayer` - ROI, centerline, landmarks
4. `OverlayLayer` - Ölçümler, etiketler, crosshair

---

## Klasör Yapısı

```
src/                    # React frontend
  components/           # UI components (Viewer/, Controls/, Panels/, common/)
  stores/               # Zustand stores (8 adet)
  lib/                  # Utilities (api.ts, canvas/, sessionUtils.ts)
  hooks/                # Custom hooks (useCanvasLayers.ts)
  types/                # TypeScript types
src-tauri/              # Tauri (Rust) desktop shell
python-backend/         # FastAPI backend
  app/api/              # API routers (dicom, segmentation, tracking, qca, rws)
  app/core/             # Core engines (DicomHandler, TrackingEngine, QCAEngine, RWSEngine)
  app/models/           # Pydantic schemas
.context/               # Meta dokümantasyon (decision.log, vision, requirements)
0000000B                # Test DICOM dosyası (56 frames, 512x512, ECG metadata)
```

---

## Bilinen Sorunlar ve Çözümler

| Sorun | Çözüm |
|-------|-------|
| CSRT bulunamıyor | `pip install opencv-contrib-python` |
| Optical flow NaN | uint8 grayscale kullan (RGB/float32 değil) |
| Mask yansıması | Koordinat (y,x)→(x,y) swap yap |
| İlk Rust build yavaş | Normal (~5-10dk, sonrakiler <30sn) |
| nnU-Net shape error | `[:, np.newaxis, :, :]` ekle (Z dimension) |

---

## Akademik Gereksinimler

Bu proje JOSS yayını hedefliyor:
- Her algoritma için **literatür referansı** ekle
- **decision.log** güncel tut
- TODO format: `# TODO(academic):`, `# TODO(performance):`, `# TODO(v1.1):`

---

## Yeni Özellik Workflow

1. `.context/decision.log` → Karar yaz
2. `PROGRESS.md` → Task'ı ekle
3. Backend: `python-backend/app/core/` (engine) → `app/api/` (router)
4. Frontend: `src/stores/` (Zustand) → `src/components/`
5. Test yaz + Commit

---

*Son Güncelleme: 2025-11-28*
*Durum: Phase 4 - Integration & Testing*
