# Requirements Specification

## Functional Requirements

### FR-1: DICOM Processing
- **FR-1.1:** Load single-frame and multi-frame DICOM files
- **FR-1.2:** Extract and display DICOM metadata (patient info anonymized)
- **FR-1.3:** Parse Siemens curved ECG data from DICOM tags
- **FR-1.4:** Extract ImagerPixelSpacing for calibration
- **FR-1.5:** Support common transfer syntaxes (JPEG, JPEG2000, RLE)

### FR-2: Video Playback
- **FR-2.1:** Display DICOM frames as video
- **FR-2.2:** Play/Pause/Stop controls
- **FR-2.3:** Frame-by-frame navigation (arrow keys)
- **FR-2.4:** Adjustable playback speed (0.25x - 2x)
- **FR-2.5:** Loop mode (full video, beat range, custom range)
- **FR-2.6:** Frame scrubbing via timeline

### FR-3: ECG Visualization
- **FR-3.1:** Display ECG waveform synchronized with video
- **FR-3.2:** Automatic R-peak detection
- **FR-3.3:** Beat boundary visualization
- **FR-3.4:** Frame-to-ECG time mapping
- **FR-3.5:** Manual R-peak correction (optional)

### FR-4: Vessel Segmentation
- **FR-4.1:** User places 2-5 seed points on vessel
- **FR-4.2:** Deep learning segmentation (AngioPy)
- **FR-4.3:** Probability map generation
- **FR-4.4:** Binary mask extraction (threshold configurable)
- **FR-4.5:** Centerline extraction from mask
- **FR-4.6:** Visual overlay of segmentation on video

### FR-5: Vessel Tracking
- **FR-5.1:** User defines ROI bounding box
- **FR-5.2:** Automatic ROI tracking across frames (CSRT)
- **FR-5.3:** Seed point tracking within ROI
- **FR-5.4:** Confidence score per frame
- **FR-5.5:** Auto-stop on low confidence
- **FR-5.6:** Manual correction capability
- **FR-5.7:** Bidirectional tracking (forward/backward)

### FR-6: QCA (Quantitative Coronary Analysis)
- **FR-6.1:** N-point diameter profiling (30/50/70 points)
- **FR-6.2:** MLD (Minimum Lumen Diameter) detection
- **FR-6.3:** Proximal Reference Diameter calculation
- **FR-6.4:** Distal Reference Diameter calculation
- **FR-6.5:** Diameter Stenosis (DS%) calculation
- **FR-6.6:** Lesion Length measurement
- **FR-6.7:** Sub-pixel diameter measurement (Gaussian/Parabolic fitting)

### FR-7: RWS (Radial Wall Strain) - PRIMARY FEATURE
- **FR-7.1:** Calculate Dmax and Dmin per cardiac cycle
- **FR-7.2:** RWS = (Dmax - Dmin) / Dmax × 100%
- **FR-7.3:** RWS at MLD position (most clinically significant)
- **FR-7.4:** RWS at Proximal RD position
- **FR-7.5:** RWS at Distal RD position
- **FR-7.6:** Beat-by-beat RWS analysis
- **FR-7.7:** RWS trend visualization across beats
- **FR-7.8:** Clinical interpretation guidance (normal <8%, vulnerable >12-14%)

### FR-8: Calibration
- **FR-8.1:** Automatic calibration from DICOM ImagerPixelSpacing
- **FR-8.2:** Manual catheter-based calibration
- **FR-8.3:** Support for 5F, 6F, 7F, 8F catheters
- **FR-8.4:** Live recalculation of all metrics on calibration change

### FR-9: Export
- **FR-9.1:** QCA metrics to CSV
- **FR-9.2:** RWS results to CSV
- **FR-9.3:** Diameter profiles to JSON
- **FR-9.4:** Centerlines to JSON
- **FR-9.5:** Clinical PDF report with charts
- **FR-9.6:** Segmentation masks (optional, NIfTI format)

### FR-10: User Interface
- **FR-10.1:** Multi-layer canvas (video + segmentation + annotation + overlay)
- **FR-10.2:** Zoom (0.1x - 10x, cursor-aware)
- **FR-10.3:** Pan (mouse drag, keyboard)
- **FR-10.4:** Reset view
- **FR-10.5:** Dark theme UI
- **FR-10.6:** Responsive layout
- **FR-10.7:** Keyboard shortcuts

---

## Non-Functional Requirements

### NFR-1: Performance
- **NFR-1.1:** Video playback: 30 FPS at 4K, 60 FPS at Full HD
- **NFR-1.2:** Zoom/Pan latency: <16ms
- **NFR-1.3:** Segmentation: <300ms per frame (CPU), <100ms (GPU)
- **NFR-1.4:** Tracking: <350ms per frame
- **NFR-1.5:** QCA calculation: <50ms per frame
- **NFR-1.6:** Application startup: <5 seconds

### NFR-2: Accuracy
- **NFR-2.1:** QCA accuracy: <3% deviation from reference
- **NFR-2.2:** RWS precision: ±0.02
- **NFR-2.3:** Centerline accuracy: sub-pixel (B-spline smoothed)

### NFR-3: Usability
- **NFR-3.1:** Complete analysis workflow in <5 minutes
- **NFR-3.2:** Minimal training required (intuitive UI)
- **NFR-3.3:** Clear error messages
- **NFR-3.4:** Undo/Redo support (desirable)

### NFR-4: Reliability
- **NFR-4.1:** Session auto-save
- **NFR-4.2:** Crash recovery
- **NFR-4.3:** Graceful degradation on missing GPU

### NFR-5: Portability
- **NFR-5.1:** Windows 10/11 support
- **NFR-5.2:** macOS support (desirable)
- **NFR-5.3:** Linux support (desirable)
- **NFR-5.4:** No internet required for core functionality

### NFR-6: Maintainability
- **NFR-6.1:** Modular code architecture
- **NFR-6.2:** Comprehensive inline documentation
- **NFR-6.3:** TypeScript strict mode
- **NFR-6.4:** Python type hints
- **NFR-6.5:** >80% test coverage (target)

### NFR-7: Academic Requirements
- **NFR-7.1:** Open-source license (MIT or Apache 2.0)
- **NFR-7.2:** Reproducible builds
- **NFR-7.3:** Documented algorithms with references
- **NFR-7.4:** Validation dataset included

---

## Constraints

### Technical Constraints
- Must use Python for ML/CV (PyTorch, OpenCV ecosystem)
- DICOM parsing via pydicom (industry standard)
- Desktop-first (not web deployment)

### Resource Constraints
- Single developer
- No dedicated QA team
- Limited access to clinical validation data

### Regulatory Constraints
- NOT for clinical diagnosis (research use only)
- Must include appropriate disclaimers
- Patient data anonymization required

---

*Last Updated: 2024-11-24*
