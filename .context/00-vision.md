# Project Vision

## Project Name
**Coronary RWS Analyser** - Open-Source Radial Wall Strain Analysis Platform

## Vision Statement
An open-source, academically rigorous coronary angiography analysis platform focused on Radial Wall Strain (RWS) calculation, designed to be the first freely available tool for this clinically significant metric.

## Primary Goal
Calculate Radial Wall Strain (RWS) from coronary angiography videos using:
- Deep learning-based vessel segmentation
- Automated vessel tracking across cardiac cycles
- Quantitative Coronary Analysis (QCA) for diameter measurements
- RWS calculation based on Hong et al. (2023) methodology

## Target Users
1. **Clinical Researchers** - Studying plaque vulnerability and coronary artery disease
2. **Interventional Cardiologists** - Assessing lesion characteristics
3. **Medical Imaging Scientists** - Developing and validating new analysis methods
4. **Academic Institutions** - Teaching and research purposes

## Why This Project?
- **No open-source alternative exists** for RWS analysis
- Commercial solutions are expensive and closed
- Reproducibility crisis in medical imaging research
- Need for transparent, validated analysis tools

## Core Features (MVP)
1. DICOM import with ECG metadata extraction (Siemens format)
2. Video playback with frame-accurate navigation
3. ECG visualization with R-peak detection
4. Seed point-based vessel segmentation
5. Centerline extraction
6. ROI-based vessel tracking (automatic + manual)
7. QCA metrics (MLD, Proximal RD, Distal RD, DS%, LL)
8. **RWS calculation** (primary deliverable)
9. Calibration (DICOM + manual catheter-based)
10. Export (CSV, JSON, PDF reports)

## Academic Deliverable
This project will be published as an open-source software paper, documenting:
- Development methodology
- Algorithm validation
- Clinical accuracy metrics
- Comparison with existing solutions

## Scope Boundaries

### In Scope (V1.0)
- Single vessel analysis per session
- Manual seed point initialization
- Semi-automatic tracking
- Beat-by-beat RWS analysis
- Standard DICOM formats

### Out of Scope (V1.0)
- Multi-vessel simultaneous analysis
- Fully automatic vessel detection (no seed points)
- 3D reconstruction
- FFR/iFR integration
- Cloud deployment
- Model training/fine-tuning capabilities

## Success Criteria
1. RWS calculation accuracy: correlation >0.95 with manual measurement
2. QCA accuracy: <3% deviation from commercial software
3. Processing time: <500ms per frame on standard hardware
4. Usability: Complete analysis in <5 minutes per video
5. Academic: Peer-reviewed publication acceptance

## Lessons from V2.2
### What Worked Well
- Zustand Map-based per-frame storage pattern
- Multi-layer canvas architecture
- Hybrid tracking (CSRT + Template + Optical Flow)
- WebSocket for real-time updates

### What Needs Improvement
- VideoPlayer.tsx was too large (1800 lines) - needs modularization
- Better separation of concerns in components
- More comprehensive error handling
- Better TypeScript strictness

### What to Change
- Cleaner code architecture from the start
- Better inline documentation
- Comprehensive test coverage
- Academic-grade documentation

---
*Created: 2024-11-24*
*Last Updated: 2024-11-24*
