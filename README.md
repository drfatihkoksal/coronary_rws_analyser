# Coronary RWS Analyser

**Open-Source Radial Wall Strain Analysis Platform for Coronary Angiography**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Development-yellow.svg)](PROGRESS.md)

## Overview

Coronary RWS Analyser is an open-source desktop application for quantitative analysis of coronary angiography videos, with a primary focus on **Radial Wall Strain (RWS)** calculation - a novel parameter for assessing plaque vulnerability.

**This is the first freely available tool for RWS analysis.**

## Key Features

- **DICOM Import** - Multi-frame DICOM with ECG metadata extraction
- **Video Playback** - Smooth playback with frame-accurate navigation
- **ECG Visualization** - R-peak detection and beat boundaries
- **Vessel Segmentation** - Deep learning-based (AngioPy)
- **Vessel Tracking** - Hybrid CSRT + Template Matching + Optical Flow
- **QCA Analysis** - MLD, Reference Diameters, Diameter Stenosis
- **RWS Calculation** - Beat-by-beat radial wall strain analysis
- **Export** - CSV, JSON, PDF clinical reports

## RWS (Radial Wall Strain)

```
RWS = (Dmax - Dmin) / Dmax × 100%
```

| RWS Value | Interpretation |
|-----------|----------------|
| < 8% | Normal vessel |
| 8-12% | Intermediate |
| > 12-14% | Possible vulnerable plaque |

*Reference: Hong et al., EuroIntervention 2023*

## Technology Stack

- **Frontend:** Tauri 2.x, React 18, TypeScript, Zustand, Tailwind CSS
- **Backend:** Python 3.10+, FastAPI, PyTorch, OpenCV
- **Segmentation:** AngioPy (InceptionResNetV2 + U-Net)

## Quick Start

### Prerequisites
- Node.js 18+
- Rust 1.70+
- Python 3.10+
- (Optional) CUDA for GPU acceleration

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/coronary_rws_analyser.git
cd coronary_rws_analyser

# Install frontend dependencies
npm install

# Setup Python backend
cd python-backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
cd ..
```

### Development

```bash
# Recommended (WSL/Linux) - starts both frontend and backend
npm run dev:full

# Manual (two terminals)
npm run tauri:dev           # Terminal 1
npm run backend:dev         # Terminal 2
```

### Build

```bash
npm run tauri:build
```

## Project Structure

```
coronary_rws_analyser/
├── .context/              # Project documentation (vision, decisions, requirements)
├── akademik/              # Academic workspace (literature, manuscript)
├── src/                   # React frontend
├── src-tauri/             # Tauri (Rust)
├── python-backend/        # Python FastAPI backend
├── docs/                  # User documentation
├── CLAUDE.md              # Developer guide
└── PROGRESS.md            # Development progress
```

## Documentation

- **[Vision & Goals](.context/00-vision.md)** - Project vision and scope
- **[Requirements](.context/01-requirements.md)** - Feature specifications
- **[Architecture](.context/architecture/overview.md)** - System design
- **[Decision Log](.context/decision.log)** - Architectural decisions
- **[RWS Methodology](.context/domain/rws-methodology.md)** - Clinical background

## Academic Publication

This project is being developed with academic publication in mind. The entire development process is documented, including:
- Architectural decisions with literature references
- Algorithm implementations with validation
- Comparison with existing solutions

Target journal: JOSS (Journal of Open Source Software)

## Contributing

Contributions are welcome! Please read the [CLAUDE.md](CLAUDE.md) for development guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Disclaimer

**For Research Use Only.** This software is not intended for clinical diagnosis. Always consult qualified medical professionals for clinical decisions.

## Acknowledgments

- AngioPy team (EPFL Center for Imaging)
- Hong et al. for RWS methodology

---

*Development Status: Early Development*
