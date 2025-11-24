# Radial Wall Strain (RWS) Methodology

## Definition

Radial Wall Strain (RWS) is a novel angiographic parameter that quantifies the radial deformation of coronary artery walls during the cardiac cycle. It measures the percentage change in vessel diameter between maximum expansion (diastole) and minimum expansion (systole).

## Formula

```
RWS = (Dmax - Dmin) / Dmax × 100%
```

Where:
- **Dmax**: Maximum diameter at a specific point during the cardiac cycle
- **Dmin**: Minimum diameter at the same point during the cardiac cycle

## Clinical Significance

### What RWS Measures
- **Arterial compliance**: How much the vessel expands/contracts
- **Wall stiffness**: Inverse relationship with RWS
- **Plaque characteristics**: Vulnerable plaques show altered RWS

### Clinical Thresholds (Hong et al., 2023)
| RWS Value | Interpretation |
|-----------|----------------|
| < 8% | Normal vessel |
| 8-12% | Intermediate |
| > 12-14% | Possible vulnerable plaque |

### Why RWS Matters
1. **Non-invasive assessment**: Unlike IVUS/OCT, uses standard angiography
2. **Plaque vulnerability**: May identify high-risk lesions
3. **Functional assessment**: Complements anatomical stenosis measurement
4. **Prognostic value**: Potential predictor of adverse events

## Measurement Locations

RWS should be calculated at three key positions:

### 1. MLD (Minimum Lumen Diameter) - MOST IMPORTANT
- Location of maximum stenosis
- Most clinically relevant measurement
- Changes here indicate lesion behavior

### 2. Proximal Reference Diameter
- Healthy segment proximal to lesion
- Serves as reference for normal RWS
- Typically higher RWS (more compliant)

### 3. Distal Reference Diameter
- Healthy segment distal to lesion
- May be affected by downstream effects
- Useful for comparison

## Calculation Process

### Step 1: Identify Cardiac Cycle
1. Detect R-peaks in ECG
2. Define beat boundaries (R-to-R interval)
3. For each beat, identify all frames

### Step 2: Measure Diameter Per Frame
1. At each measurement point (MLD, Proximal, Distal)
2. Measure perpendicular diameter to centerline
3. Use sub-pixel methods (Gaussian fitting)

### Step 3: Find Dmax and Dmin
```python
# For each measurement point
diameters = [measure_diameter(frame, point) for frame in beat_frames]
Dmax = max(diameters)
Dmin = min(diameters)
```

### Step 4: Calculate RWS
```python
RWS = (Dmax - Dmin) / Dmax * 100
```

## Implementation Considerations

### Temporal Resolution
- Adequate frame rate needed (>15 fps recommended)
- More frames = better Dmax/Dmin detection

### Spatial Resolution
- Sub-pixel diameter measurement essential
- Gaussian/Parabolic fitting for precision

### Tracking Accuracy
- Vessel must be tracked accurately across cycle
- Loss of tracking invalidates RWS measurement

### Calibration
- Pixel-to-mm calibration affects absolute values
- Relative comparison (RWS ratio) less affected

## Validation Requirements

### Ground Truth
- Manual expert measurement (gold standard)
- Repeated measurements for inter/intra-observer variability

### Metrics
- Correlation coefficient (r > 0.95 target)
- Bland-Altman analysis
- Mean absolute error

## References

1. **Hong et al.** "Radial Wall Strain: A Novel Angiographic Parameter for Plaque Vulnerability Assessment" *EuroIntervention* 2023
   - Primary methodology reference
   - Clinical validation study
   - Threshold definitions

2. **Supporting Literature**
   - IVUS-based wall strain studies
   - Coronary artery compliance research
   - Plaque vulnerability markers

---

*This document serves as the theoretical foundation for RWS implementation in the Coronary RWS Analyser.*
