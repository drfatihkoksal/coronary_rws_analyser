# Custom nnU-Net Segmentation Model Architecture

## Overview

Bu projede koroner arter segmentasyonu için özel olarak eğitilmiş bir nnU-Net v2 modeli kullanılmaktadır. Model, ROI-tabanlı dual-channel input ile çalışacak şekilde tasarlanmıştır.

## Model Framework: nnU-Net v2

### nnU-Net Nedir?

nnU-Net (no-new-Net), medikal görüntü segmentasyonu için self-configuring bir framework'tür. Veri setine göre otomatik olarak:
- Preprocessing pipeline
- Network architecture
- Training scheme
- Post-processing

parametrelerini belirler.

**Referans:** Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods* 18, 203–211 (2021).

### Neden nnU-Net?

| Özellik | Avantaj |
|---------|---------|
| Self-configuring | Manuel hyperparameter tuning gerektirmez |
| State-of-the-art | 23+ medical segmentation challenge'da birinci |
| Reproducible | Standart pipeline, kolay replikasyon |
| Well-documented | Akademik yayınlar için ideal |

---

## Model Configuration

### Dataset Specifications

```yaml
Dataset ID: [TO BE FILLED]
Dataset Name: [TO BE FILLED - e.g., Dataset502_CoronaryGaussianDualChannel]
Task: Binary coronary artery segmentation (background vs vessel)
Modality: X-ray Angiography (2D)
```

### Input Configuration

#### Dual-Channel Input Architecture

```
Channel 0: Grayscale angiography image (ROI crop)
Channel 1: Gaussian spatial attention map

Input Shape: (2, H, W)
```

#### Gaussian Spatial Attention Map

ROI'nin merkezine odaklanmayı sağlayan spatial prior:

```python
def generate_gaussian_spatial_map(shape, sigma=35):
    """
    2D Gaussian attention map generation

    Args:
        shape: (H, W) output dimensions
        sigma: Gaussian standard deviation (default: 35 pixels)

    Returns:
        Normalized Gaussian map [0, 1]
    """
    h, w = shape
    center_y, center_x = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    return (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
```

**Rationale:**
- ROI crop'un merkezinde hedef damar bulunur
- Gaussian map, modelin merkeze odaklanmasını sağlar
- Kenar bölgelerdeki gürültü/artifact'ları suppress eder
- σ=35 pixel, tipik ROI boyutu (200x200) için optimize edilmiştir

### Network Architecture

nnU-Net 2D configuration:

```yaml
Architecture: U-Net (encoder-decoder with skip connections)
Configuration: 2d
Encoder: ConvNeXt-based blocks (nnU-Net v2 default)
Decoder: Transposed convolutions with skip connections

Input channels: 2 (dual-channel)
Output channels: 2 (background, vessel)
Activation: Softmax

Normalization: Instance Normalization
Activation: LeakyReLU (negative_slope=0.01)
```

### Training Configuration

```yaml
Trainer: nnUNetTrainer (or custom variant)
Epochs: [TO BE FILLED - e.g., 500 or 1000]
Batch size: [TO BE FILLED]
Learning rate: [TO BE FILLED]
Optimizer: SGD with Nesterov momentum
Loss function: Dice + CrossEntropy (nnU-Net default)

Data augmentation:
  - Rotation: [-180°, +180°]
  - Scaling: [0.7, 1.4]
  - Gaussian noise
  - Gaussian blur
  - Brightness/Contrast adjustment
  - Mirroring (horizontal, vertical)
  - Elastic deformation
```

---

## Inference Pipeline

### 1. ROI Extraction

```python
def crop_to_roi(image, roi, padding=20):
    """
    Crop full frame to ROI with padding

    Args:
        image: Full angiography frame (H, W)
        roi: Bounding box (x, y, w, h)
        padding: Extra pixels around ROI

    Returns:
        Cropped image, adjusted coordinates
    """
    x, y, w, h = roi
    x_pad = max(0, x - padding)
    y_pad = max(0, y - padding)
    x2_pad = min(image.shape[1], x + w + padding)
    y2_pad = min(image.shape[0], y + h + padding)

    return image[y_pad:y2_pad, x_pad:x2_pad]
```

### 2. Preprocessing

```python
def preprocess(image_crop):
    """
    Prepare image for nnU-Net inference

    1. Convert to float32 [0, 1]
    2. Generate Gaussian spatial map
    3. Stack to dual-channel
    4. Add singleton Z dimension for nnU-Net 2D
    """
    # Normalize
    image = image_crop.astype(np.float32) / 255.0

    # Generate spatial attention
    spatial_map = generate_gaussian_spatial_map(image.shape, sigma=35)

    # Stack channels
    dual_channel = np.stack([image, spatial_map], axis=0)  # (2, H, W)

    # Add Z dimension for nnU-Net
    input_array = dual_channel[:, np.newaxis, :, :]  # (2, 1, H, W)

    return input_array
```

### 3. Inference

```python
def predict(input_array, predictor):
    """
    Run nnU-Net prediction

    Uses nnUNetPredictor for standardized inference
    """
    properties = {
        'spacing': [999.0, 1.0, 1.0],  # Z=999 for 2D
    }

    result = predictor.predict_single_npy_array(
        input_array,
        properties,
        save_or_return_probabilities=True
    )

    return result
```

### 4. Post-processing

#### Bifurcation Suppression

Yan dalları (side branches) kaldırarak sadece ana damarı korur:

```python
class CenterComponentKeeper:
    """
    Keep only the connected component closest to ROI center

    Rationale:
    - ROI user tarafından ana damar üzerine çizilir
    - Merkeze en yakın component ana damardır
    - Yan dallar genellikle kenarlara yakındır
    """

    def __init__(self, center_tolerance_radius=10):
        self.tolerance = center_tolerance_radius

    def process(self, binary_mask):
        # Find connected components
        # Keep component closest to center
        # Return cleaned mask
        pass
```

### 5. Coordinate Mapping

```python
def map_to_full_frame(mask_crop, original_shape, roi_offset):
    """
    Map ROI-level mask back to full frame coordinates
    """
    full_mask = np.zeros(original_shape, dtype=np.uint8)
    y_offset, x_offset = roi_offset
    h, w = mask_crop.shape
    full_mask[y_offset:y_offset+h, x_offset:x_offset+w] = mask_crop
    return full_mask
```

---

## Training Data

### Data Collection

```yaml
Source: [TO BE FILLED - e.g., Clinical angiography database]
Number of cases: [TO BE FILLED]
Number of frames: [TO BE FILLED]
Annotation method: [TO BE FILLED - e.g., Manual expert annotation]
Annotators: [TO BE FILLED]
Inter-observer variability: [TO BE FILLED]
```

### Data Preparation

```yaml
ROI extraction: Manual bounding box around vessel segment
Crop size: Variable (typically 150-300 pixels)
Preprocessing:
  - Grayscale conversion
  - Intensity normalization [0, 1]
  - Gaussian spatial map generation

Label format: Binary mask (0=background, 1=vessel)
```

### Train/Validation/Test Split

```yaml
Split strategy: [TO BE FILLED - e.g., Patient-level split]
Training: [TO BE FILLED]%
Validation: [TO BE FILLED]%
Test: [TO BE FILLED]%
```

---

## Validation Metrics

### Segmentation Accuracy

| Metric | Value | Notes |
|--------|-------|-------|
| Dice Score | [TO BE FILLED] | Primary metric |
| IoU (Jaccard) | [TO BE FILLED] | |
| Precision | [TO BE FILLED] | |
| Recall | [TO BE FILLED] | |
| HD95 | [TO BE FILLED] | 95th percentile Hausdorff Distance |

### Clinical Relevance

| Metric | Value | Notes |
|--------|-------|-------|
| Centerline accuracy | [TO BE FILLED] | Distance from ground truth |
| Diameter error | [TO BE FILLED] | For QCA validation |
| Edge detection accuracy | [TO BE FILLED] | Sub-pixel level |

---

## Comparison with Alternatives

### vs. AngioPy (InceptionResNetV2 + U-Net)

| Aspect | nnU-Net (Ours) | AngioPy |
|--------|----------------|---------|
| Input | ROI crop + Gaussian | Full frame + seed points |
| Architecture | Self-configured U-Net | Fixed InceptionResNetV2 |
| Training data | Custom dataset | Pre-trained |
| Seed points required | No | Yes (2-5 points) |
| Bifurcation handling | Post-processing | N/A |

### Design Decisions

**Why ROI-based instead of full-frame?**
1. Reduced computational cost
2. Higher resolution on target region
3. Spatial attention via Gaussian map
4. Better generalization to different frame sizes

**Why dual-channel input?**
1. Explicit spatial prior
2. Guides attention to ROI center
3. Improves accuracy on boundary regions
4. Reduces false positives at edges

---

## Model Files

```
models/
└── nnUNet_results/
    └── Dataset[XXX]_CoronaryROI/
        └── nnUNetTrainer__nnUNetPlans__2d/
            ├── dataset.json
            ├── plans.json
            └── fold_0/
                ├── checkpoint_best.pth
                └── checkpoint_latest.pth
```

---

## References

1. **Isensee, F., et al.** "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods* 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z

2. **Ronneberger, O., et al.** "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI* 2015.

3. **nnU-Net v2 Documentation:** https://github.com/MIC-DKFZ/nnUNet

---

## TODO: Information Needed

Aşağıdaki bilgiler model dokümantasyonunu tamamlamak için gereklidir:

- [ ] Dataset ID ve ismi
- [ ] Eğitim veri seti boyutu (hasta/frame sayısı)
- [ ] Eğitim parametreleri (epoch, batch size, learning rate)
- [ ] Validation metrikleri (Dice, IoU, etc.)
- [ ] Train/val/test split oranları
- [ ] Eğitim süresi ve donanım

---

*Last Updated: 2024-11-24*
*Status: DRAFT - Needs model-specific details*
