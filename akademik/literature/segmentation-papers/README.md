# Segmentasyon Literatürü

Bu klasör, koroner arter segmentasyonu ve nnU-Net ile ilgili temel makaleleri içerir.

## Temel Referanslar (Mutlaka Okunmalı)

### 1. nnU-Net Framework

**Isensee, F., et al. (2021)**
"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
*Nature Methods* 18, 203–211

- DOI: https://doi.org/10.1038/s41592-020-01008-z
- GitHub: https://github.com/MIC-DKFZ/nnUNet
- **Önemi:** Projemizin temel segmentasyon framework'ü

**Okunacak bölümler:**
- [ ] Methods: Network architecture
- [ ] Methods: Preprocessing pipeline
- [ ] Methods: Data augmentation
- [ ] Supplementary: Implementation details

### 2. U-Net (Temel Mimari)

**Ronneberger, O., et al. (2015)**
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
*MICCAI 2015*

- arXiv: https://arxiv.org/abs/1505.04597
- **Önemi:** nnU-Net'in temel aldığı encoder-decoder mimari

### 3. nnU-Net v2

**Isensee, F., et al. (2024)**
"nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation"
*arXiv preprint*

- arXiv: https://arxiv.org/abs/2404.09556
- **Önemi:** v2'deki güncellemeler ve best practices

---

## Koroner Arter Segmentasyonu

### Deep Learning Yaklaşımları

- [ ] **AngioPy papers** - EPFL coronary segmentation
- [ ] **DeepVessel** - 3D coronary segmentation
- [ ] **Attention U-Net** - Attention mechanisms for vessels

### Geleneksel Yöntemler (Karşılaştırma için)

- [ ] **Frangi filter** - Vesselness enhancement
- [ ] **Region growing** - Classical segmentation
- [ ] **Active contours** - Snake-based methods

---

## Spatial Attention Mechanisms

Dual-channel input (image + Gaussian spatial map) için teorik altyapı:

- [ ] **Attention mechanisms in medical imaging**
- [ ] **Spatial transformer networks**
- [ ] **Gaussian priors in segmentation**

---

## Validation & Metrics

- [ ] **Dice score** - Primary segmentation metric
- [ ] **Hausdorff distance** - Boundary accuracy
- [ ] **Centerline accuracy** - Vessel-specific metric

---

## Tarama Stratejisi

### Anahtar Kelimeler
```
"coronary artery segmentation" AND "deep learning"
"nnU-Net" AND "cardiovascular"
"angiography segmentation" AND "U-Net"
"vessel segmentation" AND "attention"
```

### Veritabanları
- PubMed
- IEEE Xplore
- arXiv (cs.CV, eess.IV)
- Google Scholar

---

## Dosya Adlandırma Kuralı

```
[yıl]-[ilk_yazar]-[kısa_başlık].md
```

Örnekler:
- `2021-isensee-nnunet.md`
- `2015-ronneberger-unet.md`

---

## İlerleme

| Makale | Okundu | Not Yazıldı | Projemize Etkisi |
|--------|--------|-------------|------------------|
| Isensee 2021 (nnU-Net) | [ ] | [ ] | Temel framework |
| Ronneberger 2015 (U-Net) | [ ] | [ ] | Mimari temeli |
| ... | | | |

---

*Son Güncelleme: 2024-11-24*
