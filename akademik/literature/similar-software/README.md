# Benzer Yazılımlar - Literatür Taraması

Bu klasör, koroner anjiografi analizi yapan mevcut yazılımları ve bunlarla ilgili yayınları içerir.

## Ticari Yazılımlar

### 1. QAngio XA (Medis Medical Imaging)
- **Tip:** Ticari, kapalı kaynak
- **Özellikler:** QCA, bifurkasyon analizi
- **Website:** medisimaging.com
- **Makaleler:** [Eklenecek]

### 2. CAAS (Pie Medical Imaging)
- **Tip:** Ticari, kapalı kaynak
- **Özellikler:** QCA, 3D reconstruction
- **Website:** piemedicalimaging.com
- **Makaleler:** [Eklenecek]

### 3. AngioPlus (CardioVascular Imaging Systems)
- **Tip:** Ticari
- **Özellikler:** QCA
- **Makaleler:** [Eklenecek]

## Açık Kaynak/Akademik Yazılımlar

### 1. AngioPy
- **Tip:** Açık kaynak
- **Özellikler:** Sadece segmentasyon
- **GitLab:** gitlab.com/epfl-center-for-imaging/angiopy
- **Limitasyon:** QCA yok, RWS yok

### 2. 3D Slicer (CTA için)
- **Tip:** Açık kaynak
- **Özellikler:** Genel medikal görüntüleme
- **Limitasyon:** Anjiografi-spesifik değil

## Karşılaştırma Tablosu

| Özellik | QAngio | CAAS | AngioPy | **Bizim Proje** |
|---------|--------|------|---------|-----------------|
| Açık Kaynak | ❌ | ❌ | ✅ | ✅ |
| QCA | ✅ | ✅ | ❌ | ✅ |
| RWS | ❌* | ❌ | ❌ | ✅ |
| Segmentasyon | ✅ | ✅ | ✅ | ✅ |
| Tracking | ✅ | ✅ | ❌ | ✅ |
| Ücretsiz | ❌ | ❌ | ✅ | ✅ |

*QAngio'nun RWS desteği olup olmadığı doğrulanmalı

## Bizim Farklılığımız

1. **İlk açık kaynak RWS aracı**
2. **Akademik şeffaflık** - Tüm algoritmalar dokümante
3. **Ücretsiz** - Araştırmacılar için erişilebilir
4. **Modern stack** - Deep learning segmentasyon

## Taranacak Makaleler

- [ ] "Comparison of quantitative coronary angiography software"
- [ ] "Validation of QCA software" çalışmaları
- [ ] "Open source medical imaging software" derlemeleri
- [ ] JOSS'ta yayınlanmış benzer projeler

---

*Her bulunan makale için ayrı bir .md dosyası oluşturulacak*
