# Akademik Çalışma Alanı

Bu klasör, Coronary RWS Analyser projesinin akademik yayın sürecini desteklemek için oluşturulmuştur.

## Klasör Yapısı

```
akademik/
├── literature/                    # Literatür taraması
│   ├── rws-papers/               # RWS ile ilgili makaleler
│   ├── qca-papers/               # QCA metodolojisi makaleleri
│   ├── segmentation-papers/      # Vessel segmentasyon makaleleri
│   ├── tracking-papers/          # Object tracking makaleleri
│   └── similar-software/         # Benzer yazılımlar hakkında makaleler
│
├── notes/                         # Araştırma notları
│   └── [konu-bazlı .md dosyaları]
│
├── figures/                       # Makale için figürler
│   ├── architecture/             # Sistem mimarisi diyagramları
│   ├── workflow/                 # İş akışı diyagramları
│   ├── results/                  # Sonuç grafikleri
│   └── screenshots/              # Uygulama ekran görüntüleri
│
└── manuscript/                    # Makale taslağı
    ├── draft.md                  # Ana makale taslağı
    ├── abstract.md               # Özet
    ├── introduction.md           # Giriş
    ├── methods.md                # Metodoloji
    ├── results.md                # Sonuçlar
    ├── discussion.md             # Tartışma
    ├── supplementary.md          # Ek materyaller
    └── references.bib            # Kaynakça (BibTeX)
```

## Hedef Dergi/Platform

### Birincil Hedef: JOSS (Journal of Open Source Software)
- Açık kaynak yazılım odaklı
- Kod kalitesi ve dokümantasyon önemli
- Hızlı review süreci
- DOI ile citable

### Alternatif Hedefler
- SoftwareX
- Medical Image Analysis
- Computer Methods and Programs in Biomedicine

## Literatür Tarama Stratejisi

### Anahtar Kelimeler
- "radial wall strain" + "coronary"
- "coronary artery compliance"
- "quantitative coronary angiography" + "software"
- "vessel segmentation" + "deep learning"
- "open source" + "medical imaging"

### Veritabanları
- PubMed
- IEEE Xplore
- Google Scholar
- arXiv (preprints)

## Makale Yapısı (JOSS Format)

1. **Summary** (250 words max)
2. **Statement of Need**
3. **Key Features**
4. **Acknowledgements**
5. **References**

## İlerleme Takibi

| Bölüm | Durum | Son Güncelleme |
|-------|-------|----------------|
| Literature Review | Başlanmadı | - |
| Introduction | Başlanmadı | - |
| Methods | Başlanmadı | - |
| Results | Beklemede | - |
| Discussion | Beklemede | - |
| Figures | Başlanmadı | - |

## Katkıda Bulunma

Her literatür notu için:
1. `literature/[kategori]/` altında `.md` dosyası oluştur
2. Dosya adı: `[yıl]-[ilk_yazar]-[kısa_başlık].md`
3. Standart template kullan (aşağıda)

### Literatür Notu Template
```markdown
# [Makale Başlığı]

**Yazarlar:** [Yazar listesi]
**Dergi:** [Dergi adı]
**Yıl:** [Yayın yılı]
**DOI:** [DOI linki]

## Özet
[Kendi özetiniz]

## Anahtar Bulgular
- Bulgu 1
- Bulgu 2

## Metodoloji
[Kullanılan yöntemler]

## Projemizle İlişkisi
[Bu makale projemize nasıl katkı sağlar?]

## Alıntılanabilir Cümleler
> "Doğrudan alıntı" (sayfa no)

## Notlar
[Ek düşünceler]
```

---

*Bu alan, proje geliştirme süreciyle paralel olarak güncellenecektir.*
