# Duygu Tanıma Sistemi (Emotion Recognition) - Model Karşılaştırması

## 📋 İçindekiler
1. [Proje Özeti](#proje-özeti)
2. [Model Mimarileri](#model-mimarileri)
3. [Test Sonuçları](#test-sonuçları)
4. [Detaylı Karşılaştırma](#detaylı-karşılaştırma)
5. [Sonuç ve Öneriler](#sonuç-ve-öneriler)

---

## 🎯 Proje Özeti

Bu projede iki farklı derin öğrenme mimarisi kullanarak duygu tanıma sistemi geliştirilmiştir:
- **ResNet18** (Pretrained, Full Fine-Tune)
- **Custom CNN** (Sıfırdan eğitilmiş)

Her iki model de **5 duygu sınıfı** ile eğitilmiştir:
- Angry (Kızgın)
- Fear (Korkmuş)
- Happy (Mutlu)
- Sad (Üzgün)
- Surprise (Şaşırmış)

---

## 🏗️ Model Mimarileri

### 1. ResNet18 (Full Fine-Tune)

**Özellikler:**
- Pre-trained ImageNet ağırlıkları ile başlatılmış
- **Tüm parametreler eğitime açık** (Full Fine-Tune)
- İlk conv katmanı 1 kanallı (grayscale) olarak değiştirilmiş
- Son FC katmanında Dropout (0.3) kullanılmış

**Mimarı:**
```
ResNet18 (pretrained=True)
├── Conv1: 1 → 64 channels (Grayscale input için adapte)
├── Layer1-4: Residual blocks
├── Adaptive Average Pooling
└── FC: 512 → 5 classes (with Dropout 0.3)
```

**Eğitim Parametreleri:**
- Optimizer: AdamW
- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Scheduler: ReduceLROnPlateau (factor=0.1, patience=3)
- Batch Size: 64
- Epochs: 50 (Early Stopping: patience=12)

---

### 2. Custom CNN

**Özellikler:**
- Sıfırdan eğitilmiş (pre-training yok)
- Batch Normalization ve Dropout ile regularization
- 6 Convolutional katman + 2 FC katman
- Daha basit ancak etkili mimari

**Mimarı:**
```
ConvBlock 1:
  Conv2d(1, 32, 3) → BatchNorm → ReLU
  Conv2d(32, 32, 3) → BatchNorm → ReLU
  MaxPool2d(2) 

ConvBlock 2:
  Conv2d(32, 64, 3) → BatchNorm → ReLU
  Conv2d(64, 64, 3) → BatchNorm → ReLU
  MaxPool2d(2) + Dropout(0.3)  

ConvBlock 3:
  Conv2d(64, 128, 3) → BatchNorm → ReLU
  MaxPool2d(2) 

ConvBlock 4:
  Conv2d(128, 128, 3) → BatchNorm → ReLU
  MaxPool2d(2) 

FC Layers:
  Flatten(128 × 14 × 14 = 25,088)
  Linear(25,088, 512) → BatchNorm → Dropout(0.5)
  Linear(512, 5)
```

**Eğitim Parametreleri:**
- Optimizer: AdamW
- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Scheduler: ReduceLROnPlateau (factor=0.1, patience=3)
- Batch Size: 64
- Epochs: 50 (Early Stopping: patience=10)

---

## 📊 Test Sonuçları

### Veri Seti Bilgileri

**Eğitim/Validasyon/Test Split:** 70% / 10% / 20%
- Training: 41,369 görüntü
- Validation: 5,910 görüntü
- Original Test Set: 11,820 görüntü
- FER2013 External Test Set: ~5,850 görüntü

---

## 🔍 Detaylı Karşılaştırma

### ResNet18 - Test Seti Sonuçları

#### Genel Metrikleri
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 91.31% |
| **Balanced Accuracy** | 90.47% |
| **Weighted Avg Precision** | 91.28% |
| **Weighted Avg Recall** | 91.31% |
| **Weighted Avg F1-Score** | 91.28% |

#### Sınıf Bazlı Performans
| Sınıf | Precision | Recall | F1-Score | Sensitivity | Specificity |
|-------|-----------|--------|----------|-------------|-------------|
| Angry | 87.80% | 89.36% | 88.57% | 89.36% | 97.43% |
| Fear | 88.33% | 83.62% | 85.91% | 83.62% | 97.82% |
| Happy | 95.67% | 96.97% | 96.31% | 96.97% | 98.01% |
| Sad | 88.32% | 89.45% | 88.88% | 89.45% | 96.81% |
| Surprise | 93.75% | 92.95% | 93.35% | 92.95% | 99.00% |

---

### CNN - Original Test Seti Sonuçları

#### Genel Metrikleri
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 87.43% |
| **Balanced Accuracy** | 86.75% |
| **Weighted Avg Precision** | 87.45% |
| **Weighted Avg Recall** | 87.43% |
| **Weighted Avg F1-Score** | 87.39% |

#### Sınıf Bazlı Performans
| Sınıf | Precision | Recall | F1-Score | Sensitivity | Specificity |
|-------|-----------|--------|----------|-------------|-------------|
| Angry | 82.64% | 84.19% | 83.41% | 84.19% | 96.33% |
| Fear | 86.47% | 78.83% | 82.47% | 78.83% | 97.57% |
| Happy | 93.35% | 93.30% | 93.33% | 93.30% | 96.99% |
| Sad | 84.43% | 84.47% | 84.45% | 84.47% | 95.80% |
| Surprise | 85.90% | 92.95% | 89.28% | 92.95% | 97.53% |

---

### CNN - External FER2013 Test Seti Sonuçları

#### Genel Metrikleri
| Metrik | Değer |
|--------|-------|
| **Accuracy** | 94.74% |
| **Balanced Accuracy** | 94.39% |
| **Weighted Avg Precision** | 94.78% |
| **Weighted Avg Recall** | 94.74% |
| **Weighted Avg F1-Score** | 94.71% |

#### Sınıf Bazlı Performans
| Sınıf | Precision | Recall | F1-Score | Sensitivity | Specificity |
|-------|-----------|--------|----------|-------------|-------------|
| Angry | 91.40% | 95.41% | 93.36% | 95.41% | 98.24% |
| Fear | 95.09% | 86.91% | 90.82% | 86.91% | 99.04% |
| Happy | 97.64% | 97.80% | 97.72% | 97.80% | 98.97% |
| Sad | 93.88% | 94.71% | 94.29% | 94.71% | 98.32% |
| Surprise | 93.52% | 97.12% | 95.28% | 97.12% | 98.88% |

---

## 📈 Karşılaştırmalı Analiz

### 1. Accuracy Karşılaştırması

```
ResNet18 (Test):           91.31% ████████████████████
CNN (Original Test):       87.43% █████████████████
CNN (FER2013 External):    94.74% ███████████████████████
```

### 2. Balanced Accuracy (Sınıf Dengesi) Karşılaştırması

```
ResNet18 (Test):           90.47% ████████████████████
CNN (Original Test):       86.75% █████████████████
CNN (FER2013 External):    94.39% ███████████████████████
```

### 3. Model Performans Özeti

| Model | Test Seti | Accuracy | Bal. Accuracy | Avr. Sensitivity | Avr. Specificity |
|-------|-----------|----------|---------------|------------------|------------------|
| **ResNet18** | Original | **91.31%** | **90.47%** | **90.47%** | **97.81%** |
| **CNN** | Original | 87.43% | 86.75% | 86.75% | 96.84% |
| **CNN** | FER2013 External | 94.74% | 94.39% | 94.39% | 98.69% |

### 4. Sınıf Bazlı Performans Farklılıkları

#### ResNet18 vs CNN (Original Test Set)

| Sınıf | ResNet18 Acc. | CNN Acc. | Fark |
|-------|---------------|----------|------|
| Angry | 89.36% | 84.19% | **+5.17%** ✅ |
| Fear | 83.62% | 78.83% | **+4.79%** ✅ |
| Happy | 96.97% | 93.30% | **+3.67%** ✅ |
| Sad | 89.45% | 84.47% | **+4.98%** ✅ |
| Surprise | 92.95% | 92.95% | **0.00%** (Equal) |

**ResNet18 tüm sınıflarda daha iyi veya eşit performans göstermektedir.**

#### CNN Original vs CNN FER2013 (Same Model, Different Test Set)

| Metrik | Original Test | FER2013 External | İyileşme |
|--------|---------------|------------------|----------|
| Accuracy | 87.43% | 94.74% | **+7.31%** ✅ |
| Bal. Accuracy | 86.75% | 94.39% | **+7.64%** ✅ |

**FER2013 External test seti üzerinde CNN daha iyi performans göstermektedir.**

---

## 🎓 İlginç Bulgular

### 1. **Pre-training'in Gücü**
- ResNet18 (pre-trained) diğer tüm test setlerinde CNN'i önceden eğitilmiş ağırlıklarla geçmiştir
- Pre-trained ImageNet ağırlıkları duygu tanıma için etkili transfer learning sağlamaktadır

### 2. **External Dataset Performansı**
- CNN, FER2013 external test setinde original test setine kıyasla **7.31% daha iyi** performans göstermiştir
- Bu, CNN'in daha genelleştirilebilir özellikler öğrendiğini gösterebilir

### 3. **Happy Sınıfı - En Yüksek Performans**
- Tüm modellerde Happy sınıfı en yüksek accuracy değerine sahiptir
- ResNet18: 96.97%, CNN (Original): 93.30%, CNN (FER2013): 97.80%
- **Sebep:** Happy ifadesi diğer duygulardan daha belirgin karakteristik özellikleri içermektedir

### 4. **Fear Sınıfı - En Düşük Performans**
- Tüm modellerde Fear sınıfı en düşük recall/sensitivity değerine sahiptir
- ResNet18: 83.62%, CNN (Original): 78.83%, CNN (FER2013): 86.91%
- **Sebep:** Fear ifadesi diğer duygularla visual olarak benzer özellikler içerebilmektedir

### 5. **Specificity Değerleri**
- Tüm modellerin specificity değerleri **96%+** ile oldukça yüksektir
- Bu, false positive oranının düşük olduğunu gösterir (başka duygular yanlış tanınmıyor)

---

## 🏆 Hangisi Daha İyi?

### **Genel Sonuç: ResNet18 KAZANIR ✅**

**Nedenler:**

1. **Test Doğruluğu** 
   - ResNet18: 91.31%
   - CNN: 87.43%
   - **Fark: +3.88%** ✅

2. **Dengeli Performans**
   - ResNet18 balanced accuracy: 90.47%
   - CNN: 86.75%
   - **Tüm sınıflarda tutarlı performans** ✅

3. **Pre-training Avantajı**
   - ImageNet'de pre-trained ağırlıklar transfer learning'i hızlandırmış
   - Daha hızlı yakınsamaya sebep olmuş
   - Daha iyi genelleştirme sağlamış

4. **Specificity (Özgüllük)**
   - ResNet18: 97.81% (average)
   - CNN: 96.84% (average)
   - **False positive oranı daha düşük** ✅

5. **Sınıf Dengesi**
   - ResNet18, Fear sınıfında CNN'den 4.79% daha iyi
   - CNN'nin zayıf olduğu sınıfları ResNet18 daha başarılı tanıyor

---

### **CNN'nin Güçlü Yönleri:**

1. **External Veri Uyumu**
   - FER2013 external test setinde 94.74% accuracy
   - Dış veri setleri üzerinde iyi genelleştirme yapabiliyor

2. **Basitlik**
   - Custom mimari, daha az parametre
   - Hızlı eğitim ve inference

3. **Overfitting Risk Düşük**
   - Sıfırdan eğitilen model daha az overfitting riski

---

## 🔧 Teknik Detaylar

### Veri Ön İşleme (Her İki Model)
```
- Input Size: 224 × 224 pixels
- Format: Grayscale (1 channel)
- Normalization: mean=[0.5], std=[0.5]
- Augmentation: Hayır (sadece training'de shuffle)
```

### Regularization Teknikleri

**ResNet18:**
- Dropout: 0.3 (Final FC layer)
- Weight Decay: 1e-4
- Early Stopping: patience=12

**CNN:**
- Dropout: 0.3 (Middle), 0.5 (Final FC)
- Batch Normalization (6 conv + 1 FC layer)
- Weight Decay: 1e-4
- Early Stopping: patience=10

---

## 📋 Sonuç ve Öneriler

### **Sonuçlar:**

1. **ResNet18 production için tavsiye edilir**
   - Daha yüksek accuracy (%91.31)
   - Daha dengeli performans
   - Pre-training avantajı

2. **CNN light-weight uygulama için uygun**
   - Edge devices için hızlı inference
   - Daha az bellek kullanımı
   - Yine de %87%+ accuracy

3. **Her iki model da practical kullanıma hazır**
   - %85%+ accuracy başarılı duygu tanıma için yeterli
   - Specificity değerleri yüksek (false positives az)

### **İyileştirme Önerileri:**

1. **Data Augmentation Ekle**
   - Rotation, Flip, Brightness adjustment
   - Model robustness'ını arttırır

2. **Ensemble Metodu Kullan**
   - ResNet18 + CNN kombinasyonu
   - Daha yüksek accuracy için voting mekanizması

3. **Hyperparameter Tuning**
   - Learning rate optimization
   - Batch size eksperimentleri
   - Farklı optimizers (SGD, RMSprop)

4. **Class Imbalance Çözümü**
   - Weighted loss function
   - Resampling teknikleri
   - SMOTE (Synthetic Minority Over-sampling)

5. **Model Interpretability**
   - Grad-CAM, LIME
   - Model'in ne öğrendiğini anlamak
   - Feature importance analizi

---

## 📸 Visualizations

### Confusion Matrices
- Tüm modellerin confusion matrix'leri `.png` dosyaları olarak kaydedilmiştir
- Sınıflar arası confusion pattern'ları görmek için kontrol ediniz

### Grad-CAM Visualizations
- Modellerin hangi görüntü bölgelerine odaklandığını görmek için `gradcam_*.png` dosyalarını kontrol ediniz

---

## 📁 Proje Dosyaları

```
ödev2/
├── 4_01_12_2025_resnet_18_yeni_91.ipynb    (ResNet18 Model)
├── 5_cnn_fer2013_test.ipynb                (CNN Model)
├── resnet18_full_finetune_emotion_model.pth (ResNet18 Weights)
├── cnn_emotion_model.pth                    (CNN Weights)
├── test_results_*.json                      (Test Metrikleri)
├── training_plots.png                       (Eğitim Grafikleri)
├── confusion_matrix_*.png                   (Confusion Matrices)
├── gradcam_*.png                            (Grad-CAM Visualizations)
└── README.md                                (Bu Dosya)
```

---

## 👨‍💻 Geliştirici

Emotion Recognition System - Kubra Karadumanzor
Tarih: Aralık 2025

---

## 📝 Notlar

- Tüm sonuçlar 5 duygu sınıfı üzerinde hesaplanmıştır
- Train/Val/Test split: 70/10/20
- Batch size: 64
- Input resolution: 224×224 pixels
- Eğitim, Google Colab GPU üzerinde yapılmıştır

---

**EN İYİ MODEL: ResNet18 (Full Fine-Tune) - Accuracy: 91.31% ✅**

# Human_Face_Emotation
