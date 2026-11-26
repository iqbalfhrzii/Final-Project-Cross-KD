# Deteksi Kerusakan Jalan Menggunakan Cross-Head Knowledge Distillation Berbasis YOLOv8 pada Dataset RDD2022

Repositori ini memuat kode sumber resmi dan konfigurasi dataset untuk penelitian Tugas Akhir Program Studi Informatika, Institut Teknologi Kalimantan (ITK).

* **Peneliti:** Iqbal Fahrozi (NIM: 11221034)  
* **Program Studi:** Informatika  
* **Jurusan:** Jurusan Teknik Elektro, Informatika, dan Bisnis (JTEIB)  
* **Fakultas:** Fakultas Sains dan Teknologi Informasi (FSTI)  
* **Institusi:** Institut Teknologi Kalimantan (ITK)  

---

## 📌 Ringkasan Penelitian

Penelitian ini mengusulkan penerapan teknik **Cross-Head Knowledge Distillation (CrossKD)** untuk mentransfer kapabilitas representasi fitur spasial dari model guru berkapasitas besar (*Teacher* - YOLOv8m, 25,8 juta parameter) ke model murid berarsitektur ringkas (*Student* - YOLOv8n, 3,2 juta parameter) dalam mendeteksi empat kategori anomali kerusakan jalan pada dataset multinasional **RDD2022**:
1. **D00:** Retak Memanjang (*Longitudinal Crack*)
2. **D10:** Retak Melintang (*Transverse Crack*)
3. **D20:** Retak Buaya (*Alligator Crack*)
4. **D40:** Lubang Jalan (*Pothole*)

### Temuan Utama:
* **Peningkatan Akurasi:** Model *Student CrossKD* berhasil meraih **mAP@50 sebesar 65,56%**, meningkat **+0,55%** dibandingkan *Student Baseline* mandiri (65,01%).
* **Retensi Performa Guru:** Model murid berhasil mempertahankan **96,94%** dari performa deteksi model *Teacher* (67,63% mAP@50), melampaui kriteria batas keberhasilan minimal 90%.
* **Efisiensi Komputasi Tinggi:** Model *Student CrossKD* menghemat **87,6% jumlah parameter** dan memangkas kebutuhan komputasi dari 79,1 GFLOPs menjadi hanya **8,7 GFLOPs**, sehingga sangat ideal untuk implementasi pada perangkat komputasi tepi (*edge computing* / sistem inspeksi jalan otomatis).

---

## 🗂️ Struktur Direktori Repositori

Struktur repositori ini disusun secara teratur menjadi dua folder utama:

```text
Final-Project-Cross-KD/
├── Dataset/
│   ├── data.yaml                     # Konfigurasi 4 kelas dataset RDD2022 untuk YOLOv8
│   └── README.md                     # Panduan pengunduhan dataset RDD2022 & struktur folder
│
├── Source Code/
│   ├── ultralytics/                  # Modifikasi framework YOLOv8 terintegrasi CrossKD
│   ├── OtherKD/                      # Modul metode pembanding Knowledge Distillation
│   ├── PretrainedModelFromOfficial/  # Bobot awal resmi (yolov8n.pt, yolov8s.pt, yolov8m.pt)
│   ├── crosskd_architecture.py       # Skrip visualisasi bagan arsitektur CrossKD
│   ├── drawbox.py                    # Utilitas penggambaran bounding box
│   ├── heatmap.py                    # Skrip visualisasi fitur & Grad-CAM
│   ├── main.py                       # Skrip utama pelatihan (Teacher, Baseline, CrossKD)
│   ├── predict.py                    # Skrip inferensi deteksi citra
│   ├── preprocess_and_augment.py     # Skrip peningkatan kontras CLAHE & augmentasi citra
│   ├── process_rdd2022.py            # Skrip filtering & remapping anotasi 4 kelas RDD2022
│   ├── requirements.txt              # Daftar pustaka & dependensi Python
│   ├── run_training.sh               # Skrip eksekusi pelatihan otomatis di lingkungan Linux/Cloud
│   ├── test_predict.py               # Skrip pengujian komparatif Teacher vs Baseline vs CrossKD
│   ├── train_colab.ipynb             # Notebook interaktif untuk eksekusi di Google Colab
│   └── val.py                        # Skrip evaluasi performa metrik model
│
├── .gitignore                        # Konfigurasi pengabaian berkas cache dan data besar
└── README.md                         # Dokumentasi utama proyek
```

---

## 🚀 Panduan Instalasi dan Menjalankan Program

### 1. Kloning Repositori
```bash
git clone https://github.com/iqbalfhrzii/Final-Project-Cross-KD.git
cd Final-Project-Cross-KD
```

### 2. Memasang Dependensi
Pastikan Python versi 3.10 atau yang lebih baru telah terpasang. Pasang pustaka pendukung dengan perintah:
```bash
cd "Source Code"
pip install -r requirements.txt
```

### 3. Persiapan Dataset
Dataset RDD2022 lengkap (13,5 GB) dapat diunduh melalui tautan Google Drive yang tercantum pada [Dataset/README.md](Dataset/README.md). Setelah diekstrak, letakkan folder `images/` dan `labels/` di dalam direktori `Dataset/`.

### 4. Menjalankan Pelatihan Model
Pelatihan model dikontrol melalui berkas `Source Code/main.py`. Anda dapat memilih mode pelatihan dengan mengubah variabel `STAGE` atau menggunakan argumen:

* **Melatih Model Teacher Optimized:**
  ```bash
  python main.py --stage teacher_opt
  ```
* **Melatih Model Student Baseline (Mandiri tanpa KD):**
  ```bash
  python main.py --stage student_baseline
  ```
* **Melatih Model Student dengan Bimbingan CrossKD:**
  ```bash
  python main.py --stage student_kd
  ```

### 5. Menguji Deteksi pada Citra (Inference)
Untuk menguji hasil prediksi deteksi pada citra baru secara kualitatif:
```bash
python test_predict.py --model crosskd --source path/to/image.jpg --conf 0.25
```

---

## 📊 Ringkasan Hasil Eksperimen

| Model | Arsitektur | Parameter | GFLOPs | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher Standar (Exp 1)** | YOLOv8m | 25,8 M | 79,1 | 65,12% | 58,21% | 66,54% | 37,80% |
| **Teacher Optimized (Exp 2)** | YOLOv8m | 25,8 M | 79,1 | 67,66% | 63,01% | **67,63%** | 38,47% |
| **Student Baseline (Tanpa KD)** | YOLOv8n | 3,2 M | 8,7 | 66,55% | 59,57% | 65,01% | 36,03% |
| **Student CrossKD (Hasil Usulan)**| **YOLOv8n** | **3,2 M** | **8,7** | **67,17%** | **59,75%** | **65,56%** | **36,27%** |

