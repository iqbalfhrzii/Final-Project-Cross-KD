# Pengembangan Model Lightweight YOLOv8 untuk Deteksi Kerusakan Jalan Menggunakan Cross-Head Knowledge Distillation

Repositori ini memuat kode sumber resmi, konfigurasi eksperimen, dan dokumentasi dataset untuk penelitian Tugas Akhir Program Studi Informatika, Institut Teknologi Kalimantan (ITK).

- **Peneliti:** Iqbal Fahrozi (NIM: 11221034)
- **Dosen Pembimbing Utama:** Bima Prihasto, S.Si., M.Si., Ph.D.
- **Dosen Pembimbing Pendamping:** Boby Mugi Pratama, S.Si., M.Han.
- **Program Studi:** Informatika
- **Jurusan:** Jurusan Teknik Elektro, Informatika, dan Bisnis (JTEIB)
- **Fakultas:** Fakultas Sains dan Teknologi Informasi (FSTI)
- **Institusi:** Institut Teknologi Kalimantan (ITK)

---

## Ringkasan Penelitian

Penelitian ini mengimplementasikan metode **Cross-Head Knowledge Distillation (CrossKD)** untuk mentransfer representasi fitur spasial dan prediksi *detection head* dari model guru berkapasitas besar (*Teacher* - **YOLOv8m**, 25,8 juta parameter) ke model murid berarsitektur ringkas (*Student* - **YOLOv8n**, 3,0 juta parameter) dalam mendeteksi empat kategori kerusakan jalan pada dataset multinasional **RDD2022**:

1. **D00:** Retak Memanjang (*Longitudinal Crack*)
2. **D10:** Retak Melintang (*Transverse Crack*)
3. **D20:** Retak Buaya (*Alligator Crack*)
4. **D40:** Lubang Jalan (*Pothole*)

### Temuan Utama:
- **Peningkatan Akurasi Deteksi:** Model *Student CrossKD* meraih **mAP@50 sebesar 65,56%** dan **mAP@50-95 sebesar 36,27%**, meningkat **+0,55%** dibandingkan model *Student Baseline* yang dilatih mandiri tanpa distilasi (65,01% mAP@50).
- **Retensi Performa Model Guru:** Model murid mampu mempertahankan **96,94%** performa deteksi model *Teacher Optimized* (67,63% mAP@50), melampaui target minimal retensi performa sebesar 90%.
- **Efisiensi Komputasi Tinggi:** Model *Student CrossKD* memangkas jumlah parameter sebesar **88,4%** (dari 25,8 juta menjadi **3,0 juta parameter**) dan mengurangi beban komputasi dari 78,7 GFLOPs menjadi **8,1 GFLOPs**, sehingga sangat layak diimplementasikan pada perangkat dengan keterbatasan sumber daya komputasi (*edge devices*).

---

## Struktur Direktori Repositori

```text
Final-Project-Cross-KD/
├── Dataset/
│   ├── data.yaml                        # Konfigurasi 4 kelas dataset RDD2022 untuk YOLOv8
│   └── README.md                        # Panduan unduh dataset RDD2022 & statistik data
│
├── Source Code/
│   ├── ultralytics/                     # Modifikasi framework YOLOv8 terintegrasi CrossKD
│   ├── OtherKD/                         # Modul metode pembanding Knowledge Distillation
│   ├── PretrainedModelFromOfficial/     # Bobot awal resmi (yolov8n.pt, yolov8s.pt, yolov8m.pt)
│   ├── crosskd_architecture.py          # Skrip visualisasi bagan arsitektur CrossKD
│   ├── drawbox.py                       # Utilitas penggambaran bounding box
│   ├── heatmap.py                       # Skrip visualisasi fitur & Grad-CAM
│   ├── main.py                          # Skrip utama pelatihan (Teacher, Baseline, CrossKD)
│   ├── predict.py                       # Skrip inferensi deteksi citra
│   ├── preprocess_and_augment.py        # Skrip enhancement CLAHE (LAB) & augmentasi flip
│   ├── process_rdd2022.py               # Skrip filtering & remapping anotasi 4 kelas RDD2022
│   ├── requirements.txt                 # Daftar pustaka & dependensi Python
│   ├── run_training.sh                  # Skrip eksekusi pelatihan otomatis di lingkungan Linux/Cloud
│   ├── test_predict.py                  # Skrip pengujian komparatif Teacher vs Baseline vs CrossKD
│   ├── train_colab.ipynb                # Notebook interaktif untuk eksekusi di Google Colab
│   └── val.py                           # Skrip evaluasi metrik performa model
│
├── .gitignore                           # Konfigurasi pengabaian berkas cache dan data berukuran besar
└── README.md                            # Dokumentasi utama proyek
```

---

## Panduan Instalasi dan Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/iqbalfhrzii/Final-Project-Cross-KD.git
cd Final-Project-Cross-KD
```

### 2. Pemasangan Dependensi
Pastikan lingkungan Python 3.10+ dan PyTorch dengan dukungan CUDA telah terpasang. Pasang seluruh pustaka pendukung melalui perintah:
```bash
cd "Source Code"
pip install -r requirements.txt
```

### 3. Penyiapan Dataset
Unduh dataset RDD2022 yang telah diproses (13,5 GB) melalui tautan Google Drive yang tersedia pada [Dataset/README.md](Dataset/README.md), lalu ekstrak folder `images/` dan `labels/` ke dalam direktori `Dataset/`.

### 4. Konfigurasi & Pelatihan Model
Seluruh eksperimen dilatih menggunakan konfigurasi standar (`imgsz=640`, `epochs=100`, `batch=16`, `optimizer=SGD`, `lr0=0.01`). Jalankan pelatihan melalui `Source Code/main.py` sesuai skenario berikut:

* **Melatih Model Teacher Optimized (YOLOv8m):**
  ```bash
  python main.py --stage teacher_opt
  ```
* **Melatih Model Student Baseline (YOLOv8n tanpa KD):**
  ```bash
  python main.py --stage student_baseline
  ```
* **Melatih Model Student dengan Bimbingan CrossKD (YOLOv8n + CrossKD):**
  ```bash
  python main.py --stage student_kd
  ```

### 5. Pengujian Inferensi & Evaluasi
* **Evaluasi Metrik Kuantitatif (`val.py`):**
  ```bash
  python val.py
  ```
* **Pengujian Prediksi Visual Komparatif (`test_predict.py`):**
  ```bash
  python test_predict.py path/to/image.jpg --model crosskd --conf 0.25
  ```

---

## Ringkasan Hasil Eksperimen

Evaluasi dilakukan pada partisi validasi murni dataset RDD2022 (5.758 citra) dengan ukuran masukan `640 × 640` piksel (merujuk pada Tabel 4.3 dan Tabel 4.4 Laporan Tugas Akhir untuk kategori *All Classes*):

| Model | Arsitektur | Parameter | GFLOPs | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher Standar (Exp 1)** | YOLOv8m | 25,8 M | 78,7 | 65,12% | 58,21% | 66,54% | 37,80% |
| **Teacher Optimized (Exp 2)** | YOLOv8m | 25,8 M | 78,7 | 67,66% | 63,01% | **67,63%** | **38,47%** |
| **Student Baseline (Tanpa KD)** | YOLOv8n | 3,0 M | 8,1 | 66,55% | 59,76% | 65,01% | 36,03% |
| **Student CrossKD (Usulan)** | **YOLOv8n** | **3,0 M** | **8,1** | **67,17%** | **59,75%** | **65,56%** | **36,27%** |

---

## Lingkungan Komputasi Eksperimen

- **GPU:** NVIDIA GeForce RTX 4090 (24 GB GDDR6X)
- **CPU:** AMD Ryzen 9 7950X3D (16 Cores, 32 Threads)
- **RAM:** 64 GB DDR5
- **Framework:** Python 3.10, PyTorch, Ultralytics YOLOv8 (Modified with CrossKD)

---

## Referensi

- **Metode & Teori CrossKD:** Wang, J., Chen, Y., Zheng, Z., Li, X., Cheng, M.-M., & Hou, Q. (2024). *CrossKD: Cross-Head Knowledge Distillation for Object Detection* (arXiv:2306.11369 / [jbwang1997/CrossKD](https://github.com/jbwang1997/CrossKD)).
- **Basis Implementasi YOLOv8-KD:** Adaptasi struktur kode integrasi Knowledge Distillation pada YOLOv8 (`ultralytics/` dan `OtherKD/`) merujuk pada repositori [KefanZhan/YOLOv8-KD](https://github.com/KefanZhan/YOLOv8-KD) (*Knowledge Distillation via Cross Supervising with Attention for Remote Sensing Object Detection*, Zhan et al., 2025), dengan fokus penggunaan modul `CrossKD` untuk skenario penelitian ini.
