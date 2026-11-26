# Dataset RDD2022 (Preprocessed & Augmented)

Direktori ini menyimpan konfigurasi dan panduan penyiapan **Road Damage Dataset 2022 (RDD2022)** yang telah melalui tahapan pemfilteran kelas, pra-pengolahan citra (*CLAHE & Resize*), serta augmentasi data *offline* untuk penelitian Tugas Akhir.

---

## 🔗 Tautan Unduh Dataset Lengkap

Karena ukuran dataset penuh mencapai **13,5 GB** (melebihi batas penyimpanan standar repositori GitHub), dataset lengkap yang telah diproses dan siap latih dapat diunduh melalui Google Drive resmi penelitian berikut:

👉 **[Unduh Dataset RDD2022 Lengkap di Google Drive](https://drive.google.com/drive/folders/1UsrzEdO3pofb4GKN_XoiWIE5X5vvYEFK?usp=sharing)**

Berkas arsip di Google Drive:
- `dataset_rdd2022_augmented.zip` — Dataset siap latih (4 kelas utama + CLAHE + Resize 640×640 + Augmentasi *offline* pada split *train*).

---

## 📁 Struktur Direktori Dataset

Setelah berkas `.zip` diekstrak, letakkan folder `images/` dan `labels/` ke dalam direktori `Dataset/` sehingga membentuk struktur berikut:

```text
Dataset/
├── data.yaml             # Konfigurasi path dan daftar 4 kelas untuk YOLOv8
├── README.md             # Dokumentasi dataset
├── images/
│   ├── train/            # 62.673 citra latih (pasca-augmentasi offline)
│   ├── val/              # 5.758 citra validasi (murni tanpa augmentasi)
│   └── test/             # 5.758 citra pengujian (murni tanpa augmentasi)
└── labels/
    ├── train/            # 62.673 berkas anotasi .txt (142.504 bounding box)
    ├── val/              # 5.758 berkas anotasi .txt (9.366 bounding box)
    └── test/             # 5.758 berkas anotasi .txt (9.324 bounding box)
```

---

## 🏷️ Spesifikasi 4 Kelas Kerusakan Jalan
Dataset asli RDD2022 yang terdiri dari 5 kelas diproses ulang menggunakan skrip `process_rdd2022.py` untuk mengeliminasi kelas ke-5 (*Other/Repair*) dan memetakan ulang (*remapping*) label menjadi 4 kategori utama:

| Class ID | Kode Kerusakan | Nama Kerusakan (Inggris) | Nama Kerusakan (Indonesia) | Ciri Visual Utama |
| :---: | :---: | :--- | :--- | :--- |
| **0** | **D00** | *Longitudinal Crack* | Retak Memanjang | Alur retakan linear sejajar arah jalur lalu lintas |
| **1** | **D10** | *Transverse Crack* | Retak Melintang | Alur retakan linear tegak lurus arah jalur lalu lintas |
| **2** | **D20** | *Alligator Crack* | Retak Buaya | Pola retak jaring poligonal menyerupai kulit buaya |
| **3** | **D40** | *Pothole* | Lubang Jalan | Cekungan struktural aspal dengan bayangan depresi |

---

## ⚙️ Skrip Pengolahan Data
Jika Anda memiliki data mentah RDD2022 dan ingin mereplikasi proses pemfilteran dan augmentasi secara mandiri:
1. **Pemfilteran & Remapping 4 Kelas:**  
   Jalankan `python "Source Code/process_rdd2022.py"`
2. **Peningkatan Kontras CLAHE & Augmentasi:**  
   Jalankan `python "Source Code/preprocess_and_augment.py"`
