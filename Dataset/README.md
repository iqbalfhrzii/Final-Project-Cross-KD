# 📊 Dataset RDD2022 (Road Damage Dataset 2022)

Repositori ini menggunakan dataset sekunder dari **Crowdsensing-based Road Damage Detection Challenge 2022 (RDD2022)** yang mencakup citra anomali permukaan jalan dari berbagai negara.

---

## 🔗 Tautan Unduh Dataset Lengkap
Karena ukuran dataset penuh mencapai **13,5 GB** (melebihi batas kuota penyimpanan repositori GitHub), dataset lengkap yang telah difilter dan diaugmentasi dapat diunduh langsung melalui Google Drive resmi penelitian berikut:

👉 **[Unduh Dataset RDD2022 Lengkap di Google Drive](https://drive.google.com/drive/folders/1UsrzEdO3pofb4GKN_XoiWIE5X5vvYEFK?usp=sharing)**

Berkas zip di Google Drive:
* `dataset_rdd2022_augmented.zip` (Dataset utama siap latih: 4 kelas + CLAHE + Augmentasi rotasi/flip/brightness).

---

## 🗂️ Struktur Direktori Dataset
Setelah berkas zip diekstrak, letakkan isi folder sehingga membentuk hierarki berikut di dalam folder `Dataset/`:

```text
Dataset/
├── data.yaml
├── README.md
├── images/
│   ├── train/     # 17.271 citra latihan
│   ├── val/       #  5.758 citra validasi
│   └── test/      #  5.757 citra pengujian
└── labels/
    ├── train/     # 17.271 file anotasi .txt
    ├── val/       #  5.758 file anotasi .txt
    └── test/      #  5.757 file anotasi .txt
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
