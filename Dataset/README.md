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

Dataset mentah RDD2022 yang semula memuat kelas tambahan (*Other Corruption / Repair*) diproses ulang menggunakan skrip `Source Code/process_rdd2022.py` untuk mengeliminasi kelas di luar fokus penelitian dan memetakan ulang (*remapping*) indeks kelas ke rentang `0–3` sesuai standar *benchmark* internasional RDD2022:

| Class ID | Kode Kelas | Nama Kerusakan | Deskripsi Fisik |
| :---: | :---: | :--- | :--- |
| **0** | **D00** | *Longitudinal Crack* | Retakan linier memanjang searah sumbu/lajur jalan |
| **1** | **D10** | *Transverse Crack* | Retakan linier melintang tegak lurus memotong sumbu jalan |
| **2** | **D20** | *Alligator Crack* | Retakan menyerupai sisik buaya (jaring retak saling terhubung) |
| **3** | **D40** | *Pothole* | Lubang jalan atau cekungan degradasi permukaan aspal |

*(Catatan: Citra yang tidak lagi memiliki bounding box setelah pemfilteran tetap dipertahankan sebagai background images untuk menekan tingkat False Positive).*

---

## ⚙️ Rangkuman Pipeline Pengolahan & Statistik Data

Seluruh pemrosesan citra dilakukan menggunakan skrip `Source Code/preprocess_and_augment.py` dengan tahapan:
- **Enhancement (CLAHE):** Diterapkan secara eksklusif pada kanal *Luminance* (`L`) dalam ruang warna `LAB` dengan parameter `clipLimit = 2.0` dan `tileGridSize = (8, 8)` untuk memperjelas tekstur retakan tanpa merusak informasi warna alami.
- **Standardisasi Dimensi (Resize):** Seluruh citra diseragamkan menjadi resolusi `640 × 640` piksel menggunakan interpolasi bilinear.
- **Augmentasi Data Offline:** Diterapkan hanya pada partisi `train` menggunakan teknik *Horizontal Flip* dan *Vertical Flip* beserta rekalkulasi koordinat *bounding box*. Partisi `val` dan `test` dipertahankan murni untuk mencegah kebocoran data (*data leakage*).

### Rekapitulasi Distribusi Citra & Label

| Split Dataset | Citra Awal | Citra Akhir | Label Terfilter (4 Kelas) | Label Akhir (Training-Ready) | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 26.869 | 62.673 | 41.668 | 142.504 | CLAHE + Resize + Offline Flip (H/V) |
| **Validation** | 5.758 | 5.758 | 9.366 | 9.366 | Hanya CLAHE + Resize (Murni) |
| **Test** | 5.758 | 5.758 | 9.324 | 9.324 | Hanya CLAHE + Resize (Murni) |
| **Total** | **38.385** | **74.189** | **60.358** | **161.194** | — |

---

## 📚 Sitasi Dataset Asli

Apabila menggunakan dataset ini, harap menyertakan sitasi sumber asli RDD2022:

> Arya, D., Maeda, H., Sekimoto, Y., Ghosh, S. K., & Toshniwal, D. (2022). *RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection*. figshare. https://doi.org/10.6084/m9.figshare.19425179
