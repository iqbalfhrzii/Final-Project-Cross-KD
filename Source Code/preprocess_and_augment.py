#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDD2022 Preprocessing (CLAHE + Resize) & Data Augmentation Pipeline
===================================================================
Skrip modular untuk:
1. Preprocessing:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) pada L-channel LAB
   - Resize citra menjadi 640x640
2. Data Augmentation (khusus split train):
   - Horizontal Flip (hflip)
   - Vertical Flip (vflip)
   - Rotasi 90, 180, 270 derajat
   - Transformasi koordinat bounding box YOLO (x, y, w, h) secara presisi
3. Menghasilkan data.yaml otomatis untuk pelatihan YOLOv8.

Sesuai metodologi proposal skripsi:
"Pengembangan Model Lightweight YOLOv8 untuk Deteksi Kerusakan Jalan"
"""

import argparse
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ==============================================================================
# FUNGSI TRANSFORMASI BOUNDING BOX YOLO
# ==============================================================================
def transform_yolo_bbox(
    class_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    aug_type: str,
) -> Tuple[int, float, float, float, float]:
    """
    Mentransformasi koordinat bounding box YOLO yang dinormalisasi [0.0, 1.0].
    Format YOLO: <class_id> <x_center> <y_center> <width> <height>
    """
    # Pastikan koordinat tetap dalam batas [0, 1]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    if aug_type == "none":
        return class_id, x, y, w, h

    elif aug_type == "hflip":
        # Mirror horizontal: x_center berubah, y, w, h tetap
        new_x = 1.0 - x
        return class_id, new_x, y, w, h

    elif aug_type == "vflip":
        # Mirror vertikal: y_center berubah, x, w, h tetap
        new_y = 1.0 - y
        return class_id, x, new_y, w, h

    elif aug_type == "rot90":
        # Rotasi 90 derajat searah jarum jam (Clockwise)
        # x_new = 1 - y, y_new = x, width dan height bertukar
        new_x = 1.0 - y
        new_y = x
        return class_id, new_x, new_y, h, w

    elif aug_type == "rot180":
        # Rotasi 180 derajat
        new_x = 1.0 - x
        new_y = 1.0 - y
        return class_id, new_x, new_y, w, h

    elif aug_type == "rot270":
        # Rotasi 270 derajat searah jarum jam (90 derajat CCW)
        new_x = y
        new_y = 1.0 - x
        return class_id, new_x, new_y, h, w

    else:
        raise ValueError(f"Tipe augmentasi tidak dikenal: {aug_type}")


# ==============================================================================
# PIPELINE PREPROCESSING & AUGMENTASI CITRA
# ==============================================================================
def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Penerapan CLAHE pada channel L dalam ruang warna LAB (sesuai preprocessing_demo.py).
    Meningkatkan kontras retakan aspal tanpa merusak saturasi warna.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def apply_image_transform(img: np.ndarray, aug_type: str) -> np.ndarray:
    """Menerapkan transformasi spasial pada citra."""
    if aug_type == "none":
        return img
    elif aug_type == "hflip":
        return cv2.flip(img, 1)
    elif aug_type == "vflip":
        return cv2.flip(img, 0)
    elif aug_type == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif aug_type == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    elif aug_type == "rot270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Transformasi tidak dikenal: {aug_type}")


# ==============================================================================
# PROCESSOR KELAS UTAMA
# ==============================================================================
class RDDDatasetAugmentor:
    """Processor untuk CLAHE, Resize, dan Augmentasi Dataset RDD2022."""

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        img_size: int = 640,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        augmentations: Optional[List[str]] = None,
        skip_empty_aug: bool = True,
        workers: int = 8,
        dry_run: bool = False,
    ):
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.img_size = img_size
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        # Default augmentasi: hflip, vflip, rot90 (menambah 3 variasi baru pada train)
        self.augmentations = augmentations or ["hflip", "vflip"]
        self.skip_empty_aug = skip_empty_aug
        self.workers = max(1, workers)
        self.dry_run = dry_run

        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger("RDDAugmentor")

    def read_labels(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """Membaca anotasi YOLO dari file label .txt."""
        boxes = []
        if not label_path.exists():
            return boxes

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        boxes.append((cid, x, y, w, h))
                    except ValueError:
                        continue
        return boxes

    def process_sample(
        self,
        img_path: Path,
        label_path: Path,
        out_img_dir: Path,
        out_lbl_dir: Path,
        aug_types: List[str],
    ) -> Tuple[int, int, Counter]:
        """
        Memproses 1 citra dasar dan membuat varian augmentasinya.
        Mengembalikan: (jumlah citra dibuat, jumlah label dibuat, counter kelas)
        """
        boxes = self.read_labels(label_path)
        is_empty = (len(boxes) == 0)

        # Muat citra asli
        img = cv2.imread(str(img_path))
        if img is None:
            self.logger.warning(f"Gagal membaca citra: {img_path}")
            return 0, 0, Counter()

        # 1. Terapkan CLAHE
        img_clahe = apply_clahe(img, self.clahe_clip, self.clahe_grid)

        # 2. Resize ke target size (640x640)
        img_resized = cv2.resize(img_clahe, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        images_created = 0
        labels_created = 0
        class_counts = Counter()

        # Tentukan variasi transformasi
        transforms_to_apply = ["none"]
        if not (is_empty and self.skip_empty_aug):
            transforms_to_apply.extend(aug_types)

        stem = img_path.stem
        ext = img_path.suffix

        for aug in transforms_to_apply:
            suffix = "" if aug == "none" else f"_{aug}"
            dest_img_path = out_img_dir / f"{stem}{suffix}{ext}"
            dest_lbl_path = out_lbl_dir / f"{stem}{suffix}.txt"

            # Transform citra
            if aug == "none":
                transformed_img = img_resized
            else:
                transformed_img = apply_image_transform(img_resized, aug)

            # Transform label
            transformed_boxes = []
            for cid, x, y, w, h in boxes:
                t_box = transform_yolo_bbox(cid, x, y, w, h, aug)
                transformed_boxes.append(t_box)
                class_counts[t_box[0]] += 1

            if not self.dry_run:
                # Simpan citra
                cv2.imwrite(str(dest_img_path), transformed_img)

                # Simpan label
                with open(dest_lbl_path, "w", encoding="utf-8") as f:
                    for cid, x, y, w, h in transformed_boxes:
                        f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            images_created += 1
            labels_created += 1

        return images_created, labels_created, class_counts

    def process_split(self, split: str) -> Tuple[int, int, Counter]:
        """Memproses seluruh data pada split tertentu."""
        src_img_dir = self.source_dir / "images" / split
        src_lbl_dir = self.source_dir / "labels" / split

        # Fallback jika struktur foldernya split/images
        if not src_img_dir.exists():
            src_img_dir = self.source_dir / split / "images"
            src_lbl_dir = self.source_dir / split / "labels"

        out_img_dir = self.target_dir / "images" / split
        out_lbl_dir = self.target_dir / "labels" / split

        if not self.dry_run:
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted([
            p for p in src_img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])

        # HANYA train yang diaugmentasi. val dan test HANYA di-preprocess (CLAHE + Resize)
        augs_for_split = self.augmentations if split == "train" else []

        self.logger.info(
            f"Memproses split '{split}': {len(img_files):,d} citra | "
            f"Augmentasi: {augs_for_split if augs_for_split else 'Tidak (Hanya Preprocessing CLAHE+Resize)'}"
        )

        total_images = 0
        total_labels = 0
        total_classes = Counter()

        progress_bar = None
        if tqdm is not None:
            progress_bar = tqdm(
                total=len(img_files),
                desc=f"Split {split}",
                unit="img",
                dynamic_ncols=True,
            )

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_file = {}
            for img_path in img_files:
                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
                future = executor.submit(
                    self.process_sample,
                    img_path,
                    lbl_path,
                    out_img_dir,
                    out_lbl_dir,
                    augs_for_split,
                )
                future_to_file[future] = img_path.name

            for future in as_completed(future_to_file):
                try:
                    n_img, n_lbl, c_counts = future.result()
                    total_images += n_img
                    total_labels += n_lbl
                    total_classes.update(c_counts)
                except Exception as e:
                    fname = future_to_file[future]
                    self.logger.error(f"Error memproses '{fname}': {e}")

                if progress_bar:
                    progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        return total_images, total_labels, total_classes

    def generate_data_yaml(self) -> Path:
        """Membuat data.yaml dengan path absolut dan forward slashes."""
        yaml_path = self.target_dir / "data.yaml"
        abs_target = str(self.target_dir.resolve()).replace("\\", "/")

        content = (
            f"# Konfigurasi Dataset RDD2022 (Preprocessed CLAHE + Augmented)\n"
            f"# Dibuat otomatis untuk YOLOv8 / CrossKD\n"
            f"path: {abs_target}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n\n"
            f"nc: 4\n"
            f"names: ['D00', 'D10', 'D20', 'D40']\n"
        )

        if not self.dry_run:
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"File data.yaml berhasil dibuat di: {yaml_path}")
        else:
            self.logger.info(f"[DRY-RUN] Simulasi pembuatan data.yaml di: {yaml_path}")

        return yaml_path

    def run(self):
        """Menjalankan seluruh alur pipeline."""
        start_time = time.time()
        self.logger.info("=" * 68)
        self.logger.info("PIPELINE PREPROCESSING (CLAHE + RESIZE) & DATA AUGMENTATION")
        self.logger.info("=" * 68)
        self.logger.info(f"Source Directory : {self.source_dir}")
        self.logger.info(f"Target Directory : {self.target_dir}")
        self.logger.info(f"CLAHE Params     : ClipLimit={self.clahe_clip}, Grid={self.clahe_grid}")
        self.logger.info(f"Target Size      : {self.img_size}x{self.img_size}")
        self.logger.info(f"Train Augmentasi : {self.augmentations}")
        self.logger.info(f"Skip Empty Aug   : {self.skip_empty_aug}")
        self.logger.info(f"Workers          : {self.workers}")
        self.logger.info(f"Dry Run          : {self.dry_run}")
        self.logger.info("=" * 68)

        stats_per_split = {}
        for split in ["train", "val", "test"]:
            n_img, n_lbl, c_counts = self.process_split(split)
            stats_per_split[split] = (n_img, n_lbl, c_counts)

        self.generate_data_yaml()

        elapsed = time.time() - start_time

        # Cetak Tabel Ringkasan
        sep = "=" * 70
        sub_sep = "-" * 70
        print("\n" + sep)
        print("          RINGKASAN PREPROCESSING & AUGMENTASI DATASET")
        print(sep)
        print(f" Waktu Komputasi : {elapsed:.2f} detik")
        print(sub_sep)
        print(f"{'Split':<12} | {'Citra Dihasilkan':<18} | {'Label File':<14}")
        print(sub_sep)

        grand_total_img = 0
        grand_total_lbl = 0
        grand_classes = Counter()

        for split, (n_img, n_lbl, c_counts) in stats_per_split.items():
            print(f"{split:<12} | {n_img:<18,d} | {n_lbl:<14,d}")
            grand_total_img += n_img
            grand_total_lbl += n_lbl
            grand_classes.update(c_counts)

        print(sub_sep)
        print(f"{'TOTAL':<12} | {grand_total_img:<18,d} | {grand_total_lbl:<14,d}")
        print(sep)

        class_names = {0: "D00 (Longitudinal)", 1: "D10 (Transverse)", 2: "D20 (Alligator)", 3: "D40 (Pothole)"}
        print(" TOTAL ANOTASI SETELAH AUGMENTASI:")
        print(sub_sep)
        for cid in sorted(class_names.keys()):
            print(f"  Class {cid} ({class_names[cid]:<18}) : {grand_classes[cid]:,d} bounding boxes")
        print(sep + "\n")


# ==============================================================================
# CLI
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing CLAHE + Resize dan Data Augmentation RDD2022",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="dataset_rdd2022_filtered",
        help="Path ke folder dataset input (hasil filter 4 kelas).",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="dataset_rdd2022_augmented",
        help="Path ke folder output hasil preprocessing & augmentasi.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Ukuran target resize citra (lebar & tinggi).",
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="Clip limit untuk CLAHE.",
    )
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=["hflip", "vflip"],
        choices=["hflip", "vflip", "rot90", "rot180", "rot270"],
        help="Daftar augmentasi yang diaplikasikan ke split train.",
    )
    parser.add_argument(
        "--include-empty-aug",
        action="store_true",
        help="Jika diset, citra background kosong juga ikut digandakan saat augmentasi.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 8,
        help="Jumlah worker threads paralel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulasi perhitungan tanpa menulis file ke disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    augmentor = RDDDatasetAugmentor(
        source_dir=Path(args.source_dir),
        target_dir=Path(args.target_dir),
        img_size=args.img_size,
        clahe_clip=args.clahe_clip,
        augmentations=args.augmentations,
        skip_empty_aug=not args.include_empty_aug,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    augmentor.run()


if __name__ == "__main__":
    main()
