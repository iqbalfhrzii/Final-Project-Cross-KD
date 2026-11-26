#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter dan remapping anotasi RDD2022 (5 kelas -> 4 kelas: D00, D10, D20, D40)."""

import argparse
import logging
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEFAULT_CLASS_MAPPING: Dict[int, int] = {
    0: 0,  # D00: Longitudinal Crack
    1: 1,  # D10: Transverse Crack
    2: 2,  # D20: Alligator Crack
    3: 3,  # D40: Pothole
}

TARGET_CLASS_NAMES: Dict[int, str] = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
}

CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other Corruption / Repair (Filtered Out)",
}

SUPPORTED_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetStats:
    splits: List[str] = field(default_factory=lambda: ["train", "val", "test"])
    images_processed: Counter = field(default_factory=Counter)
    labels_processed: Counter = field(default_factory=Counter)
    empty_labels: Counter = field(default_factory=Counter)
    annotations_before: Counter = field(default_factory=Counter)
    annotations_after: Counter = field(default_factory=Counter)
    annotations_dropped: Counter = field(default_factory=Counter)

    def print_summary(self, elapsed_time: float, output_path: Optional[Path] = None):
        sep = "=" * 78
        sub_sep = "-" * 78
        print("\n" + sep)
        print("RINGKASAN PROSES FILTERING & REMAPPING DATASET RDD2022")
        print(sep)

        if output_path:
            print(f" Output Dataset: {output_path.resolve()}")
        print(f" Waktu Proses  : {elapsed_time:.2f} detik")
        print(sub_sep)

        print(f"{'Split':<12} | {'Citra':<12} | {'Label File':<12} | {'Label Kosong (Bg)':<18}")
        print(sub_sep)
        total_img = sum(self.images_processed.values())
        total_lbl = sum(self.labels_processed.values())
        total_empty = sum(self.empty_labels.values())

        for split in self.splits:
            img_c = self.images_processed[split]
            lbl_c = self.labels_processed[split]
            emp_c = self.empty_labels[split]
            print(f"{split:<12} | {img_c:<12,d} | {lbl_c:<12,d} | {emp_c:<18,d}")

        print(sub_sep)
        print(f"{'TOTAL':<12} | {total_img:<12,d} | {total_lbl:<12,d} | {total_empty:<18,d}")
        print(sep)

        print("DISTRIBUSI ANOTASI BOUNDING BOX (SEBELUM vs SESUDAH FILTER)")
        print(sep)
        print(f"{'ID':<4} | {'Kelas':<8} | {'Deskripsi':<32} | {'Sebelum':<10} | {'Sesudah':<10}")
        print(sub_sep)

        all_class_ids = sorted(set(list(self.annotations_before.keys()) + list(self.annotations_after.keys())))
        total_before = sum(self.annotations_before.values())
        total_after = sum(self.annotations_after.values())

        for cid in all_class_ids:
            name = TARGET_CLASS_NAMES.get(cid, "OTHER" if cid not in TARGET_CLASS_NAMES else "")
            desc = CLASS_DESCRIPTIONS.get(cid, f"Class ID {cid}")
            bef = self.annotations_before.get(cid, 0)
            aft = self.annotations_after.get(cid, 0)
            status_desc = desc if cid in TARGET_CLASS_NAMES else f"{desc} [Dihapus]"
            print(f"{cid:<4} | {name:<8} | {status_desc:<32} | {bef:<10,d} | {aft:<10,d}")

        print(sub_sep)
        dropped_count = total_before - total_after
        print(f"{'TOTAL ANOTASI':<48} | {total_before:<10,d} | {total_after:<10,d}")
        print(f"{'TOTAL ANOTASI DIHAPUS':<48} | {dropped_count:<10,d} | -")
        print(sep + "\n")


class RDD2022FilterProcessor:
    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        class_mapping: Optional[Dict[int, int]] = None,
        empty_mode: str = "keep",
        link_mode: str = "copy",
        num_workers: int = 4,
        dry_run: bool = False,
        splits: Optional[List[str]] = None,
    ):
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
        self.empty_mode = empty_mode
        self.link_mode = link_mode
        self.num_workers = max(1, num_workers)
        self.dry_run = dry_run
        self.splits = splits or ["train", "val", "test"]

        self.stats = DatasetStats(splits=self.splits)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger("RDD2022Processor")

    def find_split_directories(self, split: str) -> Tuple[Path, Path]:
        # Support layout {split}/images and images/{split}
        pat1_images = self.source_dir / split / "images"
        pat1_labels = self.source_dir / split / "labels"
        if pat1_images.exists() and pat1_labels.exists():
            return pat1_images, pat1_labels

        pat2_images = self.source_dir / "images" / split
        pat2_labels = self.source_dir / "labels" / split
        if pat2_images.exists() and pat2_labels.exists():
            return pat2_images, pat2_labels

        alt_images = self.source_dir / split
        alt_labels = self.source_dir / split
        if alt_images.exists():
            return alt_images, alt_labels

        raise FileNotFoundError(
            f"Folder images & labels untuk split '{split}' tidak ditemukan di '{self.source_dir}'."
        )

    def prepare_target_directories(self, split: str) -> Tuple[Path, Path, Optional[Path]]:
        target_images = self.target_dir / "images" / split
        target_labels = self.target_dir / "labels" / split
        target_empty = self.target_dir / "isolated_empty" / split if self.empty_mode == "isolate" else None

        if not self.dry_run:
            target_images.mkdir(parents=True, exist_ok=True)
            target_labels.mkdir(parents=True, exist_ok=True)
            if target_empty:
                (target_empty / "images").mkdir(parents=True, exist_ok=True)
                (target_empty / "labels").mkdir(parents=True, exist_ok=True)

        return target_images, target_labels, target_empty

    def filter_label_file(self, label_path: Path) -> Tuple[List[str], Counter, Counter]:
        filtered_lines = []
        counts_before = Counter()
        counts_after = Counter()

        if not label_path.exists():
            return filtered_lines, counts_before, counts_after

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    orig_cid = int(parts[0])
                except ValueError:
                    continue

                counts_before[orig_cid] += 1

                if orig_cid in self.class_mapping:
                    new_cid = self.class_mapping[orig_cid]
                    counts_after[new_cid] += 1
                    remap_line = f"{new_cid} " + " ".join(parts[1:])
                    filtered_lines.append(remap_line)

        except Exception as e:
            self.logger.warning(f"Gagal membaca label '{label_path}': {e}")

        return filtered_lines, counts_before, counts_after

    def _transfer_file(self, src: Path, dst: Path):
        if self.dry_run or dst.exists():
            return

        if self.link_mode == "hardlink":
            try:
                os.link(src, dst)
                return
            except OSError:
                pass

        elif self.link_mode == "symlink":
            try:
                os.symlink(src, dst)
                return
            except OSError:
                pass

        shutil.copy2(src, dst)

    def process_single_sample(
        self,
        img_path: Path,
        label_path: Path,
        target_images: Path,
        target_labels: Path,
        target_empty: Optional[Path],
    ) -> Tuple[bool, bool, Counter, Counter]:
        filtered_lines, counts_before, counts_after = self.filter_label_file(label_path)
        is_empty = len(filtered_lines) == 0

        if is_empty:
            if self.empty_mode == "remove":
                return False, True, counts_before, counts_after

            elif self.empty_mode == "isolate":
                if target_empty and not self.dry_run:
                    dest_img = target_empty / "images" / img_path.name
                    dest_lbl = target_empty / "labels" / label_path.name
                    self._transfer_file(img_path, dest_img)
                    open(dest_lbl, "w", encoding="utf-8").close()
                return False, True, counts_before, counts_after

            elif self.empty_mode == "keep":
                # Pertahankan file label kosong sebagai background image
                if not self.dry_run:
                    dest_img = target_images / img_path.name
                    dest_lbl = target_labels / label_path.name
                    self._transfer_file(img_path, dest_img)
                    open(dest_lbl, "w", encoding="utf-8").close()
                return True, True, counts_before, counts_after

        if not self.dry_run:
            dest_img = target_images / img_path.name
            dest_lbl = target_labels / label_path.name
            self._transfer_file(img_path, dest_img)
            with open(dest_lbl, "w", encoding="utf-8") as f:
                for line in filtered_lines:
                    f.write(f"{line}\n")

        return True, False, counts_before, counts_after

    def process_split(self, split: str):
        src_img_dir, src_lbl_dir = self.find_split_directories(split)
        tgt_img_dir, tgt_lbl_dir, tgt_empty_dir = self.prepare_target_directories(split)

        self.logger.info(f"Scanning split '{split}' dari: {src_img_dir}")

        img_files = [
            p for p in src_img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        img_files.sort(key=lambda p: p.name)

        total_files = len(img_files)
        self.logger.info(f"Ditemukan {total_files:,d} citra pada split '{split}'.")

        tasks = [(img_path, src_lbl_dir / f"{img_path.stem}.txt") for img_path in img_files]

        progress_bar = None
        if tqdm is not None:
            progress_bar = tqdm(
                total=total_files,
                desc=f"Memproses {split}",
                unit="file",
                dynamic_ncols=True,
            )

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.process_single_sample,
                    img_path,
                    label_path,
                    tgt_img_dir,
                    tgt_lbl_dir,
                    tgt_empty_dir,
                ): img_path.name
                for img_path, label_path in tasks
            }

            for future in as_completed(future_to_file):
                try:
                    is_copied, is_empty, c_before, c_after = future.result()

                    self.stats.labels_processed[split] += 1
                    if is_copied:
                        self.stats.images_processed[split] += 1
                    if is_empty:
                        self.stats.empty_labels[split] += 1

                    self.stats.annotations_before.update(c_before)
                    self.stats.annotations_after.update(c_after)

                except Exception as e:
                    file_name = future_to_file[future]
                    self.logger.error(f"Error memproses file '{file_name}': {e}")

                if progress_bar:
                    progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

    def generate_data_yaml(self) -> Path:
        yaml_path = self.target_dir / "data.yaml"
        dataset_abs_path = str(self.target_dir.resolve()).replace("\\", "/")

        yaml_content = (
            f"path: {dataset_abs_path}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n\n"
            f"nc: 4\n"
            f"names: ['D00', 'D10', 'D20', 'D40']\n"
        )

        if not self.dry_run:
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            self.logger.info(f"data.yaml dibuat di: {yaml_path}")
        else:
            self.logger.info(f"[DRY-RUN] data.yaml: {yaml_path}")

        return yaml_path

    def run(self):
        start_time = time.time()
        self.logger.info(f"Memulai pemrosesan dataset: {self.source_dir} -> {self.target_dir}")

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Direktori sumber '{self.source_dir}' tidak ditemukan.")

        if not self.dry_run:
            self.target_dir.mkdir(parents=True, exist_ok=True)

        for split in self.splits:
            self.process_split(split)

        self.generate_data_yaml()

        elapsed = time.time() - start_time
        self.stats.print_summary(elapsed, self.target_dir if not self.dry_run else None)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter & Remap RDD2022 Dataset untuk YOLOv8 (D00, D10, D20, D40)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--source-dir", type=str, default="RDD_SPLIT", help="Folder dataset sumber")
    parser.add_argument("--target-dir", type=str, default="dataset_rdd2022_filtered", help="Folder output hasil filter")
    parser.add_argument(
        "--empty-mode",
        type=str,
        choices=["keep", "remove", "isolate"],
        default="keep",
        help="Penanganan label kosong setelah filter ('keep', 'remove', 'isolate')",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="Metode transfer file ('copy', 'hardlink', 'symlink')",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Jumlah worker thread")
    parser.add_argument("--dry-run", action="store_true", help="Simulasi tanpa menulis file")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Daftar split")

    return parser.parse_args()


def main():
    args = parse_arguments()

    processor = RDD2022FilterProcessor(
        source_dir=Path(args.source_dir),
        target_dir=Path(args.target_dir),
        empty_mode=args.empty_mode,
        link_mode=args.link_mode,
        num_workers=args.workers,
        dry_run=args.dry_run,
        splits=args.splits,
    )

    try:
        processor.run()
    except KeyboardInterrupt:
        print("\nProses dibatalkan.")
        sys.exit(1)
    except Exception as err:
        logging.error(f"Terjadi kesalahan: {err}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
