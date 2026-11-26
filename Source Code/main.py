import os
import sys
from pathlib import Path

# Pastikan encoding terminal Windows menggunakan UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ultralytics import YOLO

# ==============================================================================
# KONFIGURASI TRAINING (RDD2022 4 CLASSES)
# ==============================================================================
# Jalur konfigurasi dataset RDD2022 (otomatis mendeteksi lokasi berkas)
yaml_candidates = [
    Path("../Dataset/data.yaml"),
    Path("Dataset/data.yaml"),
    Path("data.yaml"),
    Path("dataset_rdd2022_augmented/data.yaml"),
]
DATASET_YAML = str(next((p for p in yaml_candidates if p.exists()), Path("../Dataset/data.yaml")))

# Default stage jika tanpa argumen: langsung melatih Teacher Optimized sesuai arahan dosen!
# Pilihan: 'teacher_opt', 'auto_two_stage', 'teacher', 'student_kd', 'student_baseline', 'all'
STAGE = "teacher_opt"

# Hyperparameter Training (Disesuaikan 50 Epoch sesuai instruksi dosen)
EPOCHS = 50
PATIENCE = 50        # Set 50 agar menyelesaikan 50 epoch penuh (atau 20 jika ingin early stopping)
CLOSE_MOSAIC = 10    # Matikan augmentasi mosaic pada 10 epoch terakhir untuk konvergensi presisi
IMGSZ = 640
BATCH_SIZE = 16      # Set 16 (atau 32 jika di RTX 4090 24GB)
WORKERS = 8          # 8 worker optimal untuk Linux cloud instance (Vast.ai)
DEVICE = 0           # 0 untuk GPU CUDA, atau 'cpu'


def train_teacher():
    """Tahap 1A: Melatih Model Teacher Standar (YOLOv8m - Eksperimen 1)."""
    print("\n" + "=" * 60)
    print(">>> MEMULAI TRAINING TEACHER STANDAR (YOLOv8m) <<<")
    print(f"Data       : {DATASET_YAML}")
    print(f"Batch Size : {BATCH_SIZE}")
    print("AMP (FP16) : True")
    print("=" * 60 + "\n")

    model = YOLO("PretrainedModelFromOfficial/yolov8m.pt")
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        pretrained=True,
        iou=0.6,
        name="teacher_yolov8m_rdd2022",
        Mode="Training",
        quant=False,
        amp=True,
    )
    print("\n[✓] Training Teacher YOLOv8m Standar Selesai!")


def train_teacher_optimized():
    """Tahap 1B: Melatih Model Teacher Optimized (YOLOv8m - 50 Epoch).
    Fitur Anti-Overfitting & Optimasi:
    - epochs: 50 (Sesuai arahan dosen)
    - patience: 50 (Menjamin training berjalan 50 epoch penuh)
    - cos_lr: True (Cosine Annealing LR scheduler)
    - close_mosaic: 10 (Matikan augmentasi mosaic pada 10 epoch terakhir)
    - weight_decay: 0.001 (Regularisasi L2)
    """
    print("\n" + "=" * 60)
    print(f">>> MEMULAI TRAINING TEACHER OPTIMIZED (YOLOv8m - {EPOCHS} EPOCH) <<<")
    print(f"Data           : {DATASET_YAML}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Batch Size     : {BATCH_SIZE}")
    print(f"Patience       : {PATIENCE}")
    print("Cosine LR      : True")
    print(f"Close Mosaic   : {CLOSE_MOSAIC}")
    print("Weight Decay   : 0.001 (L2 Regularization)")
    print("AMP (FP16)     : True")
    print("=" * 60 + "\n")

    model = YOLO("PretrainedModelFromOfficial/yolov8m.pt")
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        pretrained=True,
        iou=0.6,
        patience=PATIENCE,
        cos_lr=True,
        close_mosaic=CLOSE_MOSAIC,
        weight_decay=0.001,
        name="teacher_yolov8m_optimized",
        Mode="Training",
        quant=False,
        amp=True,
    )
    print(f"\n[✓] Training Teacher YOLOv8m (Optimized {EPOCHS} Epoch) Selesai!")


def train_student_baseline():
    """Tahap 2: Baseline Student (YOLOv8n Standar Tanpa KD).
    Krusial sebagai pembanding dasar untuk membuktikan efektivitas CrossKD.
    """
    print("\n" + "=" * 60)
    print(">>> MEMULAI TRAINING STUDENT BASELINE (YOLOv8n Standar Tanpa KD) <<<")
    print(f"Data       : {DATASET_YAML}")
    print(f"Batch Size : {BATCH_SIZE}")
    print("AMP (FP16) : True")
    print("=" * 60 + "\n")

    model = YOLO("PretrainedModelFromOfficial/yolov8n.pt")
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        pretrained=True,
        iou=0.6,
        cos_lr=True,
        name="student_baseline_yolov8n_rdd2022",
        Mode="Training",
        quant=False,
        amp=True,
    )
    print("\n[✓] Training Student Baseline YOLOv8n Selesai!")


def train_student_crosskd(teacher_weights_path: str):
    """Tahap 3: Melatih Model Student (YOLOv8n) menggunakan CrossKD."""
    print("\n" + "=" * 60)
    print(">>> MEMULAI TRAINING STUDENT (YOLOv8n + CrossKD) <<<")
    print(f"Teacher Weights : {teacher_weights_path}")
    print(f"Data            : {DATASET_YAML}")
    print(f"Batch Size      : {BATCH_SIZE}")
    print("Cosine LR       : True")
    print("AMP (FP16)      : True")
    print("=" * 60 + "\n")

    if not Path(teacher_weights_path).exists():
        raise FileNotFoundError(
            f"File bobot teacher tidak ditemukan di: {teacher_weights_path}\n"
            "Pastikan Anda sudah melatih Teacher terlebih dahulu."
        )

    model = YOLO("PretrainedModelFromOfficial/yolov8n.pt")
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        pretrained=True,
        iou=0.6,
        cos_lr=True,
        name="student_crosskd_yolov8n_rdd2022",
        Mode="KD",
        KD_Method="CrossKD",
        teacher=teacher_weights_path,
        model_pt_path=teacher_weights_path,
        alpha=1.0,
        quant=False,
        amp=True,
    )
    print("\n[✓] Training Student CrossKD Selesai!")


def find_latest_teacher_weights():
    """Mencari file bobot best.pt Teacher terbaru secara dinamis dari runs/detect/."""
    candidates = []
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        # Cari semua best.pt di subfolder teacher (optimized, rdd2022, dll)
        for p in runs_dir.glob("teacher_*/weights/best.pt"):
            if p.is_file():
                candidates.append(p)
    
    if candidates:
        # Urutkan berdasarkan waktu modifikasi terbaru
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(candidates[0])
    return None


def train_two_stage():
    """Jalankan 2 Tahap Sekaligus secara Otomatis:
    1. Melatih Teacher YOLOv8m (Optimized Anti-Overfitting)
    2. Langsung melatih Student YOLOv8n (CrossKD) dibimbing bobot best.pt Teacher!
    """
    print("\n" + "=" * 70)
    print(">>> MEMULAI EKSEKUSI PIPELINE 2 TAHAP SEKALIGUS (TEACHER -> STUDENT) <<<")
    print("=" * 70 + "\n")

    # Langkah 1: Training Teacher
    train_teacher_optimized()

    # Cari bobot best.pt Teacher secara dinamis
    teacher_best_str = find_latest_teacher_weights()
    if teacher_best_str and Path(teacher_best_str).exists():
        teacher_best = Path(teacher_best_str)
    else:
        teacher_best = Path("runs/detect/teacher_yolov8m_optimized/weights/best.pt")

    if not teacher_best.exists():
        print(f"\n[!] Error: File bobot Teacher '{teacher_best}' tidak ditemukan!")
        print("    Training Student CrossKD tidak dapat dimulai otomatis.")
        return

    print("\n" + "=" * 70)
    print(f">>> TAHAP 1 (TEACHER) SELESAI! BOBOT TERBAIK: {teacher_best} <<<")
    print(">>> LANGSUNG MEMULAI TAHAP 2: STUDENT (YOLOv8n + CrossKD) <<<")
    print("=" * 70 + "\n")

    # Langkah 2: Training Student CrossKD
    train_student_crosskd(str(teacher_best))

    print("\n" + "=" * 70)
    print(">>> SELURUH TRAINING SELESAI 100%! <<<")
    print(f"1. Teacher Model : {teacher_best}")
    print("2. Student Model : runs/detect/student_crosskd_yolov8n_rdd2022/weights/best.pt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Pipeline YOLOv8 & CrossKD (RDD2022)")
    parser.add_argument(
        "--stage",
        type=str,
        default=STAGE,
        choices=["auto_two_stage", "teacher_opt", "teacher", "student_baseline", "student_kd", "all"],
        help="Mode eksekusi: auto_two_stage, teacher_opt, teacher, student_baseline, student_kd, all",
    )
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default=None,
        help="Path ke bobot best.pt Teacher (jika menjalankan student_kd secara mandiri)",
    )
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Ukuran batch")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Jumlah worker dataloader")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Jumlah epoch training (default: 50)")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience (default: 50)")
    parser.add_argument("--close-mosaic", type=int, default=CLOSE_MOSAIC, help="Jumlah epoch terakhir nonaktifkan mosaic (default: 10)")
    args = parser.parse_args()

    # Perbarui konfigurasi dari argumen CLI
    BATCH_SIZE = args.batch
    WORKERS = args.workers
    EPOCHS = args.epochs
    PATIENCE = args.patience
    CLOSE_MOSAIC = args.close_mosaic
    selected_stage = args.stage

    # Cek ketersediaan dataset
    if not Path(DATASET_YAML).exists():
        print(f"[!] Warning: File '{DATASET_YAML}' belum ditemukan.")
        print("    Jalankan 'process_rdd2022.py' atau 'preprocess_and_augment.py' terlebih dahulu.")

    if selected_stage == "auto_two_stage":
        train_two_stage()

    elif selected_stage == "teacher_opt":
        train_teacher_optimized()

    elif selected_stage == "teacher":
        train_teacher()

    elif selected_stage == "student_baseline":
        train_student_baseline()

    elif selected_stage == "student_kd":
        t_weights = args.teacher_weights
        if not t_weights:
            t_weights = find_latest_teacher_weights()
            if not t_weights:
                t_weights = "runs/detect/teacher_yolov8m_optimized/weights/best.pt"
        train_student_crosskd(t_weights)

    elif selected_stage == "all":
        train_two_stage()
        train_student_baseline()

