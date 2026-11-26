import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ultralytics import YOLO

yaml_candidates = [
    Path("../Dataset/data.yaml"),
    Path("Dataset/data.yaml"),
    Path("data.yaml"),
    Path("dataset_rdd2022_augmented/data.yaml"),
]
DATASET_YAML = str(next((p for p in yaml_candidates if p.exists()), Path("../Dataset/data.yaml")))

STAGE = "teacher_opt"
EPOCHS = 50
PATIENCE = 50
CLOSE_MOSAIC = 10
IMGSZ = 640
BATCH_SIZE = 16
WORKERS = 8
DEVICE = 0


def train_teacher():
    """Training Teacher standar (YOLOv8m)."""
    print(f"\n[Teacher Standar] Data: {DATASET_YAML}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}\n")

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
    print("\nTraining Teacher standar selesai.")


def train_teacher_optimized():
    """Training Teacher dengan cosine LR, close mosaic, dan L2 regularization."""
    print(f"\n[Teacher Optimized] Data: {DATASET_YAML}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}\n")

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
    print(f"\nTraining Teacher Optimized ({EPOCHS} epoch) selesai.")


def train_student_baseline():
    """Training Student baseline (YOLOv8n tanpa Knowledge Distillation)."""
    print(f"\n[Student Baseline] Data: {DATASET_YAML}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}\n")

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
    print("\nTraining Student Baseline selesai.")


def train_student_crosskd(teacher_weights_path: str):
    """Training Student (YOLOv8n) dengan Cross-Head Knowledge Distillation."""
    print(f"\n[Student CrossKD] Teacher: {teacher_weights_path}, Data: {DATASET_YAML}\n")

    if not Path(teacher_weights_path).exists():
        raise FileNotFoundError(f"Bobot teacher tidak ditemukan: {teacher_weights_path}")

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
    print("\nTraining Student CrossKD selesai.")


def find_latest_teacher_weights():
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob("teacher_*/weights/best.pt") if p.is_file()]
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(candidates[0])
    return None


def train_two_stage():
    """Menjalankan training Teacher Optimized dilanjutkan Student CrossKD."""
    train_teacher_optimized()

    teacher_best_str = find_latest_teacher_weights()
    teacher_best = (
        Path(teacher_best_str)
        if teacher_best_str and Path(teacher_best_str).exists()
        else Path("runs/detect/teacher_yolov8m_optimized/weights/best.pt")
    )

    if not teacher_best.exists():
        print(f"\nError: Bobot Teacher '{teacher_best}' tidak ditemukan.")
        return

    print(f"\nTeacher selesai ({teacher_best}). Lanjut training Student CrossKD...")
    train_student_crosskd(str(teacher_best))
    print("\nSeluruh pipeline two-stage selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline YOLOv8 & CrossKD (RDD2022)")
    parser.add_argument(
        "--stage",
        type=str,
        default=STAGE,
        choices=["auto_two_stage", "teacher_opt", "teacher", "student_baseline", "student_kd", "all"],
        help="Stage training yang dijalankan",
    )
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default=None,
        help="Path ke bobot best.pt Teacher untuk stage student_kd",
    )
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Ukuran batch")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Jumlah worker dataloader")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Jumlah epoch training")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--close-mosaic", type=int, default=CLOSE_MOSAIC, help="Epoch terakhir nonaktifkan mosaic")
    args = parser.parse_args()

    BATCH_SIZE = args.batch
    WORKERS = args.workers
    EPOCHS = args.epochs
    PATIENCE = args.patience
    CLOSE_MOSAIC = args.close_mosaic

    if not Path(DATASET_YAML).exists():
        print(f"Warning: File '{DATASET_YAML}' tidak ditemukan.")

    if args.stage == "auto_two_stage":
        train_two_stage()
    elif args.stage == "teacher_opt":
        train_teacher_optimized()
    elif args.stage == "teacher":
        train_teacher()
    elif args.stage == "student_baseline":
        train_student_baseline()
    elif args.stage == "student_kd":
        t_weights = args.teacher_weights or find_latest_teacher_weights()
        if not t_weights:
            t_weights = "runs/detect/teacher_yolov8m_optimized/weights/best.pt"
        train_student_crosskd(t_weights)
    elif args.stage == "all":
        train_two_stage()
        train_student_baseline()
