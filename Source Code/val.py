"""
Validation Script for Final Project CrossKD (RDD2022)
Peneliti: Iqbal Fahrozi (11221034) - Informatika ITK
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Validasi model deteksi kerusakan jalan RDD2022")
    parser.add_argument("--weights", type=str, default="PretrainedModelFromOfficial/yolov8n.pt", help="Path bobot model (.pt)")
    parser.add_argument("--data", type=str, default="../Dataset/data.yaml", help="Path berkas data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi citra inferensi")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch evaluasi")
    parser.add_argument("--device", default="0", help="CUDA device ID (0) atau 'cpu'")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split untuk validasi")
    args = parser.parse_args()

    # Periksa keberadaan berkas data.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        alt_paths = [Path("Dataset/data.yaml"), Path("../Dataset/data.yaml"), Path("data.yaml")]
        for alt in alt_paths:
            if alt.exists():
                data_path = alt
                break

    print(f"Memulai validasi model: {args.weights}")
    print(f"Dataset config         : {data_path}")
    print(f"Split                  : {args.split}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        iou=0.6,
        conf=0.25,
        split=args.split,
        device=args.device,
    )

    print("\n" + "=" * 50)
    print("HASIL VALIDASI:")
    print(f"Precision  : {metrics.box.p.mean():.4f}")
    print(f"Recall     : {metrics.box.r.mean():.4f}")
    print(f"mAP@50     : {metrics.box.map50:.4f}")
    print(f"mAP@50-95  : {metrics.box.map:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
