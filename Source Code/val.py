import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validasi model deteksi kerusakan jalan RDD2022")
    parser.add_argument("--weights", type=str, default="PretrainedModelFromOfficial/yolov8n.pt", help="Path bobot model (.pt)")
    parser.add_argument("--data", type=str, default="../Dataset/data.yaml", help="Path berkas data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi citra")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch")
    parser.add_argument("--device", default="0", help="CUDA device ID atau 'cpu'")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split evaluasi")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        for alt in [Path("Dataset/data.yaml"), Path("../Dataset/data.yaml"), Path("data.yaml")]:
            if alt.exists():
                data_path = alt
                break

    print(f"Validasi model: {args.weights} | data: {data_path} | split: {args.split}")

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

    print(f"\nPrecision : {metrics.box.p.mean():.4f}")
    print(f"Recall    : {metrics.box.r.mean():.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
