import argparse
import os
import sys
from ultralytics import YOLO

MODEL_PATHS = {
    'teacher': 'runs/detect/teacher_yolov8m_optimized3/weights/best.pt',
    'baseline': 'runs/detect/student_baseline_yolov8n_rdd2022/weights/best.pt',
    'crosskd': 'runs/detect/student_crosskd_yolov8n_rdd20222/weights/best.pt',
}

CLASS_NAMES = {
    0: 'D00: Longitudinal Crack',
    1: 'D10: Transverse Crack',
    2: 'D20: Alligator Crack',
    3: 'D40: Pothole'
}


def run_prediction(model_key, img_path, conf=0.25, iou=0.6):
    weights_path = MODEL_PATHS[model_key]
    if not os.path.exists(weights_path):
        print(f"Error: File bobot {weights_path} tidak ditemukan.")
        return

    print(f"\nModel: {model_key.upper()} ({weights_path}) | Input: {img_path} | conf={conf}, iou={iou}")

    model = YOLO(weights_path)
    res = model.predict(
        source=img_path,
        conf=conf,
        iou=iou,
        save=True,
        project='runs/predict_test',
        name=model_key,
        exist_ok=True,
        verbose=False,
    )[0]

    boxes = res.boxes
    if len(boxes) == 0:
        print(f"Tidak ada objek terdeteksi (conf >= {conf}).")
    else:
        print(f"Terdeteksi {len(boxes)} objek:")
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            c_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            xyxy = [round(float(x), 1) for x in box.xyxy[0]]
            print(f"  {idx + 1}. {c_name} | conf: {score:.2f} | bbox: {xyxy}")

    saved_path = os.path.join('runs/predict_test', model_key, os.path.basename(img_path))
    print(f"Output disimpan di: {saved_path}")


def main():
    parser = argparse.ArgumentParser(description="Pengujian inferensi model deteksi kerusakan jalan YOLOv8")
    parser.add_argument('source', nargs='?', default='dataset_rdd2022_augmented/images/val/China_Drone_000142.jpg', help='Path ke citra input')
    parser.add_argument('--model', choices=['teacher', 'baseline', 'crosskd', 'all'], default='all', help='Model yang diuji')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='NMS IoU threshold')

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: File citra '{args.source}' tidak ditemukan.")
        sys.exit(1)

    if args.model == 'all':
        for m in ['teacher', 'baseline', 'crosskd']:
            run_prediction(m, args.source, conf=args.conf, iou=args.iou)
    else:
        run_prediction(args.model, args.source, conf=args.conf, iou=args.iou)


if __name__ == '__main__':
    main()
