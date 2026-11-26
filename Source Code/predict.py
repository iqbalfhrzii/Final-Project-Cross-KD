from ultralytics import YOLO

# Ganti model path sesuai yang mau ditest
# Teacher
model = YOLO('teacher_yolov8s2/weights/teacher_best.pt')

# Student CrossKD
# model = YOLO('runs/detect/student_crosskd_yolov8n2/weights/best.pt')

model.predict(
    source='predict/test1.png',  # isi path foto/folder foto di sini
    iou=0.4,
    device=0,
    save=True,
    show_conf=True,
)