#!/bin/bash
set -e

echo "=========================================================="
echo ">>> MENYIAPKAN TRAINING TEACHER OPTIMIZED (50 EPOCH) <<<"
echo "=========================================================="

# 1. Sesuaikan path di data.yaml untuk Linux jika belum
if [ -f "dataset_rdd2022_augmented/data.yaml" ]; then
    sed -i 's|path:.*|path: dataset_rdd2022_augmented|g' dataset_rdd2022_augmented/data.yaml 2>/dev/null || true
fi

# 2. Jalankan training Teacher Optimized (50 Epoch) di background dengan nohup
echo "Memulai background training Teacher (nohup)..."
nohup python3 main.py --stage teacher_opt --epochs 50 --batch 16 --workers 8 > train_teacher.log 2>&1 &

echo "[✓] Training Teacher YOLOv8m (50 Epoch) sedang berjalan di background!"
echo "Pantau log realtime dengan perintah: tail -f train_teacher.log"
echo "=========================================================="
