#!/bin/bash
set -e

if [ -f "dataset_rdd2022_augmented/data.yaml" ]; then
    sed -i 's|path:.*|path: dataset_rdd2022_augmented|g' dataset_rdd2022_augmented/data.yaml 2>/dev/null || true
fi

nohup python3 main.py --stage teacher_opt --epochs 50 --batch 16 --workers 8 > train_teacher.log 2>&1 &
echo "Training Teacher berjalan di background. Log: tail -f train_teacher.log"
