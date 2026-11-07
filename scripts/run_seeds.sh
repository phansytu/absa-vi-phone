#!/usr/bin/env bash
# Huấn luyện mô hình qua 3 seed (42, 43, 44) để tính Mean ± Std theo Quy tắc 7.1.6

set -e

CONFIG=${1:-"configs/multitask.yaml"}
SEEDS=(42 43 44)

echo "=================================================="
echo "BẮT ĐẦU HUẤN LUYỆN 3 SEED CHO DỰ ÁN ABSA"
echo "Cấu hình: $CONFIG"
echo "Seeds: ${SEEDS[*]}"
echo "=================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Đang chạy Seed $SEED ..."
    python -m src.train --config "$CONFIG" --seed "$SEED"
done

echo ""
echo "=================================================="
echo "TỔNG HỢP KẾT QUẢ 3 SEED"
echo "=================================================="

python -c "
import json
import glob
import numpy as np

files = sorted(glob.glob('outputs/metrics/multitask_seed*.json'))
f1_scores = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        dev = data.get('dev', {})
        score = dev.get('joint_f1', 0.0)
        f1_scores.append(score)
        print(f'{f}: Joint F1 = {score:.4f}')

if f1_scores:
    print('-' * 45)
    print(f'Mean Joint F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')
"
