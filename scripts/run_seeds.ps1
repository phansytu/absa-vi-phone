# PowerShell script chạy 3 seed (42, 43, 44) trên Windows

param (
    [string]$Config = "configs/multitask.yaml"
)

$ErrorActionPreference = "Stop"
$seeds = @(42, 43, 44)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "BẮT ĐẦU HUẤN LUYỆN 3 SEED CHO DỰ ÁN ABSA (POWERSHELL)" -ForegroundColor Cyan
Write-Host "Cấu hình: $Config" -ForegroundColor Cyan
Write-Host "Seeds: $($seeds -join ', ')" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

foreach ($seed in $seeds) {
    Write-Host "`n>>> Đang chạy Seed $seed ..." -ForegroundColor Yellow
    python -m src.train --config $Config --seed $seed
}

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "TỔNG HỢP KẾT QUẢ 3 SEED" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

python -c @"
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
"@
