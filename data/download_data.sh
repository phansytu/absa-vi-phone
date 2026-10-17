#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip"
RAW="data/raw"

mkdir -p "$RAW"

if [ ! -f "$RAW/Train.csv" ]; then
    curl -fL --retry 3 -o "$RAW/UIT-ViSFD.zip" "$URL"
    unzip -o -q "$RAW/UIT-ViSFD.zip" -d "$RAW"
fi

(cd "$RAW" && sha256sum Train.csv Dev.csv Test.csv | tee SHA256SUMS.txt)

python -m src.data.dataset "$RAW" data/processed
echo "Xong. Kiểm tra data/raw/SHA256SUMS.txt và ghi nhật ký khảo sát"   