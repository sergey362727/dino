#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m pip install -r requirements.txt
python3 -m playwright install chromium

echo "Если браузер не запускается из-за системных библиотек, выполните:"
echo "  sudo python3 -m playwright install-deps chromium"
