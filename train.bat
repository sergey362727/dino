@echo off
cd /d "%~dp0"
python dino.py train --steps 120000 --fresh
pause
