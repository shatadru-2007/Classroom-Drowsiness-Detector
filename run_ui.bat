@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
pip install flask>=3.0,<4 -q
python app.py
pause
