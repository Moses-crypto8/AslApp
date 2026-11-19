@echo off
cd /d "%~dp0"
py -3.12 bootstrap.py
if errorlevel 1 (
  echo See README.md for Python 3.11 manual setup or troubleshooting.
  pause
  exit /b 1
)
