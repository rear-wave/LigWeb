@echo off
setlocal
cd /d "%~dp0"
if not defined LIGWEB_TRAIN_DATA_DIR set "LIGWEB_TRAIN_DATA_DIR=%USERPROFILE%\Desktop\train_data"
if not defined LIGWEB_CORRECTION_DATA_DIR set "LIGWEB_CORRECTION_DATA_DIR=%USERPROFILE%\Desktop\correct_data"
if not defined LIGWEB_FEEDBACK_DIR set "LIGWEB_FEEDBACK_DIR=%LIGWEB_CORRECTION_DATA_DIR%\.ligedit"
if not defined LIGWEB_EXPORT_DIR set "LIGWEB_EXPORT_DIR=%LIGWEB_CORRECTION_DATA_DIR%\exports"
if not defined LIGWEB_HOST set "LIGWEB_HOST=0.0.0.0"
if not defined LIGWEB_PORT set "LIGWEB_PORT=8088"
echo LigWeb is starting at http://127.0.0.1:%LIGWEB_PORT%
echo LAN clients: http://SERVER-LAN-IP:%LIGWEB_PORT%
python -m uvicorn ligweb.app:app --host %LIGWEB_HOST% --port %LIGWEB_PORT% --workers 1
