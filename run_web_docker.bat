@echo off
setlocal
cd /d "%~dp0"
set "TRAIN_DATA=%USERPROFILE%\Desktop\train_data"
set "CORRECTION_DATA=%USERPROFILE%\Desktop\correct_data"
set "MODEL_DATA=%~dp0runtime"
if not defined LIGWEB_PORT set "LIGWEB_PORT=8088"
if not exist "%TRAIN_DATA%" goto missing_train
if not exist "%CORRECTION_DATA%" mkdir "%CORRECTION_DATA%"
if not exist "%MODEL_DATA%" mkdir "%MODEL_DATA%"
echo Building LigWeb container...
docker build -t ligweb .
if errorlevel 1 exit /b 1
docker rm -f ligweb >nul 2>nul
docker run -d --name ligweb --restart unless-stopped -p %LIGWEB_PORT%:8000 -v "%TRAIN_DATA%:/data/train" -v "%CORRECTION_DATA%:/data/correction" -v "%MODEL_DATA%:/data/runtime" -e LIGWEB_FEEDBACK_DIR=/data/runtime -e LIGWEB_MODEL_DIR=/data/runtime -e LIGWEB_EXPORT_DIR=/data/runtime/exports -e LIGWEB_BASE_MODEL_PATH=/data/runtime/main_model/current.onnx -e LIGWEB_BASE_MODEL_METADATA_PATH=/data/runtime/main_model/current.json -e LIGWEB_CORRECTION_MODEL_DIR=/data/runtime -e LIGWEB_AUTO_CORRECTION_TRAINING=1 -e LIGWEB_AUTO_IC_SYNC=1 ligweb
if errorlevel 1 exit /b 1
call "%~dp0install_training_tasks.bat"
if errorlevel 1 echo Warning: web service is running, but the 22:00 main-model task was not installed.
echo LigWeb is running at http://127.0.0.1:%LIGWEB_PORT%
echo LAN clients: http://SERVER-LAN-IP:%LIGWEB_PORT%
echo Logs: docker logs -f ligweb
exit /b 0
:missing_train
echo Training data directory not found: %TRAIN_DATA%
exit /b 1
