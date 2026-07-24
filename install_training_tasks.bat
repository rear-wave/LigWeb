@echo off
setlocal
cd /d "%~dp0"
set "TASK_NAME=LigWebMainModelTraining"
set "TRAIN_COMMAND=%~dp0run_main_training.bat"
schtasks /Create /TN "%TASK_NAME%" /SC DAILY /ST 22:00 /TR "%TRAIN_COMMAND%" /F
if errorlevel 1 goto user_session
echo Installed scheduled task: %TASK_NAME%
echo Main model will retrain every day at 22:00 (Asia/Shanghai server time).
echo Correction-model training runs inside LigWeb; IC mirroring runs in this task.
exit /b 0
:user_session
echo Windows Task Scheduler is unavailable; installing the per-user startup scheduler.
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v LigEditMainModelScheduler /f >nul 2>nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v LigWebMainModelScheduler /t REG_SZ /d "%~dp0start_main_training_scheduler.bat" /f
if errorlevel 1 goto failed
call "%~dp0start_main_training_scheduler.bat"
echo Installed per-user scheduler. It will start automatically at Windows sign-in.
echo Main model will retrain every day at 22:00 (Asia/Shanghai server time).
exit /b 0
:failed
echo Failed to install both available automatic schedulers.
exit /b 1
