@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHASHSEED=0"
set "LIGCLASSIFY_PYTHONW=%USERPROFILE%\miniconda3\envs\ligclassify\pythonw.exe"
if not exist "%LIGCLASSIFY_PYTHONW%" goto missing_environment
start "" /b "%LIGCLASSIFY_PYTHONW%" -m tools.main_training_scheduler
exit /b 0
:missing_environment
echo Conda environment not found: %LIGCLASSIFY_PYTHONW%
exit /b 1
