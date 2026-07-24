@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHASHSEED=0"
set "TRAIN_DATA=%USERPROFILE%\Desktop\train_data"
set "CORRECTION_DATA=%USERPROFILE%\Desktop\correct_data"
set "MODEL_DIR=%~dp0runtime"
set "LIGCLASSIFY_ROOT=%USERPROFILE%\Desktop\ligClassify"
set "LIGCLASSIFY_PYTHON=%USERPROFILE%\miniconda3\envs\ligclassify\python.exe"
set "TRAINING_LOG=%MODEL_DIR%\main_model\scheduled-training.log"
if not exist "%TRAIN_DATA%" goto missing_train
if not exist "%LIGCLASSIFY_ROOT%\train.py" goto missing_source
if not exist "%LIGCLASSIFY_PYTHON%" goto missing_environment
if not exist "%MODEL_DIR%\main_model" mkdir "%MODEL_DIR%\main_model"
echo [%date% %time%] nightly main-model training started>>"%TRAINING_LOG%"
echo Python: %LIGCLASSIFY_PYTHON%>>"%TRAINING_LOG%"
"%LIGCLASSIFY_PYTHON%" -m tools.train_main_model --train-data "%TRAIN_DATA%" --correction-data "%CORRECTION_DATA%" --runtime-dir "%MODEL_DIR%" --ligclassify-root "%LIGCLASSIFY_ROOT%" >>"%TRAINING_LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] nightly main-model training exited with %EXIT_CODE%>>"%TRAINING_LOG%"
exit /b %EXIT_CODE%
:missing_train
echo Training data directory not found: %TRAIN_DATA%
exit /b 1
:missing_source
echo ligClassify source not found: %LIGCLASSIFY_ROOT%
exit /b 1
:missing_environment
echo Conda environment not found: %LIGCLASSIFY_PYTHON%
exit /b 1
