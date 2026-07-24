# Repository Guidelines

## Project Structure

LigWeb is a Python 3.11 LAN web application for `.lig` waveform review and training. `ligweb/app.py` owns FastAPI routes, `ligweb/service.py` contains synchronous workflows, and `ligweb/static/` contains dependency-free browser assets. Keep ONNX inference in `ligweb/inference.py`, binary parsing/writing in `ligweb/lig_parser.py` and `ligweb/lig_io.py`, correction data logic in `ligweb/correction_dataset.py`, and IC reconciliation in `ligweb/ic_sync.py`. Scheduled main-model commands belong in `tools/`. Tests live in `tests/`.

## Data and Configuration

Never commit waveform datasets, runtime state, or model weights. Host defaults
are `Desktop\train_data` and `Desktop\correct_data`; the correction root contains
only the five label directories. Feedback, inbox, exports, backups, logs, and
model artifacts live under the ignored `LigWeb\runtime` directory. Use
`LIGWEB_*` variables from `.env.example` for overrides. Resolve all API paths
beneath configured roots.

## Development Commands

```powershell
pip install -r requirements-dev.txt
python -m compileall -q ligweb tools
python -m pytest -q
run_web.bat
run_web_docker.bat
```

Use one Uvicorn worker because model sessions and trainers are process-local.

## Style and Tests

Use four-space indentation and PEP 8 naming. Keep route handlers thin and business logic in `LigWebService`. Add tests for path confinement, malformed LIG input, deduplication, IC synchronization, model state changes, and scheduling. Run the full suite and Docker health check before deployment.

## Commits

Use concise Conventional Commit subjects such as `feat: sync reviewed IC data`. Describe visible behavior, host-data effects, validation, and screenshots for UI changes. Never commit datasets, SQLite files, generated models, exports, logs, secrets, caches, or IDE settings.
