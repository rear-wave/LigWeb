"""FastAPI entry point for the LigWeb LAN application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ligweb.feedback_store import CLASS_NAMES
from ligweb.config import LigWebConfig
from ligweb.service import LigWebService


class FeedbackRequest(BaseModel):
    dataset: str
    path: str
    piece_index: int = Field(ge=0)
    corrected_label: str


class ExportRequest(BaseModel):
    dataset: str
    path: str
    keep_indices: list[int]
    output_name: str


class DaynightExportRequest(BaseModel):
    dataset: str
    path: str
    excluded_indices: list[int]
    output_name: str


class SaveLigRequest(BaseModel):
    deleted_indices: list[int]


class CorrectionImportRequest(BaseModel):
    dataset: str
    path: str
    piece_indices: list[int]


def create_app(config: LigWebConfig | None = None) -> FastAPI:
    config = config or LigWebConfig.from_env()
    service = LigWebService(config)

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        service.start_scheduler()
        try:
            yield
        finally:
            service.stop_scheduler()

    application = FastAPI(
        title="LigWeb API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.config = config
    application.state.service = service

    @application.exception_handler(KeyError)
    async def key_error_handler(_request: Request, error: KeyError):
        return _error_response(404, str(error))

    @application.exception_handler(FileNotFoundError)
    async def missing_file_handler(_request: Request, error: FileNotFoundError):
        return _error_response(404, str(error))

    @application.exception_handler(ValueError)
    @application.exception_handler(IndexError)
    @application.exception_handler(PermissionError)
    async def invalid_request_handler(_request: Request, error: Exception):
        return _error_response(400, str(error))

    @application.get("/api/health")
    def health():
        return {
            "status": "ok",
            "train_data_available": config.train_data_dir.is_dir(),
            "correction_data_available": config.correction_data_dir.is_dir(),
            "training": service.training_status(),
            "ic_sync": service.ic_sync_status(),
        }

    @application.get("/api/config")
    def public_config():
        return {
            "datasets": [
                {"id": "train", "label": "训练集", "writable": True},
                {"id": "inbox", "label": "待处理", "writable": True},
                {"id": "correction", "label": "纠错集", "writable": True},
            ],
            "classes": list(CLASS_NAMES),
            "max_upload_bytes": 512 * 1024 * 1024,
        }

    @application.get("/api/files")
    def files(
        dataset: str = Query("train"),
        query: str = Query(""),
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=500),
    ):
        return service.list_files(dataset, query, offset, limit)

    @application.get("/api/files/{dataset}/{file_path:path}/pieces")
    def pieces(dataset: str, file_path: str, classify: bool = True):
        return service.list_pieces(dataset, file_path, classify=classify)

    @application.get("/api/files/{dataset}/{file_path:path}/piece/{piece_index}")
    def piece(
        dataset: str,
        file_path: str,
        piece_index: int,
        max_points: int = Query(4000, ge=200, le=12000),
    ):
        return service.get_piece(dataset, file_path, piece_index, max_points)

    @application.delete("/api/files/{dataset}/{file_path:path}/session")
    def close_file(dataset: str, file_path: str):
        return {"closed": service.close_document(dataset, file_path)}

    @application.post("/api/files/{dataset}/{file_path:path}/save")
    def save_file(dataset: str, file_path: str, request: SaveLigRequest):
        return service.save_document(
            dataset,
            file_path,
            request.deleted_indices,
        )

    @application.post("/api/feedback")
    def save_feedback(request: FeedbackRequest):
        return service.save_feedback(
            request.dataset,
            request.path,
            request.piece_index,
            request.corrected_label,
        )

    @application.get("/api/feedback")
    def feedback():
        return {"records": service.list_feedback()}

    @application.delete("/api/feedback/{waveform_hash}")
    def cancel_feedback(waveform_hash: str):
        if not service.cancel_feedback(waveform_hash):
            raise HTTPException(status_code=404, detail="feedback record not found")
        return {"cancelled": True}

    @application.get("/api/training")
    def training_status():
        return service.training_status()

    @application.get("/api/automation")
    def automation_status():
        return service.automation_status()

    @application.post("/api/ic-sync")
    def sync_ic_data():
        return service.sync_ic_data(force=True)

    @application.post("/api/training")
    def train():
        return service.start_training()

    @application.post("/api/correction-imports")
    def import_corrections(request: CorrectionImportRequest):
        return service.import_corrected_pieces(
            request.dataset,
            request.path,
            request.piece_indices,
        )

    @application.post("/api/export")
    def export(request: ExportRequest):
        result = service.export_pieces(
            request.dataset,
            request.path,
            request.keep_indices,
            request.output_name,
        )
        result["download_url"] = f"/api/exports/{result['name']}"
        return result

    @application.post("/api/export/daynight")
    def export_daynight(request: DaynightExportRequest):
        result = service.export_by_daynight(
            request.dataset,
            request.path,
            request.excluded_indices,
            request.output_name,
        )
        result["download_url"] = f"/api/exports/{result['name']}"
        return result

    @application.get("/api/exports/{output_name}")
    def download_export(output_name: str):
        path = service.resolve_export(output_name)
        return FileResponse(path, filename=path.name)

    @application.post("/api/uploads")
    async def upload(
        request: Request,
        filename: str = Query(..., min_length=1, max_length=180),
    ):
        limit = 512 * 1024 * 1024
        content_length = int(request.headers.get("content-length") or 0)
        if content_length > limit:
            raise HTTPException(status_code=413, detail="file is too large")
        payload = await request.body()
        if len(payload) > limit:
            raise HTTPException(status_code=413, detail="file is too large")
        return service.upload_lig(filename, payload)

    static_dir = Path(__file__).with_name("static")
    application.mount(
        "/", StaticFiles(directory=static_dir, html=True), name="frontend"
    )
    return application


def _error_response(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})


app = create_app()
