"""Runtime configuration for LigWeb's LAN deployment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DATASET_NAMES = ("train", "inbox", "correction")
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env(name: str, default, legacy_name: str | None = None):
    value = os.environ.get(name)
    if value is None and legacy_name is not None:
        value = os.environ.get(legacy_name)
    return default if value is None else value


def _enabled(name: str, default: bool, legacy_name: str | None = None) -> bool:
    value = _env(name, "1" if default else "0", legacy_name)
    return str(value).strip().casefold() not in _FALSE_VALUES


def _default_desktop() -> Path:
    configured = _env("LIGWEB_DESKTOP", "", "LIGEDIT_DESKTOP")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / "Desktop"
    )


@dataclass(frozen=True)
class LigWebConfig:
    repository_root: Path
    train_data_dir: Path
    correction_data_dir: Path
    feedback_dir: Path
    exports_dir: Path
    host: str = "0.0.0.0"
    port: int = 8088
    max_cached_files: int = 4
    auto_correction_training: bool = True
    auto_ic_sync: bool = True
    ic_sync_poll_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "LigWebConfig":
        repository_root = Path(__file__).resolve().parents[1]
        desktop = _default_desktop()
        train_data = Path(
            _env(
                "LIGWEB_TRAIN_DATA_DIR",
                desktop / "train_data",
                "LIGEDIT_TRAIN_DATA_DIR",
            )
        ).expanduser()
        correction_data = Path(
            _env(
                "LIGWEB_CORRECTION_DATA_DIR",
                desktop / "correct_data",
                "LIGEDIT_CORRECTION_DATA_DIR",
            )
        ).expanduser()
        feedback_dir = Path(
            _env(
                "LIGWEB_FEEDBACK_DIR",
                correction_data / ".ligedit",
                "LIGEDIT_FEEDBACK_DIR",
            )
        ).expanduser()
        exports_dir = Path(
            _env(
                "LIGWEB_EXPORT_DIR",
                correction_data / "exports",
                "LIGEDIT_EXPORT_DIR",
            )
        ).expanduser()
        return cls(
            repository_root=repository_root,
            train_data_dir=train_data,
            correction_data_dir=correction_data,
            feedback_dir=feedback_dir,
            exports_dir=exports_dir,
            host=str(_env("LIGWEB_HOST", "0.0.0.0", "LIGEDIT_HOST")),
            port=int(_env("LIGWEB_PORT", "8088", "LIGEDIT_PORT")),
            max_cached_files=max(
                1,
                int(
                    _env(
                        "LIGWEB_MAX_CACHED_FILES",
                        "4",
                        "LIGEDIT_MAX_CACHED_FILES",
                    )
                ),
            ),
            auto_correction_training=_enabled(
                "LIGWEB_AUTO_CORRECTION_TRAINING",
                True,
                "LIGEDIT_AUTO_CORRECTION_TRAINING",
            ),
            auto_ic_sync=_enabled("LIGWEB_AUTO_IC_SYNC", True),
            ic_sync_poll_seconds=max(
                5.0, float(_env("LIGWEB_IC_SYNC_POLL_SECONDS", "60"))
            ),
        )

    @property
    def main_model_dir(self) -> Path:
        return self.feedback_dir / "main_model"

    @property
    def inbox_dir(self) -> Path:
        return self.feedback_dir / "inbox"

    @property
    def main_model_path(self) -> Path:
        return self.main_model_dir / "current.onnx"

    @property
    def main_model_metadata_path(self) -> Path:
        return self.main_model_dir / "current.json"

    @property
    def main_training_status_path(self) -> Path:
        return self.main_model_dir / "status.json"

    @property
    def ic_sync_status_path(self) -> Path:
        return self.feedback_dir / "ic-sync.json"

    def ensure_directories(self) -> None:
        if not self.train_data_dir.is_dir():
            raise FileNotFoundError(
                f"training data directory does not exist: {self.train_data_dir}"
            )
        self.correction_data_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.main_model_dir.mkdir(parents=True, exist_ok=True)

    def dataset_root(self, dataset: str) -> Path:
        if dataset == "train":
            return self.train_data_dir
        if dataset == "inbox":
            return self.inbox_dir
        if dataset == "correction":
            return self.correction_data_dir
        raise KeyError(f"unknown dataset: {dataset}")
