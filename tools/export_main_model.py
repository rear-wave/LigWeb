"""Export a ligClassify ``five_class_v1`` checkpoint for LigWeb inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from tools.export_ligclassify_model import _work_around_broken_windows_asyncio


CLASS_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def export_main_model(checkpoint, output, metadata, ligclassify_root):
    _work_around_broken_windows_asyncio()
    import torch

    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    metadata = Path(metadata).resolve()
    source_root = Path(ligclassify_root).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (source_root / "checkpoints.py").is_file():
        raise FileNotFoundError(source_root / "checkpoints.py")

    sys.path.insert(0, str(source_root))
    try:
        from checkpoints import FIVE_CLASS_SCHEMA, load_model_checkpoint

        loaded = load_model_checkpoint(checkpoint, device="cpu")
        if loaded.schema != FIVE_CLASS_SCHEMA:
            raise ValueError(f"expected {FIVE_CLASS_SCHEMA}, got {loaded.schema}")
        if tuple(loaded.type_names) != CLASS_NAMES:
            raise ValueError(f"unexpected class order: {loaded.type_names}")

        class MainModelExport(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, local, global_view, daylight):
                forward_type = getattr(self.model, "forward_type", None)
                if forward_type is not None:
                    return forward_type(local, global_view, daylight)
                result = self.model(local, global_view, daylight)
                return result.type_logits, result.features

        wrapper = MainModelExport(loaded.model.eval()).eval()
        sample_local = torch.linspace(
            -1.0, 1.0, 16000, dtype=torch.float32
        ).reshape(2, 1, 8000)
        sample_global = torch.linspace(
            -0.5, 0.5, 4000, dtype=torch.float32
        ).reshape(2, 1, 2000)
        sample_daylight = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.onnx.export(
            wrapper,
            (sample_local, sample_global, sample_daylight),
            temporary,
            input_names=["local", "global_view", "daylight"],
            output_names=["type_logits", "features"],
            dynamic_axes={
                "local": {0: "batch"},
                "global_view": {0: "batch"},
                "daylight": {0: "batch"},
                "type_logits": {0: "batch"},
                "features": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

        import onnxruntime as ort

        with torch.inference_mode():
            expected_logits, expected_features = wrapper(
                sample_local, sample_global, sample_daylight
            )
        session = ort.InferenceSession(
            str(temporary), providers=["CPUExecutionProvider"]
        )
        actual_logits, actual_features = session.run(
            None,
            {
                "local": sample_local.numpy(),
                "global_view": sample_global.numpy(),
                "daylight": sample_daylight.numpy(),
            },
        )
        np.testing.assert_allclose(
            actual_logits, expected_logits.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            actual_features, expected_features.numpy(), rtol=1e-4, atol=1e-5
        )
        os.replace(temporary, output)

        value = {
            "schema": "ligedit_main_model_v2",
            "class_names": list(CLASS_NAMES),
            "feature_dim": int(actual_features.shape[1]),
            "preprocess_schema": "five_class_dual_view_v1",
            "preprocess_config": dict(loaded.preprocess_config),
            "onnx_sha256": _sha256(output),
            "source_checkpoint_sha256": _sha256(checkpoint),
        }
        _atomic_json(metadata, value)
        return value
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--ligclassify-root", required=True)
    args = parser.parse_args(argv)
    value = export_main_model(
        args.checkpoint, args.output, args.metadata, args.ligclassify_root
    )
    print(
        f"classes={value['class_names']} feature_dim={value['feature_dim']} "
        f"sha256={value['onnx_sha256']} parity=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
