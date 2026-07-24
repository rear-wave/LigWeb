"""Export the confirmed ligClassify legacy five-class model for LigWeb."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np


CLASS_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")


def _work_around_broken_windows_asyncio():
    """Allow build-only torch import when this machine's Winsock is unavailable."""
    try:
        import _overlapped  # noqa: F401
    except OSError:
        module = types.ModuleType("asyncio")
        module.__path__ = []
        module.iscoroutinefunction = lambda _value: False
        coroutines = types.ModuleType("asyncio.coroutines")
        coroutines._is_coroutine = object()
        coroutines.iscoroutinefunction = module.iscoroutinefunction
        module.coroutines = coroutines
        sys.modules.setdefault("asyncio", module)
        sys.modules.setdefault("asyncio.coroutines", coroutines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(checkpoint, output, metadata, ligclassify_root):
    _work_around_broken_windows_asyncio()
    import torch

    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    metadata = Path(metadata).resolve()
    ligclassify_root = Path(ligclassify_root).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (ligclassify_root / "checkpoints.py").is_file():
        raise FileNotFoundError(ligclassify_root / "checkpoints.py")

    sys.path.insert(0, str(ligclassify_root))
    try:
        from checkpoints import LEGACY_FIVE_CLASS_SCHEMA, load_model_checkpoint

        loaded = load_model_checkpoint(checkpoint, device="cpu")
        if loaded.schema != LEGACY_FIVE_CLASS_SCHEMA:
            raise ValueError(f"expected {LEGACY_FIVE_CLASS_SCHEMA}, got {loaded.schema}")
        if tuple(loaded.type_names) != CLASS_NAMES:
            raise ValueError(f"unexpected class order: {loaded.type_names}")

        class LegacyFeatureExport(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, waveform):
                features = self.model._encode(waveform)
                return self.model.type_head(features), features

        wrapper = LegacyFeatureExport(loaded.model.eval()).eval()
        sample = torch.linspace(-1.0, 1.0, 16000, dtype=torch.float32).reshape(2, 1, 8000)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            sample,
            output,
            input_names=["waveform"],
            output_names=["type_logits", "features"],
            dynamic_axes={
                "waveform": {0: "batch"},
                "type_logits": {0: "batch"},
                "features": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

        import onnxruntime as ort

        with torch.inference_mode():
            expected_logits, expected_features = wrapper(sample)
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual_logits, actual_features = session.run(
            None, {"waveform": sample.numpy()}
        )
        np.testing.assert_allclose(
            actual_logits, expected_logits.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            actual_features, expected_features.numpy(), rtol=1e-4, atol=1e-5
        )

        value = {
            "schema": "ligedit_base_model_v1",
            "class_names": list(CLASS_NAMES),
            "feature_dim": int(actual_features.shape[1]),
            "preprocess_schema": "legacy_minmax_8000_v1",
            "onnx_sha256": _sha256(output),
            "source_checkpoint_sha256": _sha256(checkpoint),
        }
        metadata.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return value
    finally:
        try:
            sys.path.remove(str(ligclassify_root))
        except ValueError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--ligclassify-root", required=True)
    args = parser.parse_args(argv)
    value = export_model(
        args.checkpoint, args.output, args.metadata, args.ligclassify_root
    )
    print(
        f"classes={value['class_names']} feature_dim={value['feature_dim']} "
        f"sha256={value['onnx_sha256']} parity=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
