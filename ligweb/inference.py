#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LigWeb ONNX waveform inference and correction-model integration.

整合 ligClassify 的预处理管线，为 LigWeb 提供：
  1. 单片段分类推理 (classify_single)
  2. 批量文件夹分类 (classify_folder)

使用 ONNX Runtime 推理，无需安装 PyTorch，包体积 ~30MB。
"""

import os
import sys
import csv
from dataclasses import dataclass
import hashlib
import json
import logging
from threading import RLock

import numpy as np

logger = logging.getLogger(__name__)
_dll_directory_handles = []

# ============================================================================
#                          预处理函数 (从 ligClassify 迁移)
# ============================================================================

def butterworth_filter(piece, fc=120000, fs=5000000, order=2):
    """Butterworth 低通滤波 (sos 形式，数值稳定)"""
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, fc / (fs / 2), btype='low', output='sos')
        return sosfiltfilt(sos, piece).astype(np.float32)
    except ImportError:
        return piece.astype(np.float32)


def cut_around_peak(piece, before=2000, after=6000, target_length=8000):
    """以最大值为中心裁剪波形"""
    index_max = int(np.argmax(piece))
    begin = index_max - before
    end = index_max + after
    if begin < 0:
        begin = 0
        end = target_length
    elif end > len(piece):
        end = len(piece)
        begin = end - target_length
    cut = piece[begin:end]
    if len(cut) < target_length:
        cut = np.pad(cut, (0, target_length - len(cut)), 'constant')
    elif len(cut) > target_length:
        cut = cut[:target_length]
    return cut


def normalize_minmax(piece):
    """MinMax 归一化: (x - mean) / (max - min)"""
    pmin, pmax = piece.min(), piece.max()
    if pmax - pmin < 1e-8:
        return (piece - piece.mean()).astype(np.float32)
    return ((piece - piece.mean()) / (pmax - pmin)).astype(np.float32)


def preprocess_waveform(piece, use_filter=True, cut_peak=True,
                        target_length=8000, normalize_mode='minmax'):
    """完整预处理流水线: 滤波 → 峰值裁剪 → 归一化"""
    piece = piece.astype(np.float32)
    if use_filter:
        piece = butterworth_filter(piece)
    if cut_peak:
        piece = cut_around_peak(piece, target_length=target_length)
    if normalize_mode == 'minmax':
        piece = normalize_minmax(piece)
    elif normalize_mode == 'zscore':
        std = piece.std()
        if std > 1e-8:
            piece = ((piece - piece.mean()) / std).astype(np.float32)
    return piece


# ============================================================================
#                          Batch 预处理 (批量加速)
# ============================================================================

def butterworth_filter_batch(pieces, fc=120000, fs=5000000, order=2):
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, fc / (fs / 2), btype='low', output='sos')
        return sosfiltfilt(sos, pieces, axis=-1).astype(np.float32)
    except ImportError:
        return pieces.astype(np.float32)


def cut_around_peak_batch(pieces, before=2000, after=6000, target_length=8000):
    N, T = pieces.shape
    peak_indices = np.argmax(pieces, axis=1).astype(np.int64)
    begins = peak_indices - before
    ends = peak_indices + after
    clamp_begin = (begins < 0)
    begins[clamp_begin] = 0
    ends[clamp_begin] = target_length
    clamp_end = (ends > T)
    ends[clamp_end] = T
    begins[clamp_end] = ends[clamp_end] - target_length
    result = np.empty((N, target_length), dtype=np.float32)
    for i in range(N):
        seg = pieces[i, begins[i]:ends[i]]
        L = len(seg)
        if L < target_length:
            result[i, :L] = seg
            result[i, L:] = 0.0
        else:
            result[i] = seg[:target_length]
    return result


def normalize_minmax_batch(pieces):
    pmin = pieces.min(axis=1, keepdims=True)
    pmax = pieces.max(axis=1, keepdims=True)
    pmean = pieces.mean(axis=1, keepdims=True)
    denom = pmax - pmin
    denom[denom < 1e-8] = 1.0
    return ((pieces - pmean) / denom).astype(np.float32)


def preprocess_batch(pieces, use_filter=True, cut_peak=True,
                     target_length=8000, normalize_mode='minmax'):
    if pieces.ndim == 1:
        pieces = pieces.reshape(1, -1)
    if use_filter:
        pieces = butterworth_filter_batch(pieces)
    if cut_peak:
        pieces = cut_around_peak_batch(pieces, target_length=target_length)
    if normalize_mode == 'minmax':
        pieces = normalize_minmax_batch(pieces)
    return pieces


def _robust_normalize_batch(values):
    centered = values - np.median(values, axis=1, keepdims=True)
    scale = np.quantile(np.abs(centered), 0.95, axis=1, keepdims=True)
    return (centered / np.maximum(scale, 1e-6)).astype(np.float32)


def preprocess_dual_view_batch(pieces, local_length=8000, global_length=2000,
                               use_filter=True, cutoff_hz=120000.0,
                               sample_rate_hz=5000000.0):
    """Match ligClassify's signed local/global five-class preprocessing."""
    values = np.asarray(pieces, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    if use_filter:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(
            2,
            float(cutoff_hz) / (float(sample_rate_hz) / 2.0),
            btype='low',
            output='sos',
        )
        values = sosfiltfilt(sos, values, axis=-1).astype(np.float32)

    rows, source_length = values.shape
    local = np.zeros((rows, int(local_length)), dtype=np.float32)
    centered = values - np.median(values, axis=1, keepdims=True)
    peaks = np.argmax(np.abs(centered), axis=1)
    before = int(local_length) // 4
    for row_index, peak in enumerate(peaks):
        if source_length <= int(local_length):
            local[row_index, :source_length] = values[row_index]
        else:
            start = min(
                max(int(peak) - before, 0), source_length - int(local_length)
            )
            local[row_index] = values[
                row_index, start:start + int(local_length)
            ]

    if source_length == int(global_length):
        global_view = values.copy()
    elif (
        source_length > int(global_length)
        and source_length % int(global_length) == 0
    ):
        width = source_length // int(global_length)
        global_view = values.reshape(rows, int(global_length), width).mean(axis=2)
    else:
        source_axis = np.linspace(0.0, 1.0, source_length)
        target_axis = np.linspace(0.0, 1.0, int(global_length))
        global_view = np.stack([
            np.interp(target_axis, source_axis, row) for row in values
        ]).astype(np.float32)
    return _robust_normalize_batch(local), _robust_normalize_batch(global_view)


# ============================================================================
#                          模型加载 (ONNX Runtime 懒加载单例)
# ============================================================================

_session = None
_class_names = None
_model_hash = None
_model_signature = None
_model_path = None
_model_schema = None
_model_metadata = None
_model_lock = RLock()

# 默认类别名 (硬编码，与 checkpoint 中一致)
CLASS_NAMES = ('IC', 'NCG', 'NNBE', 'PCG', 'PNBE')
_DEFAULT_CLASS_NAMES = list(CLASS_NAMES)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _get_resource_path(relative_path):
    """获取资源文件路径，兼容 PyInstaller 打包"""
    import sys
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_path, relative_path))


def _configured_model_paths(checkpoint_path=None):
    if checkpoint_path is not None:
        path = os.path.abspath(os.fspath(checkpoint_path))
        metadata = os.environ.get(
            "LIGWEB_BASE_MODEL_METADATA_PATH"
        ) or os.environ.get("LIGEDIT_BASE_MODEL_METADATA_PATH")
        if not metadata:
            metadata = os.path.splitext(path)[0] + ".json"
        return path, os.path.abspath(metadata)

    configured = os.environ.get("LIGWEB_BASE_MODEL_PATH") or os.environ.get(
        "LIGEDIT_BASE_MODEL_PATH"
    )
    if configured and os.path.isfile(configured):
        metadata = os.environ.get(
            "LIGWEB_BASE_MODEL_METADATA_PATH"
        ) or os.environ.get("LIGEDIT_BASE_MODEL_METADATA_PATH")
        if not metadata:
            metadata = os.path.splitext(configured)[0] + ".json"
        return os.path.abspath(configured), os.path.abspath(metadata)
    return (
        _get_resource_path("checkpoints/ligclassify_legacy.onnx"),
        _get_resource_path("checkpoints/ligclassify_legacy.json"),
    )


def _checkpoint_signature(path):
    stat = os.stat(path)
    return os.path.abspath(path), stat.st_mtime_ns, stat.st_size


def load_model(checkpoint_path=None, force=False):
    """
    懒加载分类模型 (ONNX Runtime 单例)。

    Args:
        checkpoint_path: ONNX 模型文件路径，默认使用五分类特征模型

    Returns:
        (session, class_names, 'cpu')
    """
    global _session, _class_names, _model_hash
    global _model_signature, _model_path, _model_schema, _model_metadata

    try:
        # PyInstaller 打包后，.pyd 加载时需要能找到同级目录下的 DLL
        if getattr(sys, 'frozen', False):
            ort_capi_dir = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
            if os.path.isdir(ort_capi_dir):
                handle = os.add_dll_directory(ort_capi_dir)
                _dll_directory_handles.append(handle)
        import onnxruntime as ort
    except ImportError as e:
        preload_error = getattr(sys, "_ligweb_onnxruntime_preload_error", None)
        detail = preload_error or str(e)
        raise ImportError(f"ONNX Runtime 加载失败（详情: {detail}）") from e

    checkpoint_path, metadata_path = _configured_model_paths(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ONNX 模型不存在: {checkpoint_path}")
    signature = _checkpoint_signature(checkpoint_path)
    with _model_lock:
        if not force and _session is not None and signature == _model_signature:
            return _session, _class_names, 'cpu'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"ONNX 模型元数据不存在: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        model_hash = _sha256_file(checkpoint_path)
        schema = metadata.get('schema')
        supported = {
            'ligedit_base_model_v1': 'legacy_minmax_8000_v1',
            'ligedit_main_model_v2': 'five_class_dual_view_v1',
        }
        if schema not in supported:
            raise ValueError("不支持的基础模型元数据")
        if tuple(metadata.get('class_names', ())) != CLASS_NAMES:
            raise ValueError("基础模型类别顺序不正确")
        if metadata.get('preprocess_schema') != supported[schema]:
            raise ValueError("基础模型预处理版本不正确")
        if metadata.get('onnx_sha256') != model_hash:
            raise ValueError("基础模型校验失败")

        logger.info(f"加载 ONNX 模型: {checkpoint_path}")
        candidate = ort.InferenceSession(
            checkpoint_path, providers=['CPUExecutionProvider']
        )
        expected_inputs = (
            {'waveform'} if schema == 'ligedit_base_model_v1'
            else {'local', 'global_view', 'daylight'}
        )
        actual_inputs = {item.name for item in candidate.get_inputs()}
        if actual_inputs != expected_inputs:
            raise ValueError(f"ONNX模型输入不匹配: {sorted(actual_inputs)}")
        _session = candidate
        _class_names = _DEFAULT_CLASS_NAMES
        _model_hash = model_hash
        _model_signature = signature
        _model_path = checkpoint_path
        _model_schema = schema
        _model_metadata = metadata

        logger.info(f"ONNX 模型加载完成: {len(_class_names)} 类 {_class_names}")
        return _session, _class_names, 'cpu'


def _ensure_model(checkpoint_path=None):
    """确保模型已加载"""
    # A path-less session is an explicitly injected in-memory session (used by
    # tests and embedders); it cannot be checked for filesystem replacement.
    if _session is not None and _model_path is None:
        return _session, _class_names, 'cpu'
    return load_model(checkpoint_path)


def is_model_loaded():
    """检查模型是否已加载"""
    return _session is not None


def get_base_model_hash():
    """Return the SHA-256 identity, noticing atomically replaced models."""
    global _model_hash, _model_signature
    path, _metadata = _configured_model_paths()
    if not os.path.exists(path):
        return ''
    signature = _checkpoint_signature(path)
    if _model_hash is None or signature != _model_signature:
        _model_hash = _sha256_file(path)
        if _session is None:
            _model_signature = signature
    return _model_hash or ''


# ============================================================================
#                          Softmax 工具 (纯 numpy)
# ============================================================================

def _softmax(logits):
    """稳定 softmax: exp(x-max) / sum(exp(x-max))"""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class BasePrediction:
    label: str
    confidence: float
    probabilities: tuple
    feature: np.ndarray


@dataclass(frozen=True)
class PredictionResult:
    base_label: str
    base_confidence: float
    probabilities: tuple
    feature: np.ndarray
    effective_label: str
    source: str
    correction_similarity: float | None = None


@dataclass(frozen=True)
class CorrectionContext:
    records: dict
    index: object | None
    base_model_hash: str

    @property
    def generation(self):
        return None if self.index is None else self.index.generation


def apply_correction(base, exact_label=None, suppressed=False, index=None,
                     base_model_hash=None):
    """Combine a frozen base prediction with exact or guarded adapter feedback."""
    from ligweb.correction_model import resolve_correction

    current_hash = base_model_hash if base_model_hash is not None else get_base_model_hash()
    if index is not None and index.base_model_hash != current_hash:
        index = None
    decision = resolve_correction(
        base.label, base.feature, exact_label, suppressed, index
    )
    return PredictionResult(
        base_label=base.label,
        base_confidence=base.confidence,
        probabilities=base.probabilities,
        feature=base.feature,
        effective_label=decision.label,
        source=decision.source,
        correction_similarity=decision.similarity,
    )


def load_correction_context(feedback_dir=None, correction_model_dir=None):
    """Load exact feedback and the active guarded adapter once per job."""
    from pathlib import Path

    from ligweb.correction_model import load_active_index
    from ligweb.feedback_store import FeedbackStore, default_feedback_dir

    root = Path(feedback_dir) if feedback_dir is not None else default_feedback_dir()
    model_root = (
        Path(correction_model_dir)
        if correction_model_dir is not None
        else Path(os.environ.get("LIGWEB_CORRECTION_MODEL_DIR", root))
    )
    store = FeedbackStore(root / "feedback.sqlite3")
    records, _failures = store.list_records_with_failures()
    model_hash = get_base_model_hash()
    return CorrectionContext(
        records={record.waveform_hash: record for record in records},
        index=load_active_index(model_root, model_hash),
        base_model_hash=model_hash,
    )


def apply_feedback_batch(waveforms, base_predictions, context=None,
                         feedback_dir=None, correction_model_dir=None):
    """Apply exact feedback and the active correction model to base outputs."""
    from ligweb.feedback_store import waveform_digest

    if len(waveforms) != len(base_predictions):
        raise ValueError("waveforms and base_predictions must have equal lengths")
    if context is None:
        context = load_correction_context(feedback_dir, correction_model_dir)

    results = []
    for waveform, base in zip(waveforms, base_predictions):
        record = context.records.get(waveform_digest(waveform))
        results.append(apply_correction(
            base,
            exact_label=(
                record.corrected_label if record is not None and record.enabled else None
            ),
            suppressed=record is not None and not record.enabled,
            index=context.index,
            base_model_hash=context.base_model_hash,
        ))
    return results


# ============================================================================
#                          单片段分类
# ============================================================================

def classify_single(waveform, checkpoint_path=None):
    """
    对单个波形片段进行分类。

    Args:
        waveform: (T,) float64/float32 原始波形数据
        checkpoint_path: ONNX 模型路径（首次调用时使用）

    Returns:
        (class_name, confidence): 预测类别名和置信度 (0~1)
        例如: ("NCG", 0.8723)
    """
    result = classify_batch_with_feedback(
        [waveform], checkpoint_path=checkpoint_path, batch_size=1
    )[0]
    return result.effective_label, result.base_confidence


def classify_single_with_probs(waveform, checkpoint_path=None):
    """
    对单个波形片段进行分类，返回完整概率分布。

    Returns:
        (class_name, confidence, {class: prob})
    """
    result = classify_batch_with_feedback(
        [waveform], checkpoint_path=checkpoint_path, batch_size=1
    )[0]
    prob_dict = {
        CLASS_NAMES[index]: float(value)
        for index, value in enumerate(result.probabilities)
    }
    return result.effective_label, result.base_confidence, prob_dict


# ============================================================================
#                          批量文件夹分类
# ============================================================================

def classify_folder(input_dir, output_dir=None, checkpoint_path=None,
                    batch_size=256, max_pieces=None,
                    progress_cb=None, log_cb=None):
    """
    对文件夹中所有 .lig 文件的波形片段进行分类。

    Args:
        input_dir:      输入目录（递归搜索 .lig 文件）
        output_dir:     输出目录（默认: input_dir/classified/），保存 summary.csv
        checkpoint_path:ONNX 模型路径
        batch_size:     推理批次大小
        max_pieces:     最大处理片段数
        progress_cb:    进度回调 (step, message, percent)
        log_cb:         日志回调 (message)

    Returns:
        dict: 分类汇总
    """
    from ligweb.lig_parser import ReadLigFile

    _session_value, class_names, _ = _ensure_model(checkpoint_path)
    correction_context = load_correction_context()

    if output_dir is None:
        output_dir = os.path.join(input_dir, "classified")
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 .lig 文件
    lig_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.lig'):
                lig_files.append(os.path.join(root, f))

    if not lig_files:
        if log_cb:
            log_cb("[终止] 未找到 .lig 文件")
        return {}

    logger.info(f"找到 {len(lig_files)} 个 .lig 文件")

    # 收集所有波形片段
    if progress_cb:
        progress_cb(0, "读取 .lig 文件...", 0)

    all_waveforms = []
    all_meta = []

    for file_idx, lig_file in enumerate(lig_files):
        if progress_cb:
            progress_cb(0, f"读取文件 {file_idx+1}/{len(lig_files)}",
                        int((file_idx + 1) / max(len(lig_files), 1) * 30))
        try:
            lig_data = ReadLigFile(lig_file)
            for piece_index, (time_key, piece_data) in enumerate(lig_data.items()):
                if '0' not in piece_data:
                    continue
                wf = np.array(piece_data['0'], dtype=np.float64)
                all_waveforms.append(wf)
                all_meta.append((os.path.basename(lig_file), piece_index, time_key))
                if max_pieces and len(all_waveforms) >= max_pieces:
                    break
        except Exception as e:
            if log_cb:
                log_cb(f"[错误] 读取 {lig_file}: {e}")
        if max_pieces and len(all_waveforms) >= max_pieces:
            break

    total = len(all_waveforms)
    if total == 0:
        if log_cb:
            log_cb("[终止] 未读取到任何波形数据")
        return {}

    logger.info(f"共读取 {total} 个波形片段")

    if progress_cb:
        progress_cb(0, f"共 {total} 个片段，开始分类...", 30)

    csv_path = os.path.join(output_dir, "summary.csv")
    counts = {c: 0 for c in class_names}
    correction_counts = {"manual_exact": 0, "adapter": 0}

    if log_cb:
        enabled_records = sum(
            1 for record in correction_context.records.values() if record.enabled
        )
        if correction_context.generation is None:
            log_cb(f"[纠错] 已加载 {enabled_records} 条人工记录，暂无泛化纠错模型")
        else:
            log_cb(
                f"[纠错] 已加载第 {correction_context.generation} 代纠错模型，"
                f"人工记录 {enabled_records} 条"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "piece_index", "time", "predicted_class", "confidence",
            "base_predicted_class", "base_confidence", "prediction_source",
            "correction_similarity",
        ] + [f"prob_{c}" for c in class_names])

        for start in range(0, total, 5000):
            end = min(start + 5000, total)
            chunk_waveforms = all_waveforms[start:end]

            base_predictions = classify_batch_detailed(
                chunk_waveforms, checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                daylights=[
                    _event_is_daylight(item[2]) for item in all_meta[start:end]
                ],
            )
            predictions = apply_feedback_batch(
                chunk_waveforms, base_predictions, correction_context
            )

            for offset, prediction in enumerate(predictions):
                gi = start + offset
                counts[prediction.effective_label] += 1
                if prediction.source in correction_counts:
                    correction_counts[prediction.source] += 1
                fname, piece_idx, time_key = all_meta[gi]
                effective_confidence = (
                    f"{prediction.base_confidence:.4f}"
                    if prediction.source == "base" else ""
                )
                similarity = (
                    "" if prediction.correction_similarity is None
                    else f"{prediction.correction_similarity:.4f}"
                )
                w.writerow([
                    fname, piece_idx, time_key, prediction.effective_label,
                    effective_confidence, prediction.base_label,
                    f"{prediction.base_confidence:.4f}", prediction.source,
                    similarity,
                ] + [f"{value:.4f}" for value in prediction.probabilities])
            pct = 30 + int((end / max(total, 1)) * 70)
            if progress_cb:
                progress_cb(0, f"分类中... {end}/{total}", pct)

    summary_lines = []
    summary_lines.append(f"{'='*50}")
    summary_lines.append(
        "纠错生效: "
        f"人工精确匹配 {correction_counts['manual_exact']} 条，"
        f"纠错模型 {correction_counts['adapter']} 条"
    )
    for c in class_names:
        n = counts.get(c, 0)
        pct_val = n / max(sum(counts.values()), 1) * 100
        summary_lines.append(f"  {c:<8}: {n:>7d}  ({pct_val:5.1f}%)")
    summary_lines.append(f"{'='*50}")
    summary_lines.append(f"结果保存至: {csv_path}")

    for line in summary_lines:
        logger.info(line)
        if log_cb:
            log_cb(line)

    if progress_cb:
        progress_cb(0, "分类完成！", 100)

    return counts


# ============================================================================
#                          批量原始波形分类 (用于文件加载时预分类)
# ============================================================================

def classify_batch_detailed(waveforms, checkpoint_path=None, batch_size=256,
                            daylights=None):
    """Batch inference returning base probabilities and encoder features."""
    session, class_names, _ = _ensure_model(checkpoint_path)

    if not waveforms:
        return []

    # Padding 到相同长度
    max_len = max(len(wf) for wf in waveforms)
    wf_array = np.zeros((len(waveforms), max_len), dtype=np.float32)
    for i, wf in enumerate(waveforms):
        L = min(len(wf), max_len)
        wf_array[i, :L] = wf[:L]

    if daylights is None:
        daylight_values = np.zeros((len(waveforms), 1), dtype=np.float32)
    else:
        if len(daylights) != len(waveforms):
            raise ValueError("daylights and waveforms must have equal lengths")
        daylight_values = np.asarray(daylights, dtype=np.float32).reshape(-1, 1)

    with _model_lock:
        schema = _model_schema
        metadata = dict(_model_metadata or {})
    if schema == 'ligedit_main_model_v2':
        config = metadata.get('preprocess_config') or {}
        local, global_view = preprocess_dual_view_batch(
            wf_array,
            local_length=int(config.get('local_length', 8000)),
            global_length=int(config.get('global_length', 2000)),
            use_filter=bool(config.get('use_filter', True)),
            cutoff_hz=float(config.get('cutoff_hz', 120000.0)),
            sample_rate_hz=float(config.get('sample_rate_hz', 5000000.0)),
        )
    else:
        wf_proc = preprocess_batch(wf_array, normalize_mode='minmax')

    results = []
    N = len(waveforms)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        if schema == 'ligedit_main_model_v2':
            inputs = {
                'local': local[s:e].reshape(-1, 1, 8000).astype(np.float32),
                'global_view': global_view[s:e].reshape(
                    -1, 1, 2000
                ).astype(np.float32),
                'daylight': daylight_values[s:e].astype(np.float32),
            }
        else:
            x = wf_proc[s:e].reshape(-1, 1, 8000).astype(np.float32)
            inputs = {'waveform': x}
        outputs = session.run(None, inputs)
        if len(outputs) < 2:
            raise ValueError("ONNX模型缺少纠错特征输出")
        logits, features = outputs[0], np.asarray(outputs[1], dtype=np.float32)
        if logits.shape != (e - s, len(CLASS_NAMES)):
            raise ValueError(f"ONNX分类输出形状错误: {logits.shape}")
        if features.ndim != 2 or features.shape[0] != e - s:
            raise ValueError(f"ONNX特征输出形状错误: {features.shape}")
        probs = _softmax(logits)
        for i in range(e - s):
            pred_idx = int(probs[i].argmax())
            feature = features[i]
            norm = float(np.linalg.norm(feature))
            if np.isfinite(norm) and norm > 0.0:
                feature = feature / norm
            else:
                feature = np.zeros_like(feature)
            results.append(BasePrediction(
                label=class_names[pred_idx],
                confidence=float(probs[i, pred_idx]),
                probabilities=tuple(float(value) for value in probs[i]),
                feature=np.asarray(feature, dtype=np.float32),
            ))

    return results


def classify_batch_raw(waveforms, checkpoint_path=None, batch_size=256):
    """Return final labels after applying feedback corrections."""
    return [
        (result.effective_label, result.base_confidence)
        for result in classify_batch_with_feedback(
            waveforms, checkpoint_path, batch_size
        )
    ]


def classify_batch_with_feedback(waveforms, checkpoint_path=None, batch_size=256,
                                 feedback_dir=None, context=None, daylights=None):
    """Run the base model and return labels corrected by local feedback."""
    base_predictions = classify_batch_detailed(
        waveforms,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        daylights=daylights,
    )
    return apply_feedback_batch(
        waveforms, base_predictions, context=context, feedback_dir=feedback_dir
    )


def _event_is_daylight(event_time):
    try:
        hour = int(str(event_time)[6:8])
        minute = int(str(event_time)[8:10])
    except (TypeError, ValueError):
        return False
    local_hour = (hour + 8 + minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0


def get_correction_model_info(feedback_dir=None, correction_model_dir=None):
    """Return user-facing metadata for the active feedback layer."""
    try:
        context = load_correction_context(feedback_dir, correction_model_dir)
        enabled_records = sum(
            1 for record in context.records.values() if record.enabled
        )
        return {
            "feedback_records": enabled_records,
            "correction_generation": context.generation,
            "correction_ready": (
                context.index is not None and context.index.threshold is not None
            ),
        }
    except Exception:
        return {
            "feedback_records": 0,
            "correction_generation": None,
            "correction_ready": False,
        }


# ============================================================================
#                          模型信息查询
# ============================================================================

def get_model_info(checkpoint_path=None):
    """获取模型信息（类别名等），不强制加载"""
    correction_info = get_correction_model_info()
    if _class_names is not None:
        return {
            "class_names": _class_names,
            "device": "cpu",
            **correction_info,
        }

    try:
        import onnx
        if checkpoint_path is None:
            checkpoint_path = _get_resource_path("checkpoints/ligclassify_legacy.onnx")
        if os.path.exists(checkpoint_path):
            return {
                "class_names": _DEFAULT_CLASS_NAMES,
                "device": "cpu",
                "model_hash": _sha256_file(checkpoint_path),
                **correction_info,
            }
    except ImportError:
        pass

    return {
        "class_names": _DEFAULT_CLASS_NAMES,
        "device": "cpu",
        **correction_info,
    }
