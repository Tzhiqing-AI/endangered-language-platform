"""
濒危语言风险预测推理服务。

模型由 ``ml/train_models.py`` 离线训练完成，Web 进程只做加载与推理：
- 详情页 / 地图 / 预警列表默认读取 ``ml/artifacts/language_risk_oof.csv``
  （按语系分组 5 折的 out-of-fold 评分，每条语言由没见过它的模型预测）；
  OOF 产物缺失时回退到 ``language_risk_predictions.csv``（训练集重代入，仅对照）
- 需要对新样本评分时调用 :func:`predict_features`，复用全量重训的部署模型，
  不会在请求中重新训练

缓存按文件指纹（mtime + 大小）自动失效：模型重训后新的 Web 进程无需重启，
下一次取数时用 ``os.stat`` 级别的检查发现文件变化并重建缓存。
"""

from __future__ import annotations

import csv
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from django.conf import settings

from Dashboard import risk_calibration
from Dashboard.common import display_language_name

PREDICTION_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'language_risk_oof.csv'
RESUBSTITUTION_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'language_risk_predictions.csv'
SERVICE_META_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'risk_service_meta.json'
MODEL_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'risk_model.joblib'
CALIBRATION_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'risk_calibration.json'

RISK_BAND_LABELS = {
    'low': '低风险',
    'medium': '中风险',
    'high': '高风险',
    'critical': '极高风险',
}

RISK_BAND_COLORS = {
    'low': '#10b981',
    'medium': '#f59e0b',
    'high': '#f97316',
    'critical': '#dc2626',
}

# 新的三档口径来自累积概率，与旧的四档 risk_band 并存。
RISK_TIER_LABELS = risk_calibration.RISK_TIER_LABELS
RISK_TIER_COLORS = risk_calibration.RISK_TIER_COLORS

_lock = threading.Lock()
# 缓存状态统一保存为 (文件指纹, 数据)，整体替换保证读写原子性。
# 预测缓存指纹包含「预测文件 + 校准文件」，任一份变化都会触发重建。
_predictions_state: Optional[Tuple[Optional[Tuple], Dict[int, Dict]]] = None
_meta_state: Optional[Tuple[Optional[Tuple[int, int]], Dict]] = None
_model_state: Optional[Tuple[Optional[Tuple[int, int]], object]] = None
_calibration_state: Optional[Tuple[Optional[Tuple[int, int]], Dict]] = None

# 缓存实际重建次数，仅用于验证「文件未变化时不重复解析」
_load_counts = {'predictions': 0, 'meta': 0, 'model': 0, 'calibration': 0}


def _fingerprint(path: Path) -> Optional[Tuple[int, int]]:
    """返回 (mtime_ns, size)；文件不可用时返回 None。"""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _file_sha256(path: Path) -> Optional[str]:
    """计算文件的 sha256；文件不可读时返回 None。"""
    try:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _legacy_risk_band(score: float) -> str:
    """旧的四档分箱，与 ml/train_models.py 的 pandas.cut 保持一致。"""
    if score <= 40:
        return 'low'
    if score <= 60:
        return 'medium'
    if score <= 80:
        return 'high'
    return 'critical'


def reload_cache() -> None:
    """清空进程内缓存，模型重新训练后调用（文件变化也会自动触发重建）。"""
    global _predictions_state, _meta_state, _model_state, _calibration_state
    with _lock:
        _predictions_state = None
        _meta_state = None
        _model_state = None
        _calibration_state = None


def cache_stats() -> Dict[str, int]:
    """返回各缓存的实际重建次数，便于确认未变化时没有重复解析。"""
    return dict(_load_counts)


def _active_prediction_file() -> Path:
    """优先使用 OOF 产物；缺失时回退到训练集重代入结果（仅对照）。"""
    return PREDICTION_FILE if PREDICTION_FILE.exists() else RESUBSTITUTION_FILE


def _load_calibration() -> Tuple[Optional[Tuple[int, int]], Dict]:
    """懒加载校准产物；缺失或损坏时返回明确的降级状态，不抛异常。"""
    global _calibration_state
    signature = _fingerprint(CALIBRATION_FILE)
    state = _calibration_state
    if state is not None and state[0] == signature:
        return signature, state[1]

    with _lock:
        state = _calibration_state
        if state is not None and state[0] == signature:
            return signature, state[1]

        payload: Optional[Dict] = None
        if signature is None:
            status = 'missing'
            error = '校准文件缺失'
        else:
            try:
                payload = json.loads(CALIBRATION_FILE.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                status = 'invalid'
                error = f'校准文件损坏：{exc.__class__.__name__}'
                payload = None
            else:
                valid, reason = risk_calibration.validate_calibration_payload(payload)
                if valid:
                    expected_hash = (payload.get('fitted_on') or {}).get('oof_sha256')
                    current_hash = (_file_sha256(PREDICTION_FILE)
                                    if expected_hash and PREDICTION_FILE.exists() else None)
                    if expected_hash and current_hash and current_hash != expected_hash:
                        # OOF 已被重新生成，旧校准器不能静默套用。
                        status = 'stale'
                        error = '校准文件与当前 OOF 产物不匹配，请重新运行 calibrate_risk'
                        payload = None
                    else:
                        status = 'ok'
                        error = ''
                else:
                    status = 'invalid'
                    error = f'校准文件不可用：{reason}'
                    payload = None

        result = {'status': status, 'payload': payload, 'error': error}
        _calibration_state = (signature, result)
        _load_counts['calibration'] += 1
        return signature, result


def _load_predictions() -> Dict[int, Dict]:
    global _predictions_state
    path = _active_prediction_file()
    prediction_signature = _fingerprint(path)
    calibration_signature, calibration_state = _load_calibration()
    # 预测缓存同时受预测文件和校准文件指纹约束
    signature = (prediction_signature, calibration_signature)
    state = _predictions_state
    if state is not None and state[0] == signature:
        return state[1]

    with _lock:
        state = _predictions_state
        if state is not None and state[0] == signature:
            return state[1]
        data = _parse_predictions(path, calibration_state)
        _predictions_state = (signature, data)
        _load_counts['predictions'] += 1
        return data


def _parse_predictions(path: Path, calibration_state: Dict) -> Dict[int, Dict]:
    data: Dict[int, Dict] = {}
    if not path.exists():
        return data

    calibration_payload = (calibration_state.get('payload')
                           if calibration_state.get('status') == 'ok' else None)
    calibration_error = (calibration_state.get('error') or ''
                         if calibration_state.get('status') != 'ok' else '')

    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            try:
                code_id = int(row.get('code_id') or 0)
            except (TypeError, ValueError):
                continue
            try:
                factors = json.loads(row.get('top_factors') or '[]')
            except (TypeError, json.JSONDecodeError):
                factors = []

            raw_values = [row.get(f'p_{name}') for name in risk_calibration.CLASS_ORDER]
            if all(value in (None, '') for value in raw_values):
                # 兼容旧产物：没有概率列时，用预测等级构造 one-hot 作为兜底。
                predicted = (row.get('predicted_level') or '').strip()
                raw_values = [
                    1.0 if name == predicted else 0.0
                    for name in risk_calibration.CLASS_ORDER
                ]

            profile = risk_calibration.build_probability_profile(
                raw_values, calibration_payload, calibration_error)

            band = (row.get('risk_band') or 'low').strip()
            data[code_id] = {
                # ---- 向后兼容字段（risk_score / confidence / risk_band 为旧口径）----
                'code_id': code_id,
                'language_name': display_language_name({
                    'language_id': code_id,
                    'language_name': row.get('language_name'),
                }),
                'risk_score': round(float(row.get('risk_score') or 0), 1),
                'confidence': round(float(row.get('confidence') or 0), 3),
                'risk_band': band,
                'risk_band_label': RISK_BAND_LABELS.get(band, band),
                'risk_band_color': RISK_BAND_COLORS.get(band, '#64748b'),
                'factors': factors,
                'prediction_source': (row.get('prediction_source') or 'resubstitution').strip(),
                # ---- 新口径：校准后概率、累积概率、三档、top-2 ----
                **profile,
            }
    return data


def get_service_meta() -> Dict:
    global _meta_state
    signature = _fingerprint(SERVICE_META_FILE)
    state = _meta_state
    if state is not None and state[0] == signature:
        return state[1]

    with _lock:
        state = _meta_state
        if state is not None and state[0] == signature:
            return state[1]
        payload: Dict = {}
        if SERVICE_META_FILE.exists():
            try:
                payload = json.loads(SERVICE_META_FILE.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                payload = {}
        _meta_state = (signature, payload)
        _load_counts['meta'] += 1
        return payload


def get_risk_profile(language_id: int) -> Optional[Dict]:
    """按语言主键返回风险画像。

    新口径字段：
    - ``high_risk_prob`` / ``at_least_danger_prob`` / ``severe_risk_prob``：累积概率；
    - ``risk_tier``：按累积概率过半规则得到的三档等级；
    - ``top2``：前两类及各自概率，让模型在两类间摇摆时可见。

    为兼容旧调用方，原有 ``risk_score``、``confidence``、``risk_band``、
    ``probabilities`` 等键仍然保留；其中 ``risk_score`` / ``confidence`` /
    ``risk_band`` 仍是旧口径，已标记为 deprecated，界面不再作为主展示。
    """
    return _load_predictions().get(int(language_id))


def get_alerts(limit: int = 10, min_score: float = 80.0,
               band: Optional[str] = None,
               min_high_prob: Optional[float] = None) -> List[Dict]:
    """返回高风险语言预警列表。

    ``min_score`` 为兼容旧调用保留的参数名，现按「高危及以上累积概率」的
    百分比解释（0-100）；``min_high_prob`` 是同一阈值的小数写法（0-1）。
    ``band`` 仍指旧的四档 ``risk_band``，仅用于兼容旧筛选。
    排序优先级为：高危累积概率 → 垂危及以上累积概率 → 灭绝概率。
    """
    threshold = (float(min_high_prob) * 100.0
                 if min_high_prob is not None else float(min_score))
    items = [
        item for item in _load_predictions().values()
        if item['high_risk_prob_pct'] >= threshold
        and (band is None or item['risk_band'] == band)
    ]
    items.sort(
        key=lambda item: (
            item['high_risk_prob'],
            item['severe_risk_prob'],
            item['extinct_prob'],
            item['risk_score'],
        ),
        reverse=True,
    )
    return items[:limit]


def get_risk_stats() -> Dict:
    """风险预测概览统计，供数据管理页面展示。"""
    predictions = _load_predictions()
    _, calibration_state = _load_calibration()
    calibration_applied = calibration_state.get('status') == 'ok'
    if not predictions:
        return {
            'total': 0,
            'band_counts': {},
            'tier_counts': {},
            'critical_count': 0,
            'high_count': 0,
            'average_score': 0.0,
            'average_high_risk_prob': 0.0,
            'average_high_risk_prob_pct': 0.0,
            'calibration_applied': calibration_applied,
            'calibration_status': calibration_state.get('status', 'missing'),
            'calibration_reason': calibration_state.get('error', ''),
            'model_name': get_service_meta().get('model_name', ''),
            'trained_at': get_service_meta().get('trained_at', ''),
        }
    band_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    for item in predictions.values():
        band_counts[item['risk_band']] = band_counts.get(item['risk_band'], 0) + 1
        tier_counts[item['risk_tier']] = tier_counts.get(item['risk_tier'], 0) + 1
    scores = [item['risk_score'] for item in predictions.values()]
    high_risk_probs = [item['high_risk_prob'] for item in predictions.values()]
    average_high_risk_prob = sum(high_risk_probs) / len(high_risk_probs)
    return {
        'total': len(predictions),
        # 旧四档统计保留，供地图/旧调用方使用
        'band_counts': band_counts,
        'critical_count': band_counts.get('critical', 0),
        'high_count': band_counts.get('high', 0),
        'average_score': round(sum(scores) / len(scores), 1),
        # 新三档与累积概率统计
        'tier_counts': tier_counts,
        'average_high_risk_prob': round(average_high_risk_prob, 4),
        'average_high_risk_prob_pct': round(average_high_risk_prob * 100.0, 1),
        'calibration_applied': calibration_applied,
        'calibration_status': calibration_state.get('status', 'missing'),
        'calibration_reason': calibration_state.get('error', ''),
        'model_name': get_service_meta().get('model_name', ''),
        'trained_at': get_service_meta().get('trained_at', ''),
    }


def get_model_bundle():
    """返回推理模型包（含特征列与预处理管线）；文件缺失时为 None。"""
    global _model_state
    signature = _fingerprint(MODEL_FILE)
    state = _model_state
    if state is not None and state[0] == signature:
        return state[1]

    with _lock:
        state = _model_state
        if state is not None and state[0] == signature:
            return state[1]
        bundle = None
        if signature is not None:
            import joblib

            bundle = joblib.load(MODEL_FILE)
        _model_state = (signature, bundle)
        _load_counts['model'] += 1
        return bundle


def predict_features(feature_row: Dict) -> Optional[Dict]:
    """对单条特征记录做推理。

    传入的特征按 ``ml/dataset_meta.json`` 中的 feature 名称组织，
    缺失字段会由训练时的填充器补齐。
    """
    bundle = get_model_bundle()
    if bundle is None:
        return None

    import numpy as np
    import pandas as pd

    frame = pd.DataFrame([feature_row])
    columns = bundle['feature_columns']
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan

    for column in bundle['categorical_columns']:
        frame[column] = frame[column].fillna('unknown').astype(str).str.strip()

    matrix = bundle['preprocessor'].transform(frame[columns])
    proba = bundle['estimator'].predict_proba(matrix)[0]

    classes = list(bundle['class_names'])
    weights = np.array(bundle['risk_weights'], dtype=float)
    predicted_index = int(np.argmax(proba))
    score = float(np.dot(proba, weights))
    band = _legacy_risk_band(score)

    # 单条推理也走与页面相同的校准与累积概率口径。
    _, calibration_state = _load_calibration()
    calibration_payload = (calibration_state.get('payload')
                           if calibration_state.get('status') == 'ok' else None)
    calibration_error = (calibration_state.get('error') or ''
                         if calibration_state.get('status') != 'ok' else '')
    raw_in_order = [
        float(proba[classes.index(name)]) if name in classes else 0.0
        for name in risk_calibration.CLASS_ORDER
    ]
    profile = risk_calibration.build_probability_profile(
        raw_in_order, calibration_payload, calibration_error)

    return {
        # ---- 向后兼容字段 ----
        'predicted_level': classes[predicted_index],
        'risk_score': round(score, 1),
        'confidence': round(float(proba[predicted_index]), 3),
        'risk_band': band,
        'risk_band_label': RISK_BAND_LABELS.get(band, band),
        'risk_band_color': RISK_BAND_COLORS.get(band, '#64748b'),
        # ---- 新口径 ----
        **profile,
    }
