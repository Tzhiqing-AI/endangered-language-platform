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
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from django.conf import settings

from Dashboard.common import display_language_name

PREDICTION_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'language_risk_oof.csv'
RESUBSTITUTION_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'language_risk_predictions.csv'
SERVICE_META_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'risk_service_meta.json'
MODEL_FILE = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'risk_model.joblib'

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

_lock = threading.Lock()
# 缓存状态统一保存为 (文件指纹, 数据)，整体替换保证读写原子性
_predictions_state: Optional[Tuple[Optional[Tuple[int, int]], Dict[int, Dict]]] = None
_meta_state: Optional[Tuple[Optional[Tuple[int, int]], Dict]] = None
_model_state: Optional[Tuple[Optional[Tuple[int, int]], object]] = None

# 缓存实际重建次数，仅用于验证「文件未变化时不重复解析」
_load_counts = {'predictions': 0, 'meta': 0, 'model': 0}


def _fingerprint(path: Path) -> Optional[Tuple[int, int]]:
    """返回 (mtime_ns, size)；文件不可用时返回 None。"""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def reload_cache() -> None:
    """清空进程内缓存，模型重新训练后调用（文件变化也会自动触发重建）。"""
    global _predictions_state, _meta_state, _model_state
    with _lock:
        _predictions_state = None
        _meta_state = None
        _model_state = None


def cache_stats() -> Dict[str, int]:
    """返回各缓存的实际重建次数，便于确认未变化时没有重复解析。"""
    return dict(_load_counts)


def _active_prediction_file() -> Path:
    """优先使用 OOF 产物；缺失时回退到训练集重代入结果（仅对照）。"""
    return PREDICTION_FILE if PREDICTION_FILE.exists() else RESUBSTITUTION_FILE


def _load_predictions() -> Dict[int, Dict]:
    global _predictions_state
    path = _active_prediction_file()
    signature = _fingerprint(path)
    state = _predictions_state
    if state is not None and state[0] == signature:
        return state[1]

    with _lock:
        state = _predictions_state
        if state is not None and state[0] == signature:
            return state[1]
        data = _parse_predictions(path)
        _predictions_state = (signature, data)
        _load_counts['predictions'] += 1
        return data


def _parse_predictions(path: Path) -> Dict[int, Dict]:
    data: Dict[int, Dict] = {}
    if not path.exists():
        return data
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
            probabilities = {
                key[2:]: float(value)
                for key, value in row.items()
                if key.startswith('p_') and value not in (None, '')
            }
            band = (row.get('risk_band') or 'low').strip()
            data[code_id] = {
                'code_id': code_id,
                'language_name': display_language_name({
                    'language_id': code_id,
                    'language_name': row.get('language_name'),
                }),
                'predicted_level': row.get('predicted_level') or '',
                'predicted_level_zh': row.get('predicted_level_zh') or '',
                'risk_score': round(float(row.get('risk_score') or 0), 1),
                'confidence': round(float(row.get('confidence') or 0), 3),
                'risk_band': band,
                'risk_band_label': RISK_BAND_LABELS.get(band, band),
                'risk_band_color': RISK_BAND_COLORS.get(band, '#64748b'),
                'probabilities': probabilities,
                'factors': factors,
                'prediction_source': (row.get('prediction_source') or 'resubstitution').strip(),
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
    """按语言主键返回风险评分、概率分布和主要风险因素。"""
    return _load_predictions().get(int(language_id))


def get_alerts(limit: int = 10, min_score: float = 80.0,
               band: Optional[str] = None) -> List[Dict]:
    """返回高风险语言预警列表，按风险评分降序。"""
    items = [
        item for item in _load_predictions().values()
        if item['risk_score'] >= min_score and (band is None or item['risk_band'] == band)
    ]
    items.sort(key=lambda item: item['risk_score'], reverse=True)
    return items[:limit]


def get_risk_stats() -> Dict:
    """风险预测概览统计，供数据管理页面展示。"""
    predictions = _load_predictions()
    if not predictions:
        return {
            'total': 0,
            'band_counts': {},
            'critical_count': 0,
            'high_count': 0,
            'average_score': 0.0,
            'model_name': get_service_meta().get('model_name', ''),
            'trained_at': get_service_meta().get('trained_at', ''),
        }
    band_counts: Dict[str, int] = {}
    for item in predictions.values():
        band_counts[item['risk_band']] = band_counts.get(item['risk_band'], 0) + 1
    scores = [item['risk_score'] for item in predictions.values()]
    return {
        'total': len(predictions),
        'band_counts': band_counts,
        'critical_count': band_counts.get('critical', 0),
        'high_count': band_counts.get('high', 0),
        'average_score': round(sum(scores) / len(scores), 1),
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

    classes = bundle['class_names']
    weights = np.array(bundle['risk_weights'], dtype=float)
    predicted_index = int(np.argmax(proba))
    score = float(np.dot(proba, weights))
    return {
        'predicted_level': classes[predicted_index],
        'risk_score': round(score, 1),
        'confidence': round(float(proba[predicted_index]), 3),
        'probabilities': {name: round(float(p), 4) for name, p in zip(classes, proba)},
    }
