"""风险预测推理服务测试：单条推理、模型包指纹、分档边界与校准降级。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from Dashboard import risk_calibration, risk_service


class _RecordingPreprocessor:
    def __init__(self):
        self.seen_columns = None
        self.seen_frame = None

    def transform(self, frame):
        self.seen_columns = list(frame.columns)
        self.seen_frame = frame.copy()
        return np.zeros((len(frame), 1))


class _FixedEstimator:
    def __init__(self, proba):
        self.proba = list(proba)

    def predict_proba(self, matrix):
        return np.array([self.proba for _ in range(len(matrix))])


@pytest.fixture(autouse=True)
def _clear_cache():
    risk_service.reload_cache()
    for key in risk_service._load_counts:
        risk_service._load_counts[key] = 0
    yield
    risk_service.reload_cache()


def _bundle(preprocessor=None, proba=None):
    features = ['f1', 'f2', 'cat']
    return {
        'feature_columns': features,
        'categorical_columns': ['cat'],
        'preprocessor': preprocessor or _RecordingPreprocessor(),
        'estimator': _FixedEstimator(proba or [0.1, 0.2, 0.4, 0.2, 0.1]),
        'class_names': list(risk_calibration.CLASS_ORDER),
        'risk_weights': [0.0, 25.0, 50.0, 75.0, 100.0],
    }


def _calibration_payload(**overrides):
    payload = {
        'method': risk_calibration.CALIBRATION_METHOD,
        'temperature': 1.5,
        'classes': list(risk_calibration.CLASS_ORDER),
    }
    payload.update(overrides)
    return payload


def test_predict_features_aligns_columns_and_fills_missing(monkeypatch):
    preprocessor = _RecordingPreprocessor()
    monkeypatch.setattr(risk_service, 'get_model_bundle', lambda: _bundle(preprocessor))
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', Path('does-not-exist.json'))
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', Path('does-not-exist.csv'))

    result = risk_service.predict_features({'f1': 0.8, 'unrelated': 'x'})

    assert preprocessor.seen_columns == ['f1', 'f2', 'cat']
    assert preprocessor.seen_frame['f2'].isna().all()
    assert preprocessor.seen_frame['cat'].iloc[0] == 'unknown'
    assert result['predicted_level'] == 'endangered'
    assert result['risk_score'] == 50.0  # 0*0.1 + 25*0.2 + 50*0.4 + 75*0.2 + 100*0.1
    assert result['risk_band'] == 'medium'
    assert result['confidence'] == 0.4
    assert result['calibration']['applied'] is False


def test_predict_features_returns_none_without_model(monkeypatch):
    monkeypatch.setattr(risk_service, 'MODEL_FILE', Path('missing-model.joblib'))
    assert risk_service.get_model_bundle() is None
    assert risk_service.predict_features({'f1': 1.0}) is None


def test_model_bundle_is_cached_and_reloaded_on_change(tmp_path, monkeypatch):
    model_path = tmp_path / 'risk_model.joblib'
    joblib.dump({'marker': 1}, model_path)
    monkeypatch.setattr(risk_service, 'MODEL_FILE', model_path)

    assert risk_service.get_model_bundle()['marker'] == 1
    assert risk_service.get_model_bundle()['marker'] == 1
    assert risk_service.cache_stats()['model'] == 1

    joblib.dump({'marker': 2}, model_path)
    stat = model_path.stat()
    os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert risk_service.get_model_bundle()['marker'] == 2
    assert risk_service.cache_stats()['model'] == 2


@pytest.mark.parametrize('score,expected', [
    (0.0, 'low'),
    (40.0, 'low'),
    (40.1, 'medium'),
    (60.0, 'medium'),
    (60.1, 'high'),
    (80.0, 'high'),
    (80.1, 'critical'),
])
def test_legacy_risk_band_boundaries(score, expected):
    assert risk_service._legacy_risk_band(score) == expected


@pytest.mark.parametrize('values,expected', [
    ([0.6, 0.4, 0.0, 0.0, 0.0], 'low'),            # 垂危及以上 0.4 < 0.5
    ([0.2501, 0.25, 0.4999, 0.0, 0.0], 'medium'),  # 高危概率 0.4999
    ([0.25, 0.25, 0.5, 0.0, 0.0], 'high'),          # 恰好 0.5
    ([0.2499, 0.25, 0.5001, 0.0, 0.0], 'high'),     # 0.5001
])
def test_tier_boundaries(values, expected):
    profile = risk_calibration.build_probability_profile(values)
    assert profile['risk_tier'] == expected


def test_calibration_missing_marks_not_applied(monkeypatch, tmp_path):
    monkeypatch.setattr(risk_service, 'get_model_bundle', lambda: _bundle())
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', tmp_path / 'missing.json')
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', tmp_path / 'missing.csv')

    result = risk_service.predict_features({'f1': 1.0})

    assert result['calibration']['applied'] is False
    assert '缺失' in result['calibration']['reason']


def test_calibration_corrupt_marks_not_applied(monkeypatch, tmp_path):
    calibration_path = tmp_path / 'calibration.json'
    calibration_path.write_text('{not-json', encoding='utf-8')
    monkeypatch.setattr(risk_service, 'get_model_bundle', lambda: _bundle())
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', calibration_path)
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', tmp_path / 'missing.csv')

    result = risk_service.predict_features({'f1': 1.0})

    assert result['calibration']['applied'] is False
    assert '损坏' in result['calibration']['reason']


def test_calibration_hash_mismatch_is_stale(monkeypatch, tmp_path):
    calibration_path = tmp_path / 'calibration.json'
    calibration_path.write_text(
        json.dumps(_calibration_payload(
            fitted_on={'oof_sha256': 'deadbeef'},
        )),
        encoding='utf-8',
    )
    prediction_path = tmp_path / 'oof.csv'
    prediction_path.write_text('code_id\n', encoding='utf-8')
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', calibration_path)
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', prediction_path)
    monkeypatch.setattr(risk_service, 'RESUBSTITUTION_FILE', tmp_path / 'missing.csv')

    assert risk_service.get_risk_stats()['calibration_status'] == 'stale'
    stats = risk_service.get_risk_stats()
    assert stats['calibration_applied'] is False


def test_calibration_applied_when_valid(monkeypatch, tmp_path):
    calibration_path = tmp_path / 'calibration.json'
    calibration_path.write_text(json.dumps(_calibration_payload()), encoding='utf-8')
    monkeypatch.setattr(risk_service, 'get_model_bundle', lambda: _bundle())
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', calibration_path)
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', tmp_path / 'missing.csv')

    result = risk_service.predict_features({'f1': 1.0})

    assert result['calibration']['applied'] is True
    assert result['calibration']['temperature'] == 1.5


def test_risk_stats_empty_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', tmp_path / 'missing.csv')
    monkeypatch.setattr(risk_service, 'RESUBSTITUTION_FILE', tmp_path / 'missing2.csv')
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', tmp_path / 'missing.json')
    monkeypatch.setattr(risk_service, 'SERVICE_META_FILE', tmp_path / 'missing_meta.json')

    stats = risk_service.get_risk_stats()

    assert stats['total'] == 0
    assert stats['average_score'] == 0.0
    assert stats['calibration_status'] == 'missing'
    assert stats['model_name'] == ''


def test_old_artifact_without_probability_columns_falls_back(monkeypatch, tmp_path):
    csv_path = tmp_path / 'resubstitution.csv'
    csv_path.write_text(
        'code_id,predicted_level,risk_band,top_factors\n'
        '7,extinct,high,"[]"\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', tmp_path / 'missing_oof.csv')
    monkeypatch.setattr(risk_service, 'RESUBSTITUTION_FILE', csv_path)
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', tmp_path / 'missing.json')
    monkeypatch.setattr(risk_service, 'SERVICE_META_FILE', tmp_path / 'missing_meta.json')

    stats = risk_service.get_risk_stats()

    assert stats['total'] == 1
    assert stats['high_count'] == 1
    item = risk_service.get_alerts(limit=1, min_score=0)[0]
    assert item['code_id'] == 7
    assert item['risk_tier'] == 'high'


def test_get_risk_profile_and_alerts_use_predictions(monkeypatch, tmp_path):
    csv_path = tmp_path / 'preds.csv'
    csv_path.write_text(
        'code_id,language_name,predicted_level,risk_band,risk_score,confidence,'
        'top_factors,p_safe,p_vulnerable,p_endangered,p_seriously_endangered,p_extinct\n'
        '11,TestLang,endangered,high,70.0,0.7,"[]",0.1,0.1,0.6,0.1,0.1\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', csv_path)
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', tmp_path / 'missing.json')
    monkeypatch.setattr(risk_service, 'SERVICE_META_FILE', tmp_path / 'missing_meta.json')

    profile = risk_service.get_risk_profile(11)
    assert profile['risk_tier'] == 'high'
    assert profile['risk_score'] == 70.0

    alerts = risk_service.get_alerts(limit=5, min_score=0)
    assert [a['code_id'] for a in alerts] == [11]
    assert risk_service.get_risk_profile(999) is None
