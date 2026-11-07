"""离线拟合风险概率校准器。

用法::

    python manage.py calibrate_risk

只读 ``ml/artifacts/language_risk_oof.csv`` 与 ``ml/endangerment_dataset.csv``，
在 OOF 概率上最小化负对数似然拟合温度缩放参数，写入独立产物
``ml/artifacts/risk_calibration.json``。不重训模型，不覆盖 metrics.json
或任何现有实验产物。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from Dashboard import risk_calibration as rc

OOF_PATH = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'language_risk_oof.csv'
DATASET_PATH = Path(settings.BASE_DIR) / 'ml' / 'endangerment_dataset.csv'
SPLITS_PATH = Path(settings.BASE_DIR) / 'ml' / 'reports' / 'splits_group_family_5fold.json'
OUTPUT_PATH = Path(settings.BASE_DIR) / 'ml' / 'artifacts' / 'risk_calibration.json'


def _load_oof_probabilities(path: Path) -> dict:
    if not path.exists():
        raise CommandError(f'未找到 OOF 产物：{path}')
    rows = {}
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                code_id = int(row.get('code_id') or 0)
            except (TypeError, ValueError):
                continue
            rows[code_id] = [row.get(f'p_{name}') for name in rc.CLASS_ORDER]
    return rows


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(settings.BASE_DIR))
    except ValueError:
        return str(path)


class Command(BaseCommand):
    help = '离线拟合风险概率温度校准器（只读 OOF，不重训模型）'

    def handle(self, *args, **options):
        if not DATASET_PATH.exists():
            raise CommandError(f'未找到数据集：{DATASET_PATH}')
        if not SPLITS_PATH.exists():
            raise CommandError(f'未找到折索引：{SPLITS_PATH}')

        dataset = pd.read_csv(
            DATASET_PATH, encoding='utf-8-sig',
            usecols=['code_id', 'target_5', 'trainable'],
        )
        trainable = dataset[dataset['trainable'] == 1].reset_index(drop=True)
        if trainable.empty:
            raise CommandError('数据集中没有可训练样本')

        labels = trainable['target_5'].map(
            {name: index for index, name in enumerate(rc.CLASS_ORDER)}
        )
        if labels.isna().any():
            bad = trainable.loc[labels.isna(), 'target_5'].unique().tolist()
            raise CommandError(f'存在无法映射的标签：{bad}')

        oof_rows = _load_oof_probabilities(OOF_PATH)
        missing = [int(code_id) for code_id in trainable['code_id']
                   if int(code_id) not in oof_rows]
        if missing:
            raise CommandError(f'OOF 缺少 {len(missing)} 条语言，例如 {missing[:5]}')

        raw = np.asarray(
            [[float(value) if value not in (None, '') else 0.0
              for value in oof_rows[int(code_id)]]
             for code_id in trainable['code_id']],
            dtype=float,
        )
        raw = rc.temperature_scale_matrix(raw, 1.0)  # 统一归一化，T=1 不改变分布
        labels_array = labels.to_numpy(dtype=int)

        temperature = rc.fit_temperature(raw, labels_array)
        calibrated = rc.temperature_scale_matrix(raw, temperature)
        raw_metrics = rc.calibration_metrics(raw, labels_array)
        calibrated_metrics = rc.calibration_metrics(calibrated, labels_array)

        raw_argmax = raw.argmax(axis=1)
        calibrated_argmax = calibrated.argmax(axis=1)
        argmax_changes = int((raw_argmax != calibrated_argmax).sum())

        # 交叉拟合评估：每折温度只用其余折拟合，避免在同一批 OOF 上自评。
        splits = json.loads(SPLITS_PATH.read_text(encoding='utf-8'))
        cross_fitted = np.zeros_like(raw)
        covered = np.zeros(len(raw), dtype=bool)
        fold_temperatures = []
        for fold in splits.get('folds', []):
            fit_idx = np.asarray(fold.get('train_idx', []), dtype=int)
            test_idx = np.asarray(fold.get('test_idx', []), dtype=int)
            if len(fit_idx) == 0 or len(test_idx) == 0:
                raise CommandError(f'折 {fold.get("fold")} 的索引为空')
            expected_ids = trainable.iloc[test_idx]['code_id'].tolist()
            saved_ids = fold.get('test_code_ids')
            if saved_ids is not None and list(saved_ids) != expected_ids:
                raise CommandError(
                    f'折 {fold.get("fold")} 的 test_code_ids 与数据集顺序不一致')
            fold_temperature = rc.fit_temperature(raw[fit_idx], labels_array[fit_idx])
            cross_fitted[test_idx] = rc.temperature_scale_matrix(
                raw[test_idx], fold_temperature)
            covered[test_idx] = True
            fold_temperatures.append({
                'fold': fold.get('fold'),
                'temperature': round(float(fold_temperature), 6),
                'rows': int(len(test_idx)),
            })
        if not covered.all():
            raise CommandError('交叉拟合未覆盖全部 OOF 样本')

        cross_metrics = rc.calibration_metrics(cross_fitted, labels_array)

        payload = {
            'version': 1,
            'method': rc.CALIBRATION_METHOD,
            'temperature': round(float(temperature), 6),
            'classes': list(rc.CLASS_ORDER),
            'fitted_on': {
                'oof_file': _relative(OOF_PATH),
                'label_file': _relative(DATASET_PATH),
                'split_file': _relative(SPLITS_PATH),
                'rows': int(len(raw)),
                'oof_sha256': hashlib.sha256(OOF_PATH.read_bytes()).hexdigest(),
                'generated_at': datetime.now().isoformat(timespec='seconds'),
            },
            'metrics': {
                'n_bins': 10,
                'uncalibrated': raw_metrics,
                'calibrated': calibrated_metrics,
                'cross_fitted_calibrated': cross_metrics,
                'delta': {
                    'ece': round(calibrated_metrics['ece'] - raw_metrics['ece'], 6),
                    'brier': round(calibrated_metrics['brier'] - raw_metrics['brier'], 6),
                },
                'argmax_changes': argmax_changes,
                'argmax_change_rate': round(argmax_changes / len(raw), 6),
                'argmax_changed': argmax_changes > 0,
                'cross_fitted_folds': fold_temperatures,
            },
            'notes': [
                '温度缩放在 OOF 概率上最小化 NLL 拟合，只调整概率尺度，不改变 argmax。',
                'calibrated 指标为同一批 OOF 上的拟合后表现；cross_fitted_calibrated 为逐折交叉拟合的保守估计。',
                '校准只用于改善概率展示与累积概率口径，不代表经过外部验证的绝对灭绝概率。',
            ],
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = OUTPUT_PATH.with_name(OUTPUT_PATH.name + '.tmp')
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp_path, OUTPUT_PATH)

        self.stdout.write(f'rows: {len(raw)}')
        self.stdout.write(f'temperature: {temperature:.6f}')
        self.stdout.write('ECE  raw={:.6f}  calibrated={:.6f}  cross_fitted={:.6f}'.format(
            raw_metrics['ece'], calibrated_metrics['ece'], cross_metrics['ece']))
        self.stdout.write('Brier raw={:.6f}  calibrated={:.6f}  cross_fitted={:.6f}'.format(
            raw_metrics['brier'], calibrated_metrics['brier'], cross_metrics['brier']))
        self.stdout.write(
            'max confidence raw mean={:.4f}  calibrated mean={:.4f}'.format(
                raw_metrics['max_confidence']['mean'],
                calibrated_metrics['max_confidence']['mean']))
        self.stdout.write(f'argmax changes: {argmax_changes}')
        self.stdout.write(f'macro F1 raw={raw_metrics["macro_f1"]:.6f}  '
                          f'calibrated={calibrated_metrics["macro_f1"]:.6f}')
        self.stdout.write(f'written: {OUTPUT_PATH}')
