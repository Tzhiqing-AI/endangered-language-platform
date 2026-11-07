"""
弱信息 9 特征 × 多模型横向对比。

只新增产物，不覆盖任何既有文件：
  ml/reports/weak_models_comparison.json
  ml/reports/weak_models_comparison.csv

口径（与既有实验严格一致）：
  - 数据：ml/endangerment_dataset.csv 中 trainable=1 的 3,462 行
  - 折：ml/reports/splits_group_family_5fold.json（group_family_5fold，seed=42）
  - 留出集：同一份 splits 文件里的 holdout（train 2,258 / test 1,204）
  - 特征：ml/feature_groups.json 的弱信息 8 列 + neighbor_density_100km
  - 预处理：train_models.SklearnRiskModel（训练折内中位数填充 + 标准化；
    类别缺失记 unknown + OneHot），与主实验完全相同

复算：
  python ml/weak_models_comparison.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from catboost import CatBoostClassifier  # noqa: E402

from train_models import (  # noqa: E402
    CATEGORICAL, DATASET_PATH, META_PATH, LEVEL_ORDER, REPORT_DIR, SEED,
    SPLITS_PATH, SklearnRiskModel, build_weak_matrix, compute_metrics,
    dump_json, load_feature_groups, log, model_factories, prepare_frame,
    run_cv, summarize_folds,
)

METRICS_PATH = REPORT_DIR / 'metrics.json'
OUT_JSON = REPORT_DIR / 'weak_models_comparison.json'
OUT_CSV = REPORT_DIR / 'weak_models_comparison.csv'
CORE_MODELS = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 'MLP-Embedding']
BASELINE_CV = 0.2878451033742687  # metrics.json: weak_information 的 5 折 macro F1 均值


def extra_factories() -> dict:
    """9 列弱信息下补充的模型：都走同一 SklearnRiskModel 预处理。"""
    return {
        'CatBoost': lambda: SklearnRiskModel(CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            loss_function='MultiClass', random_seed=SEED,
            verbose=0, thread_count=-1,
            train_dir=str(Path(tempfile.gettempdir()) / 'catboost_info'),
        )),
        'KNN': lambda: SklearnRiskModel(KNeighborsClassifier(
            n_neighbors=5, weights='distance', n_jobs=-1,
        )),
        'GaussianNB': lambda: SklearnRiskModel(GaussianNB()),
        'DecisionTree': lambda: SklearnRiskModel(DecisionTreeClassifier(
            random_state=SEED, class_weight='balanced',
        )),
    }


def dummy_factories() -> dict:
    return {
        'Dummy-most_frequent': lambda: SklearnRiskModel(
            DummyClassifier(strategy='most_frequent')),
        'Dummy-stratified': lambda: SklearnRiskModel(
            DummyClassifier(strategy='stratified', random_state=SEED)),
    }


def evaluate_set(X: pd.DataFrame, y: np.ndarray, folds, holdout_tr, holdout_te,
                 factories: dict, tag: str) -> dict:
    out = {}
    for name, factory in factories.items():
        t0 = time.time()
        rows, _, _ = run_cv(
            X, y, folds, factory, LEVEL_ORDER, 'target_5',
            'group_family_5fold', name, verbose=False)
        cv = summarize_folds(rows, LEVEL_ORDER)

        model = factory()
        model.fit(X.iloc[holdout_tr].reset_index(drop=True), y[holdout_tr])
        proba = model.predict_proba(X.iloc[holdout_te].reset_index(drop=True))
        pred = np.argmax(proba, axis=1)
        ho = compute_metrics(y[holdout_te], pred, proba, LEVEL_ORDER)

        out[name] = {
            'cv': cv,
            'holdout': ho,
            'seconds': round(time.time() - t0, 1),
        }
        log(f'{tag}/{name} cv={cv["macro_f1"]["mean"]:.4f}±{cv["macro_f1"]["std"]:.4f} '
            f'holdout={ho["macro_f1"]:.4f} extinct_recall_cv={cv["per_class_recall"]["extinct"]:.4f} '
            f'({out[name]["seconds"]}s)')
    return out


def parse_splits():
    payload = json.loads(SPLITS_PATH.read_text(encoding='utf-8'))
    folds = [
        (np.array(f['train_idx'], dtype=int), np.array(f['test_idx'], dtype=int))
        for f in payload['folds']
    ]
    holdout_tr = np.array(payload['holdout']['train_idx'], dtype=int)
    holdout_te = np.array(payload['holdout']['test_idx'], dtype=int)
    return payload, folds, holdout_tr, holdout_te


def main() -> None:
    started = time.time()

    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    features = [f for f in meta['features'] if f in df.columns]
    trainable = df[df['trainable'] == 1].reset_index(drop=True)
    for col in CATEGORICAL:
        if col not in features:
            features.append(col)

    X_full = prepare_frame(trainable, features)
    y5 = trainable['target_5'].map({c: i for i, c in enumerate(LEVEL_ORDER)}).astype(int).values

    groups_payload = load_feature_groups(features)
    X_weak, weak_cols, dropped = build_weak_matrix(X_full, groups_payload)
    log(f'弱信息特征 {X_weak.shape[1]} 列，丢弃常量 {dropped}')

    payload, folds, holdout_tr, holdout_te = parse_splits()
    if payload.get('n_rows') != len(trainable):
        raise ValueError('折索引行数与当前数据集不一致')

    core = model_factories(len(LEVEL_ORDER), include_torch=True)
    weak_factories = {name: core[name] for name in CORE_MODELS if name in core}
    weak_factories.update(extra_factories())
    missing = [m for m in CORE_MODELS if m not in weak_factories]
    if missing:
        log(f'跳过不可用模型: {missing}')

    log('=== 弱信息 9 特征 ===')
    weak_results = evaluate_set(
        X_weak, y5, folds, holdout_tr, holdout_te, weak_factories, 'weak')

    log('=== dummy 下界 ===')
    dummy_results = evaluate_set(
        X_weak, y5, folds, holdout_tr, holdout_te, dummy_factories(), 'dummy')

    # 上界参照：CV 直接引用 metrics.json；留出集原产物只存了最优模型，这里补算 5 个
    old = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    full_cv_ref = old['cv_comparison']['target_5']['group_family_5fold']
    log('=== full 59 特征（只补算留出集，CV 引用 metrics.json）===')
    full_factories = {name: core[name] for name in CORE_MODELS if name in core}
    full_holdout = {}
    for name, factory in full_factories.items():
        t0 = time.time()
        model = factory()
        model.fit(X_full.iloc[holdout_tr].reset_index(drop=True), y5[holdout_tr])
        proba = model.predict_proba(X_full.iloc[holdout_te].reset_index(drop=True))
        pred = np.argmax(proba, axis=1)
        ho = compute_metrics(y5[holdout_te], pred, proba, LEVEL_ORDER)
        full_holdout[name] = {
            'holdout': ho,
            'cv': full_cv_ref.get(name),
            'seconds': round(time.time() - t0, 1),
        }
        log(f'full/{name} holdout={ho["macro_f1"]:.4f} ({full_holdout[name]["seconds"]}s)')

    # ---- 汇总比较 ----
    weak_cv = {m: r['cv']['macro_f1']['mean'] for m, r in weak_results.items()}
    best_model = max(weak_cv, key=weak_cv.get)
    ranked = sorted(weak_cv.items(), key=lambda kv: kv[1], reverse=True)
    non_dummy = [v for m, v in weak_cv.items() if not m.startswith('Dummy-')]
    summary = {
        'baseline_cv_macro_f1': BASELINE_CV,
        'baseline_source': 'metrics.json → ablation[weak_information].macro_f1_mean',
        'best_weak_model': best_model,
        'best_weak_cv_macro_f1': weak_cv[best_model],
        'best_increment_vs_baseline': weak_cv[best_model] - BASELINE_CV,
        'weak_cv_range': max(non_dummy) - min(non_dummy),
        'weak_cv_min': min(non_dummy),
        'weak_cv_max': max(non_dummy),
        'ranking_cv': [{'model': m, 'cv_macro_f1': v,
                        'delta_vs_baseline': v - BASELINE_CV} for m, v in ranked],
        'full_reference_holdout': {
            m: r['holdout']['macro_f1'] for m, r in full_holdout.items()},
        'full_reference_cv': {
            m: r['cv']['macro_f1']['mean'] for m, r in full_holdout.items() if r['cv']},
        'reproduction_check': {
            'weak_LightGBM_cv': weak_results['LightGBM']['cv']['macro_f1']['mean'],
            'expected': BASELINE_CV,
            'delta': weak_results['LightGBM']['cv']['macro_f1']['mean'] - BASELINE_CV,
            'full_LightGBM_holdout': full_holdout['LightGBM']['holdout']['macro_f1'],
            'expected_full_holdout': old['holdout']['macro_f1'],
        },
    }

    out = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'ml/weak_models_comparison.py',
        'purpose': '弱信息 9 特征的模型横向对比（不覆盖主实验）',
        'data': {
            'dataset': 'ml/endangerment_dataset.csv',
            'n_trainable': int(len(trainable)),
            'label': 'target_5',
            'class_counts': {c: int((y5 == i).sum()) for i, c in enumerate(LEVEL_ORDER)},
        },
        'split': {
            'splits_file': 'ml/reports/splits_group_family_5fold.json',
            'scheme': 'group_family_5fold',
            'seed': SEED,
            'holdout_train': int(len(holdout_tr)),
            'holdout_test': int(len(holdout_te)),
        },
        'pipeline': 'train_models.SklearnRiskModel（中位数填充 + StandardScaler；类别 unknown + OneHot）',
        'features': {
            'weak': list(X_weak.columns),
            'weak_n': int(X_weak.shape[1]),
            'weak_zero_variance_dropped': dropped,
            'full_n': int(X_full.shape[1]),
        },
        'models': {
            'weak': weak_results,
            'dummy': dummy_results,
            'full_reference': full_holdout,
        },
        'summary': summary,
        'notes': [
            'dummy 与弱信息模型使用同一折索引、同一预处理管线',
            'full 的 CV 数值引用 metrics.json 的 cv_comparison.target_5.group_family_5fold',
            'full 的留出集数值为本脚本补算（原产物只保存了最优模型的留出指标）',
            '所有模型均未调参；KNN 使用 sklearn 默认 k=5（distance 加权），CatBoost 固定 500 轮',
        ],
        'runtime_seconds': round(time.time() - started, 1),
    }
    dump_json(OUT_JSON, out)

    rows = []
    for group, results in (('weak', weak_results), ('dummy', dummy_results)):
        for name, r in results.items():
            rows.append({
                'group': group,
                'model': name,
                'cv_macro_f1_mean': r['cv']['macro_f1']['mean'],
                'cv_macro_f1_std': r['cv']['macro_f1']['std'],
                'cv_accuracy_mean': r['cv']['accuracy']['mean'],
                'cv_extinct_recall': r['cv']['per_class_recall']['extinct'],
                'holdout_macro_f1': r['holdout']['macro_f1'],
                'holdout_accuracy': r['holdout']['accuracy'],
                'holdout_extinct_recall': r['holdout']['per_class']['extinct']['recall'],
                'delta_vs_0.2878': r['cv']['macro_f1']['mean'] - BASELINE_CV,
            })
    for name, r in full_holdout.items():
        rows.append({
            'group': 'full_reference',
            'model': name,
            'cv_macro_f1_mean': r['cv']['macro_f1']['mean'] if r['cv'] else np.nan,
            'cv_macro_f1_std': r['cv']['macro_f1']['std'] if r['cv'] else np.nan,
            'cv_accuracy_mean': r['cv']['accuracy']['mean'] if r['cv'] else np.nan,
            'cv_extinct_recall': r['cv']['per_class_recall']['extinct'] if r['cv'] else np.nan,
            'holdout_macro_f1': r['holdout']['macro_f1'],
            'holdout_accuracy': r['holdout']['accuracy'],
            'holdout_extinct_recall': r['holdout']['per_class']['extinct']['recall'],
            'delta_vs_0.2878': (r['cv']['macro_f1']['mean'] - BASELINE_CV) if r['cv'] else np.nan,
        })
    frame = pd.DataFrame(rows)
    tmp = OUT_CSV.with_name(OUT_CSV.name + '.tmp')
    frame.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(OUT_CSV)

    log(f'完成，用时 {time.time() - started:.0f}s → {OUT_JSON.name}, {OUT_CSV.name}')
    log(f'最优弱信息模型：{best_model} cv={weak_cv[best_model]:.4f} '
        f'（相对 0.2878 {weak_cv[best_model] - BASELINE_CV:+.4f}）')


if __name__ == '__main__':
    main()
