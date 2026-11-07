"""
留出比例对照实验：把留出集从「按语系分组、约 65:35」改为「约 75:25」。

只新增产物，不覆盖任何既有文件：
  ml/reports/holdout25_experiment.json
  ml/reports/holdout25_results.csv

复算：
  python ml/holdout25_experiment.py

切分口径（为什么不用 GroupShuffleSplit(test_size=0.25)）：
  现有 65:35 来自 GroupShuffleSplit(test_size=0.2, random_state=42)。该参数作用于
  「语系组」而非「行」：31 个语系里 0.2 只抽到 7 个语系，却占 1,204/3,462 = 34.8% 的行。
  直接改成 0.25 会抽到 8 个语系、占 36.5% 的行，仍然不是 25%。
  因此这里改为：按语系整组、随机顺序（seed=42）、累计行数首次达到 25% 即停止。
  结果：留出 873/3,462 = 25.22%，训练 2,589/3,462 = 74.78%，语系不跨集。

CV 口径：
  cv_all       —— 在全部 3,462 行上做 StratifiedGroupKFold(5, shuffle, seed=42)，
                  直接复用 ml/reports/splits_group_family_5fold.json，并校验与重建一致。
                  这一口径与留出比例无关，用来验证「改留出比例不会连带改变 CV 数字」。
  cv_train75   —— 只在 75% 训练子集（2,589 行）内部重做 StratifiedGroupKFold(5, seed=42)。
                  这才是「新切分下」的 CV 数字。
  留出集评估   —— 在 75% 训练子集上重拟合，预测 25% 留出子集。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from train_models import (  # noqa: E402
    CATEGORICAL, DATASET_PATH, META_PATH, LEVEL_ORDER, REPORT_DIR, SEED,
    build_weak_matrix, compute_metrics, dump_json, load_feature_groups,
    log, model_factories, prepare_frame, run_cv, summarize_folds,
)

HOLDOUT_FRAC = 0.25
SPLITS_PATH = REPORT_DIR / 'splits_group_family_5fold.json'
METRICS_PATH = REPORT_DIR / 'metrics.json'
OUT_JSON = REPORT_DIR / 'holdout25_experiment.json'
OUT_CSV = REPORT_DIR / 'holdout25_results.csv'
MODEL_ORDER = ['LightGBM', 'RandomForest', 'XGBoost', 'LogisticRegression', 'MLP-Embedding']


def greedy_group_holdout(groups: pd.Series, frac: float, seed: int):
    """按语系整组随机抽取，累计行数首次达到目标比例即停止。"""
    sizes = groups.value_counts()
    order = list(sizes.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    target = frac * len(groups)
    chosen, acc = [], 0
    for name in order:
        if acc >= target:
            break
        chosen.append(name)
        acc += int(sizes[name])
    mask = groups.isin(chosen).to_numpy()
    train_idx = np.where(~mask)[0]
    test_idx = np.where(mask)[0]
    return train_idx, test_idx, chosen


def class_counts(y: np.ndarray, idx: np.ndarray) -> dict:
    values, counts = np.unique(y[idx], return_counts=True)
    return {LEVEL_ORDER[int(v)]: int(c) for v, c in zip(values, counts)}


def main() -> None:
    started = time.time()

    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    features = [f for f in meta['features'] if f in df.columns]

    trainable = df[df['trainable'] == 1].reset_index(drop=True)
    for col in CATEGORICAL:
        if col not in features:
            features.append(col)

    X_all = prepare_frame(trainable, features)
    y5 = trainable['target_5'].map({c: i for i, c in enumerate(LEVEL_ORDER)}).astype(int).values
    family = trainable['family_top'].fillna('unknown').astype(str)

    train_idx, test_idx, chosen_families = greedy_group_holdout(family, HOLDOUT_FRAC, SEED)
    overlap = set(family.iloc[train_idx]) & set(family.iloc[test_idx])
    if overlap:
        raise RuntimeError(f'语系跨训练/留出集泄漏: {sorted(overlap)}')
    log(f'75:25 留出：训练 {len(train_idx)} / 留出 {len(test_idx)} '
        f'（留出占 {len(test_idx) / len(trainable):.4%}，语系 {len(chosen_families)} 个）')

    # ---- 既有 5 折划分：读取并校验，确保 cv_all 与主实验完全一致 ----
    payload = json.loads(SPLITS_PATH.read_text(encoding='utf-8'))
    if payload.get('n_rows') != len(trainable):
        raise ValueError(f'落盘划分行数 {payload.get("n_rows")} 与当前 {len(trainable)} 不一致')
    cv_all_splits = [
        (np.array(f['train_idx'], dtype=int), np.array(f['test_idx'], dtype=int))
        for f in payload['folds']
    ]
    rebuilt = list(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED).split(X_all, y5, groups=family))
    for i, ((a, b), (c, d)) in enumerate(zip(cv_all_splits, rebuilt), start=1):
        if not (np.array_equal(np.sort(a), np.sort(c)) and np.array_equal(np.sort(b), np.sort(d))):
            raise RuntimeError(f'落盘 5 折与重建不一致（fold {i}），请先核对 splits 文件')

    X_train_all = X_all.iloc[train_idx].reset_index(drop=True)
    y_train = y5[train_idx]
    family_train = family.iloc[train_idx].reset_index(drop=True)
    cv_train_splits = list(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED).split(
            X_train_all, y_train, groups=family_train))

    groups_payload = load_feature_groups(features)
    X_weak, weak_cols, dropped = build_weak_matrix(X_all, groups_payload)
    log(f'弱信息特征 {len(X_weak.columns)} 列（基础 {len(weak_cols)}，丢弃常量 {dropped}）')

    feature_sets = {
        'full': {'X': X_all, 'n_features': int(X_all.shape[1])},
        'weak_information': {'X': X_weak, 'n_features': int(X_weak.shape[1])},
    }
    factories = model_factories(len(LEVEL_ORDER), include_torch=True)
    models = [m for m in MODEL_ORDER if m in factories]
    if len(models) < len(MODEL_ORDER):
        log(f'缺少模型: {set(MODEL_ORDER) - set(models)}（torch 不可用则为 MLP）')

    results: dict = {}
    csv_rows = []

    for set_name, spec in feature_sets.items():
        X = spec['X']
        X_train = X.iloc[train_idx].reset_index(drop=True)
        results[set_name] = {
            'n_features': spec['n_features'],
            'feature_columns': list(X.columns),
            'models': {},
        }
        for model_name in models:
            factory = factories[model_name]
            t0 = time.time()

            rows_all, _, _ = run_cv(
                X, y5, cv_all_splits, factory, LEVEL_ORDER,
                'target_5', 'group_family_5fold', model_name, verbose=False)
            cv_all = summarize_folds(rows_all, LEVEL_ORDER)

            rows_tr, _, _ = run_cv(
                X_train, y_train, cv_train_splits, factory, LEVEL_ORDER,
                'target_5', 'group_family_5fold_train75', model_name, verbose=False)
            cv_train = summarize_folds(rows_tr, LEVEL_ORDER)

            model = factory()
            model.fit(X_train, y_train)
            proba = model.predict_proba(X.iloc[test_idx].reset_index(drop=True))
            pred = np.argmax(proba, axis=1)
            holdout = compute_metrics(y5[test_idx], pred, proba, LEVEL_ORDER)

            results[set_name]['models'][model_name] = {
                'cv_all_data': cv_all,
                'cv_train_only_75': cv_train,
                'holdout': holdout,
                'seconds': round(time.time() - t0, 1),
            }
            csv_rows.append({
                'feature_set': set_name,
                'n_features': spec['n_features'],
                'model': model_name,
                'cv_all_macro_f1_mean': cv_all['macro_f1']['mean'],
                'cv_all_macro_f1_std': cv_all['macro_f1']['std'],
                'cv_train75_macro_f1_mean': cv_train['macro_f1']['mean'],
                'cv_train75_macro_f1_std': cv_train['macro_f1']['std'],
                'holdout_macro_f1': holdout['macro_f1'],
                'holdout_accuracy': holdout['accuracy'],
                **{f'holdout_recall_{c}': holdout['per_class'][c]['recall'] for c in LEVEL_ORDER},
                'holdout_support_extinct': holdout['per_class']['extinct']['support'],
                'seconds': round(time.time() - t0, 1),
            })
            log(f'{set_name}/{model_name} '
                f'cv_all={cv_all["macro_f1"]["mean"]:.4f}±{cv_all["macro_f1"]["std"]:.4f} '
                f'cv_train75={cv_train["macro_f1"]["mean"]:.4f}±{cv_train["macro_f1"]["std"]:.4f} '
                f'holdout={holdout["macro_f1"]:.4f} acc={holdout["accuracy"]:.4f}')

    # ---- 与既有 65:35 结果对照（只读，不修改 metrics.json）----
    reference = {}
    if METRICS_PATH.exists():
        old = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
        old_weak = old.get('weak_ablation', {})
        reference = {
            'source': 'ml/reports/metrics.json（既有 65:35 结果，只读）',
            'full_holdout_macro_f1': old.get('holdout', {}).get('macro_f1'),
            'full_holdout_accuracy': old.get('holdout', {}).get('accuracy'),
            'full_cv_macro_f1_mean': next(
                (r['macro_f1_mean'] for r in old.get('ablation', []) if r.get('ablation') == 'full'),
                None),
            'weak_holdout_macro_f1': old_weak.get('holdout', {}).get('macro_f1'),
            'weak_holdout_accuracy': old_weak.get('holdout', {}).get('accuracy'),
            'weak_cv_macro_f1_mean': old_weak.get('cv', {}).get('macro_f1', {}).get('mean'),
            'best_model': old.get('best_model', {}).get('name'),
        }
        reference['holdout_gap_65_35'] = (
            None if reference['full_holdout_macro_f1'] is None
            or reference['weak_holdout_macro_f1'] is None
            else reference['full_holdout_macro_f1'] - reference['weak_holdout_macro_f1']
        )
        reference['cv_gap_65_35'] = (
            None if reference['full_cv_macro_f1_mean'] is None
            or reference['weak_cv_macro_f1_mean'] is None
            else reference['full_cv_macro_f1_mean'] - reference['weak_cv_macro_f1_mean']
        )

    # 同一模型（LightGBM）下的新差距，便于与旧差距比较
    comparison = {}
    if 'LightGBM' in models:
        full_ho = results['full']['models']['LightGBM']['holdout']['macro_f1']
        weak_ho = results['weak_information']['models']['LightGBM']['holdout']['macro_f1']
        full_cv = results['full']['models']['LightGBM']['cv_all_data']['macro_f1']['mean']
        weak_cv = results['weak_information']['models']['LightGBM']['cv_all_data']['macro_f1']['mean']
        full_cv75 = results['full']['models']['LightGBM']['cv_train_only_75']['macro_f1']['mean']
        weak_cv75 = results['weak_information']['models']['LightGBM']['cv_train_only_75']['macro_f1']['mean']
        comparison = {
            'model': 'LightGBM',
            'holdout_gap_75_25': full_ho - weak_ho,
            'cv_all_gap_75_25': full_cv - weak_cv,
            'cv_train75_gap_75_25': full_cv75 - weak_cv75,
            'ratio_increment_full_holdout': (
                None if reference.get('full_holdout_macro_f1') is None
                else full_ho - reference['full_holdout_macro_f1']),
            'ratio_increment_weak_holdout': (
                None if reference.get('weak_holdout_macro_f1') is None
                else weak_ho - reference['weak_holdout_macro_f1']),
        }

    payload_out = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'ml/holdout25_experiment.py',
        'purpose': '留出比例 65:35 → 75:25 的对照实验；不与既有产物混用',
        'data': {
            'dataset': 'ml/endangerment_dataset.csv',
            'meta': 'ml/dataset_meta.json',
            'n_trainable': int(len(trainable)),
            'label': 'target_5（由 legacy_endangerment_level 映射）',
            'class_counts_all': class_counts(y5, np.arange(len(y5))),
        },
        'split': {
            'method': '按 family_top 整组、随机排列（seed=42）、累计行数首次达到 25% 即停',
            'target_fraction': HOLDOUT_FRAC,
            'seed': SEED,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'actual_test_fraction': round(len(test_idx) / len(trainable), 6),
            'holdout_class_counts': class_counts(y5, test_idx),
            'train_class_counts': class_counts(y5, train_idx),
            'holdout_families': sorted(chosen_families),
            'train_families': sorted(set(family.iloc[train_idx])),
            'holdout_test_idx': test_idx.tolist(),
            'holdout_train_idx': train_idx.tolist(),
            'note': 'GroupShuffleSplit(test_size=0.25) 会得到 36.48% 的行，故未采用',
        },
        'cv_settings': {
            'cv_all_data': {
                'scheme': 'group_family_5fold',
                'splits_file': 'ml/reports/splits_group_family_5fold.json',
                'rows': int(len(trainable)),
                'note': '与主实验同一份折索引，应逐位复现；与留出比例无关',
            },
            'cv_train_only_75': {
                'scheme': 'group_family_5fold_train75',
                'splitter': 'StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)',
                'rows': int(len(train_idx)),
                'note': '只在 75% 训练子集内部切分，留出集不参与',
            },
            'holdout': {
                'fit_rows': int(len(train_idx)),
                'eval_rows': int(len(test_idx)),
                'note': '在 75% 训练子集上重拟合，预测 25% 留出子集',
            },
        },
        'models': models,
        'results': results,
        'reference_65_35': reference,
        'comparison_75_25': comparison,
        'comparability': '新旧数字因留出比例与留出集合不同，不可直接比较；只能比较同一比例内部的 full/weak 差距',
        'runtime_seconds': round(time.time() - started, 1),
    }
    dump_json(OUT_JSON, payload_out)

    frame = pd.DataFrame(csv_rows)
    tmp = OUT_CSV.with_name(OUT_CSV.name + '.tmp')
    frame.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(OUT_CSV)

    log(f'完成，用时 {time.time() - started:.0f}s → {OUT_JSON.name}, {OUT_CSV.name}')
    if comparison:
        log(f'LightGBM：75:25 留出 full−weak 差距 {comparison["holdout_gap_75_25"]:.4f}；'
            f'65:35 旧差距 {reference.get("holdout_gap_65_35")}')


if __name__ == '__main__':
    main()
