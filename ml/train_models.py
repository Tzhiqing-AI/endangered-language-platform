"""
濒危语言风险预测：模型训练、评估与可解释性分析。

输入：
  ml/endangerment_dataset.csv
  ml/dataset_meta.json

输出：
  ml/artifacts/risk_model.joblib            部署用模型（预处理器 + 估计器）
  ml/artifacts/risk_service_meta.json       特征标签、等级映射、风险权重
  ml/artifacts/language_risk_predictions.csv 可训练语言的风险评分（训练集重代入，仅对照）
  ml/artifacts/language_risk_oof.csv        可训练语言的 OOF 评分（页面展示口径）
  ml/reports/metrics.json                   完整评估结果
  ml/reports/cv_results.csv                 交叉验证明细
  ml/reports/ablation.csv                   消融实验
  ml/reports/permutation_importance.csv     置换重要性
  ml/reports/*.png                          对比图 / 混淆矩阵 / 重要性 / 消融图

用法：
  python ml/train_models.py
  python ml/train_models.py --no-torch
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, make_scorer,
)
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    HAS_TORCH = False

try:
    import shap
    HAS_SHAP = True
except Exception:  # noqa: BLE001
    HAS_SHAP = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except Exception:  # noqa: BLE001
    HAS_TABNET = False


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'endangerment_dataset.csv'
META_PATH = BASE_DIR / 'dataset_meta.json'
FEATURE_GROUPS_PATH = BASE_DIR / 'feature_groups.json'
ARTIFACT_DIR = BASE_DIR / 'artifacts'
REPORT_DIR = BASE_DIR / 'reports'
SPLITS_PATH = REPORT_DIR / 'splits_group_family_5fold.json'
SEED = 42

WEAK_FEATURE_SET_NAME = 'weak_information'
WEAK_DENSITY_FEATURE = 'neighbor_density_100km'
WEAK_DENSITY_RADIUS_KM = 100.0

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LEVEL_ORDER = ['unsafe', 'danger', 'endangered', 'seriously_endangered', 'extinct']
LEVEL_LABELS_ZH = {
    'unsafe': '不安全',
    'danger': '危险',
    'endangered': '濒危',
    'seriously_endangered': '垂危',
    'extinct': '灭绝',
}
RISK_WEIGHTS_5 = np.array([20.0, 40.0, 60.0, 80.0, 100.0])

BINARY_ORDER = ['low_risk', 'high_risk']
BINARY_LABELS_ZH = {'low_risk': '低风险', 'high_risk': '高风险'}

CATEGORICAL = [
    'family_top', 'country', 'region',
    'speaker_trend', 'transmission', 'domain_of_use',
]

FEATURE_LABELS = {
    'speakers': '使用人数',
    'log_speakers': '使用人数（对数）',
    'has_speakers': '是否有使用者数据',
    'notes_length': '备注信息量',
    'has_media': '多媒体资料情况',
    'family_depth': '语系层级深度',
    'alternate_name_count': '别名数量',
    'classification_depth': '分类深度',
    'has_iso': 'ISO 代码可用',
    'family_top': '语系',
    'country': '国家',
    'region': '地理区域',
    'speaker_trend': '使用者趋势',
    'transmission': '代际传承',
    'domain_of_use': '使用领域',
    'sample_count': '语言样本数量',
    'sample_type_count': '样本类型数',
    'bibliography_count': '文献数量',
    'source_count': '文献来源数量',
    'cited_count': '被引文献数量',
    'source_type_count': '文献来源类型数',
    'activity_count': '语言活动数量',
    'activity_type_count': '活动类型数',
    'orthography_count': '正字法记录数',
    'script_count': '文字系统数量',
    'documentation_count': '文献记录数',
    'max_documentation_index': '文献记录完整度',
    'resource_count': '语言资源数量',
    'media_count': '多媒体资料数量',
    'context_count': '使用场景记录数',
    'government_support_flag': '政府支持记录',
    'institutional_support_flag': '机构支持记录',
    'speaker_attitude_flag': '使用者态度记录',
    'location_count': '地理位置记录数',
    'country_count': '分布国家数',
    'vitality_count': '活力记录数',
    'speaker_records': '使用者记录数',
    'max_speaker_number': '最大使用者数',
    'max_ethnic_population': '族群人口',
    'max_semi_speakers': '半流利使用者',
    'max_children': '儿童使用者',
    'max_young_adults': '青年使用者',
    'max_older_adults': '中年使用者',
    'max_elders': '老年使用者',
    'log_ethnic_population': '族群人口（对数）',
    'log_speaker_number_max': '最大使用者数（对数）',
    'log_children': '儿童使用者（对数）',
    'log_young_adults': '青年使用者（对数）',
    'log_older_adults': '中年使用者（对数）',
    'log_elders': '老年使用者（对数）',
    'latitude': '纬度',
    'longitude': '经度',
    'has_location': '是否有坐标',
    'sample_A': '样本类型 A',
    'sample_D': '样本类型 D',
    'sample_G': '样本类型 G',
    'sample_I': '样本类型 I',
    'sample_L': '样本类型 L',
    'sample_V': '样本类型 V',
}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def dump_json(path: Path, payload: Dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding='utf-8',
    )
    os.replace(tmp, path)  # 原子替换，避免读到半成品 JSON


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    table = {c: w for c, w in zip(classes, weights)}
    return np.array([table[v] for v in y], dtype=np.float32)


def prepare_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """统一数据类型：数值列为 float，类别列为去空白的字符串。"""
    X = frame[list(features)].copy()
    for col in features:
        if col in CATEGORICAL:
            series = X[col].astype('object').where(X[col].notna(), 'unknown')
            X[col] = series.astype(str).str.strip().replace({'': 'unknown', 'nan': 'unknown'})
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype('float64')
    return X


def numeric_columns(features: Sequence[str]) -> List[str]:
    return [c for c in features if c not in CATEGORICAL]


def build_preprocessor(features: Sequence[str]) -> ColumnTransformer:
    num = numeric_columns(features)
    cat = [c for c in features if c in CATEGORICAL]
    transformers = []
    if num:
        transformers.append((
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            num,
        ))
    if cat:
        transformers.append((
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]),
            cat,
        ))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)


def build_ordinal_preprocessor(features: Sequence[str]) -> ColumnTransformer:
    num = numeric_columns(features)
    cat = [c for c in features if c in CATEGORICAL]
    transformers = []
    if num:
        transformers.append((
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            num,
        ))
    if cat:
        transformers.append((
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('ordinal', OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1,
                )),
            ]),
            cat,
        ))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)


# ---------------------------------------------------------------------------
# 模型封装
# ---------------------------------------------------------------------------
class SklearnRiskModel:
    """统一的 sklearn 模型包装，内部完成缺失值填充与编码。"""

    def __init__(self, estimator, use_sample_weight: bool = False):
        self.estimator = estimator
        self.use_sample_weight = use_sample_weight

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_columns = list(X.columns)
        self.pre = build_preprocessor(self.feature_columns)
        Xt = self.pre.fit_transform(X[self.feature_columns])
        kwargs = {}
        if self.use_sample_weight:
            kwargs['sample_weight'] = balanced_sample_weights(y)
        self.estimator.fit(Xt, y, **kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X[self.feature_columns])
        return self.estimator.predict_proba(Xt)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


if HAS_TORCH:

    class EntityEmbeddingMLP(nn.Module):
        """类别特征实体嵌入 + 数值特征拼接的轻量 MLP。"""

        def __init__(self, cardinalities, n_numeric, n_classes,
                     emb_dim: int = 8, hidden=(128, 64), dropout: float = 0.25):
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.Embedding(card, min(emb_dim, max(2, card // 2)))
                for card in cardinalities
            ])
            emb_total = sum(emb.embedding_dim for emb in self.embeddings)
            layers = []
            in_dim = emb_total + n_numeric
            for h in hidden:
                layers += [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_dim = h
            layers.append(nn.Linear(in_dim, n_classes))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num, x_cat):
            emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
            x = torch.cat(emb + [x_num], dim=1)
            return self.mlp(x)


class TorchRiskModel:
    """PyTorch 实体嵌入 MLP 的 sklearn 风格封装。"""

    def __init__(self, n_classes: int, epochs: int = 45, patience: int = 8,
                 batch_size: int = 128, lr: float = 1e-3, dropout: float = 0.25):
        self.n_classes = n_classes
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.net = None

    def _build_preprocessor(self):
        self.feature_columns = list(self.feature_columns)
        self.pre = build_ordinal_preprocessor(self.feature_columns)
        cat_cols = [c for c in self.feature_columns if c in CATEGORICAL]
        self.cat_cols = cat_cols
        self.num_cols = numeric_columns(self.feature_columns)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.feature_columns = list(X.columns)
        self._build_preprocessor()
        Xt = self.pre.fit_transform(X[self.feature_columns])

        num_cols = self.num_cols
        cat_cols = self.cat_cols
        output_names = self.pre.get_feature_names_out()
        self.n_num_out = sum(1 for name in output_names if name.startswith('num__'))
        n_num_out = self.n_num_out
        x_num_all = np.asarray(Xt[:, :n_num_out], dtype=np.float32)
        if cat_cols:
            raw_cat = np.asarray(Xt[:, n_num_out:])
            x_cat_all = np.where(raw_cat < 0, 0, raw_cat + 1).astype(np.int64)
        else:
            x_cat_all = np.zeros((len(Xt), 0), dtype=np.int64)

        # 类别基数：OrdinalEncoder 的类别数 + 1（0 号留给未知值）
        cardinalities = [len(cats) + 1 for cats in self.pre.named_transformers_['cat']
                         .named_steps['ordinal'].categories_] if cat_cols else []

        try:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=0.12, stratify=y, random_state=SEED,
            )
        except ValueError:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=0.12, random_state=SEED,
            )

        n_numeric = len(num_cols)
        self.net = EntityEmbeddingMLP(
            cardinalities, n_numeric, self.n_classes, dropout=self.dropout,
        )

        weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_table = {c: w for c, w in zip(np.unique(y), weights)}
        class_weights = torch.tensor(
            [weight_table.get(i, 1.0) for i in range(self.n_classes)],
            dtype=torch.float32,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        train_ds = TensorDataset(
            torch.tensor(x_num_all[tr_idx]), torch.tensor(x_cat_all[tr_idx]),
            torch.tensor(y[tr_idx].astype(np.int64)),
        )
        loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            drop_last=len(tr_idx) > self.batch_size,
        )

        x_num_va = torch.tensor(x_num_all[va_idx])
        x_cat_va = torch.tensor(x_cat_all[va_idx])
        y_va = y[va_idx]

        best_state = copy.deepcopy(self.net.state_dict())
        best_score = -1.0
        stale = 0

        for epoch in range(self.epochs):
            self.net.train()
            for xb_num, xb_cat, yb in loader:
                optimizer.zero_grad()
                logits = self.net(xb_num, xb_cat)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                logits = self.net(x_num_va, x_cat_va)
                pred = torch.argmax(logits, dim=1).numpy()
            score = f1_score(y_va, pred, average='macro', zero_division=0)

            if score > best_score + 1e-4:
                best_score = score
                best_state = copy.deepcopy(self.net.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_val_macro_f1 = float(best_score)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        Xt = self.pre.transform(X[self.feature_columns])
        n_num_out = self.n_num_out
        x_num = torch.tensor(np.asarray(Xt[:, :n_num_out], dtype=np.float32))
        if self.cat_cols:
            raw_cat = np.asarray(Xt[:, n_num_out:])
            x_cat = torch.tensor(np.where(raw_cat < 0, 0, raw_cat + 1).astype(np.int64))
        else:
            x_cat = torch.zeros((len(Xt), 0), dtype=torch.int64)

        outputs = []
        with torch.no_grad():
            for start in range(0, len(Xt), 512):
                logits = self.net(x_num[start:start + 512], x_cat[start:start + 512])
                outputs.append(torch.softmax(logits, dim=1).numpy())
        return np.vstack(outputs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class TabNetRiskModel:
    """PyTorch TabNet 包装。

    类别特征先做 one-hot，再作为数值输入；适合与本项目的中小规模表格数据。
    """

    def __init__(self, n_classes: int, max_epochs: int = 100,
                 patience: int = 20, batch_size: int = 512):
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_columns = list(X.columns)
        self.pre = build_preprocessor(self.feature_columns)
        Xt = self.pre.fit_transform(X[self.feature_columns])

        try:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=0.12, stratify=y, random_state=SEED,
            )
        except ValueError:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=0.12, random_state=SEED,
            )

        self.model = TabNetClassifier(
            n_d=16, n_a=16, n_steps=4, gamma=1.3,
            optimizer_fn=torch.optim.Adam, optimizer_params={'lr': 2e-2},
            scheduler_params={'step_size': 10, 'gamma': 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=SEED, verbose=0, device_name='cpu',
        )
        self.model.fit(
            X_train=Xt[tr_idx], y_train=y[tr_idx],
            eval_set=[(Xt[va_idx], y[va_idx])], eval_name=['valid'],
            eval_metric=['balanced_accuracy'],
            loss_fn=torch.nn.CrossEntropyLoss(),
            weights=1,
            max_epochs=self.max_epochs, patience=self.patience,
            batch_size=self.batch_size, virtual_batch_size=128,
            num_workers=0, drop_last=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.pre.transform(X[self.feature_columns]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def model_factories(n_classes: int, include_torch: bool,
                    include_tabnet: bool = False) -> Dict[str, callable]:
    factories = {
        'LogisticRegression': lambda: SklearnRiskModel(
            LogisticRegression(
                max_iter=3000, C=1.0, class_weight='balanced',
                solver='lbfgs', n_jobs=-1,
            )
        ),
        'RandomForest': lambda: SklearnRiskModel(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                class_weight='balanced_subsample', n_jobs=-1, random_state=SEED,
            )
        ),
        'XGBoost': lambda: SklearnRiskModel(
            XGBClassifier(
                n_estimators=350, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, min_child_weight=1,
                reg_lambda=1.0, tree_method='hist', n_jobs=-1,
                random_state=SEED, eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                verbosity=0,
            ), use_sample_weight=True,
        ),
        'LightGBM': lambda: SklearnRiskModel(
            LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.8, class_weight='balanced',
                n_jobs=-1, random_state=SEED, verbose=-1,
            )
        ),
    }
    if include_torch and HAS_TORCH:
        factories['MLP-Embedding'] = lambda: TorchRiskModel(n_classes)
    if include_tabnet and HAS_TABNET:
        factories['TabNet'] = lambda: TabNetRiskModel(n_classes)
    return factories


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray, class_names: Sequence[str]) -> Dict:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'per_class': {
            name: {
                'precision': float(report[name]['precision']),
                'recall': float(report[name]['recall']),
                'f1': float(report[name]['f1-score']),
                'support': int(report[name]['support']),
            }
            for name in class_names
        },
        'confusion_matrix': cm.tolist(),
    }


def flatten_row(task: str, scheme: str, model_name: str, fold: int,
                metrics: Dict, n_train: int, n_test: int) -> Dict:
    row = {
        'task': task, 'scheme': scheme, 'model': model_name,
        'fold': fold, 'n_train': n_train, 'n_test': n_test,
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'weighted_f1': metrics['weighted_f1'],
        'balanced_accuracy': metrics['balanced_accuracy'],
    }
    for cls, values in metrics['per_class'].items():
        row[f'recall_{cls}'] = values['recall']
        row[f'f1_{cls}'] = values['f1']
    return row


def summarize_folds(rows: Sequence[Dict], class_names: Sequence[str]) -> Dict:
    keys = ['accuracy', 'macro_f1', 'weighted_f1', 'balanced_accuracy']
    summary = {}
    for key in keys:
        values = [r[key] for r in rows]
        summary[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    summary['per_class_recall'] = {
        cls: float(np.mean([r[f'recall_{cls}'] for r in rows])) for cls in class_names
    }
    summary['per_class_f1'] = {
        cls: float(np.mean([r[f'f1_{cls}'] for r in rows])) for cls in class_names
    }
    return summary


def make_splits(X: pd.DataFrame, y: np.ndarray, groups: pd.Series,
                scheme: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    if scheme == 'stratified_5fold':
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(splitter.split(X, y))
    if scheme == 'group_family_5fold':
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(splitter.split(X, y, groups=groups['family']))
    if scheme == 'group_country_5fold':
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(splitter.split(X, y, groups=groups['country']))
    raise ValueError(f'未知划分方案: {scheme}')


def run_cv(X: pd.DataFrame, y: np.ndarray, splits, factory: callable,
           class_names: Sequence[str], task: str, scheme: str,
           model_name: str, verbose: bool = True) -> Tuple[List[Dict], List[Dict], np.ndarray]:
    rows, raw_rows = [], []
    oof_pred = np.full(len(y), -1, dtype=int)
    for fold, (tr, te) in enumerate(splits, start=1):
        model = factory()
        model.fit(X.iloc[tr].reset_index(drop=True), y[tr])
        proba = model.predict_proba(X.iloc[te].reset_index(drop=True))
        pred = np.argmax(proba, axis=1)
        oof_pred[te] = pred
        metrics = compute_metrics(y[te], pred, proba, class_names)
        raw_rows.append(metrics)
        rows.append(flatten_row(task, scheme, model_name, fold, metrics, len(tr), len(te)))
        if verbose:
            log(f'  {task}/{scheme}/{model_name} fold{fold} '
                f'macroF1={metrics["macro_f1"]:.4f} acc={metrics["accuracy"]:.4f}')
    return rows, raw_rows, oof_pred


# ---------------------------------------------------------------------------
# 弱信息消融
# ---------------------------------------------------------------------------
def load_feature_groups(features: Sequence[str]) -> Dict:
    """读取显式特征分组，并校验与数据集特征逐列对齐。"""
    if not FEATURE_GROUPS_PATH.exists():
        raise FileNotFoundError(f'缺少特征分组清单: {FEATURE_GROUPS_PATH}')
    payload = json.loads(FEATURE_GROUPS_PATH.read_text(encoding='utf-8'))
    archive = list(payload.get('archive_features', []))
    weak = list(payload.get('weak_features', []))
    expected = set(features)
    got = set(archive) | set(weak)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    overlap = sorted(set(archive) & set(weak))
    if missing or extra or overlap:
        raise ValueError(
            'feature_groups.json 与数据集特征不一致：'
            f'missing={missing} extra={extra} overlap={overlap}')
    return payload


def compute_neighbor_density(frame: pd.DataFrame, radius_km: float) -> np.ndarray:
    """统计每个语言在半径内的其他可训练语言数，无坐标返回 NaN。"""
    lat = pd.to_numeric(frame['latitude'], errors='coerce').to_numpy(dtype=float)
    lon = pd.to_numeric(frame['longitude'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    density = np.full(len(frame), np.nan)
    if valid.any():
        coords = np.radians(np.column_stack([lat[valid], lon[valid]]))
        distances = haversine_distances(coords) * 6371.0088
        density[valid] = ((distances <= radius_km).sum(axis=1) - 1).astype(float)
    return density


def constant_feature_columns(X: pd.DataFrame) -> List[str]:
    """找出整列取值唯一（含常量/全缺失）的特征。"""
    return [c for c in X.columns if X[c].dropna().nunique() <= 1]


def build_weak_matrix(X_all: pd.DataFrame, groups_payload: Dict):
    weak_cols = [c for c in groups_payload['weak_features'] if c in X_all.columns]
    X_weak = X_all[weak_cols].copy()
    for spec in groups_payload.get('derived_weak_features', []):
        name = spec['name']
        if name != WEAK_DENSITY_FEATURE:
            raise ValueError(f'未实现的派生弱特征: {name}')
        X_weak[name] = compute_neighbor_density(
            X_weak, float(spec.get('radius_km', WEAK_DENSITY_RADIUS_KM)))
    dropped = constant_feature_columns(X_weak)
    X_weak = X_weak.drop(columns=dropped)
    return X_weak, weak_cols, dropped


def summarize_cv(rows: List[Dict], raw_rows: List[Dict], oof_pred: np.ndarray,
                 y: np.ndarray, class_names: Sequence[str]) -> Dict:
    per_class_recall = {}
    for name in class_names:
        values = [r['per_class'][name]['recall'] for r in raw_rows]
        per_class_recall[name] = float(np.mean(values))
    per_class_f1 = {}
    for name in class_names:
        values = [r['per_class'][name]['f1'] for r in raw_rows]
        per_class_f1[name] = float(np.mean(values))
    labels = list(range(len(class_names)))
    return {
        'macro_f1': {
            'mean': float(np.mean([r['macro_f1'] for r in rows])),
            'std': float(np.std([r['macro_f1'] for r in rows])),
        },
        'accuracy_mean': float(np.mean([r['accuracy'] for r in rows])),
        'balanced_accuracy_mean': float(np.mean([r['balanced_accuracy'] for r in rows])),
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix_oof': confusion_matrix(y, oof_pred, labels=labels).tolist(),
    }


def write_splits_payload(trainable: pd.DataFrame, splits5, train_idx: np.ndarray,
                         test_idx: np.ndarray, scheme: str) -> Dict:
    payload = {
        'scheme': scheme,
        'seed': SEED,
        'n_rows': int(len(trainable)),
        'n_splits': len(splits5),
        'folds': [
            {
                'fold': i + 1,
                'train_idx': tr.tolist(),
                'test_idx': te.tolist(),
                'test_code_ids': trainable.iloc[te]['code_id'].astype(int).tolist(),
            }
            for i, (tr, te) in enumerate(splits5)
        ],
        'holdout': {
            'train_idx': train_idx.tolist(),
            'test_idx': test_idx.tolist(),
            'test_code_ids': trainable.iloc[test_idx]['code_id'].astype(int).tolist(),
        },
    }
    dump_json(SPLITS_PATH, payload)
    return payload


def run_weak_ablation(X_all: pd.DataFrame, y5: np.ndarray, splits5, train_idx: np.ndarray,
                      test_idx: np.ndarray, groups_payload: Dict,
                      full_macro_f1: Optional[float]):
    """弱信息消融：只训 LightGBM（附 RandomForest 对照），复用主实验划分。"""
    X_weak, weak_cols, dropped = build_weak_matrix(X_all, groups_payload)
    log(f'弱信息消融：特征 {len(X_weak.columns)}（基础 {len(weak_cols)}，'
        f'剔除常量 {dropped or "无"}）')

    lgb_factory = model_factories(len(LEVEL_ORDER), False)['LightGBM']
    rows, raw_rows, oof_pred = run_cv(
        X_weak, y5, splits5, lgb_factory, LEVEL_ORDER, 'target_5',
        'group_family_5fold', 'LightGBM:weak-information', verbose=False,
    )
    cv_summary = summarize_cv(rows, raw_rows, oof_pred, y5, LEVEL_ORDER)

    rf_factory = model_factories(len(LEVEL_ORDER), False)['RandomForest']
    rf_rows, rf_raw_rows, rf_oof = run_cv(
        X_weak, y5, splits5, rf_factory, LEVEL_ORDER, 'target_5',
        'group_family_5fold', 'RandomForest:weak-information', verbose=False,
    )
    rf_summary = summarize_cv(rf_rows, rf_raw_rows, rf_oof, y5, LEVEL_ORDER)

    # 留出集：与主实验同一组索引
    holdout_model = lgb_factory()
    holdout_model.fit(X_weak.iloc[train_idx].reset_index(drop=True), y5[train_idx])
    proba = holdout_model.predict_proba(
        X_weak.iloc[test_idx].reset_index(drop=True))
    holdout = compute_metrics(y5[test_idx], proba.argmax(axis=1), proba, LEVEL_ORDER)
    plot_confusion(
        np.array(holdout['confusion_matrix']), LEVEL_ORDER,
        REPORT_DIR / 'weak_confusion_matrix_holdout.png',
        'LightGBM · weak-information · hold-out',
    )
    plot_confusion(
        np.array(cv_summary['confusion_matrix_oof']), LEVEL_ORDER,
        REPORT_DIR / 'weak_confusion_matrix_oof.png',
        'LightGBM · weak-information · OOF',
    )

    # 重算置换重要性，观察档案特征缺失后哪些特征顶上来
    perm = permutation_importance(
        holdout_model, X_weak.iloc[test_idx].reset_index(drop=True), y5[test_idx],
        scoring=make_scorer(f1_score, average='macro', zero_division=0),
        n_repeats=8, random_state=SEED, n_jobs=1,
    )
    importance_frame = pd.DataFrame({
        'feature': X_weak.columns,
        'label': [FEATURE_LABELS.get(f, f) for f in X_weak.columns],
        'importance': perm.importances_mean,
        'std': perm.importances_std,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    importance_frame.to_csv(
        REPORT_DIR / 'weak_permutation_importance.csv', index=False,
        encoding='utf-8-sig',
    )
    plot_importance(
        importance_frame, REPORT_DIR / 'weak_permutation_importance.png',
        'Permutation importance · weak-information',
    )

    full_mean = float(full_macro_f1) if full_macro_f1 is not None else float('nan')
    weak_mean = cv_summary['macro_f1']['mean']
    drop = full_mean - weak_mean
    drop_pct = drop / full_mean * 100 if math.isfinite(full_mean) and full_mean else float('nan')

    payload = {
        'feature_set': WEAK_FEATURE_SET_NAME,
        'feature_groups_file': 'ml/feature_groups.json',
        'rows': int(len(y5)),
        'folds': int(len(splits5)),
        'base_weak_features': list(weak_cols),
        'derived_features': [
            {
                'name': WEAK_DENSITY_FEATURE,
                'radius_km': float(spec.get('radius_km', WEAK_DENSITY_RADIUS_KM)),
            }
            for spec in groups_payload.get('derived_weak_features', [])
        ],
        'features_used': list(X_weak.columns),
        'zero_variance_dropped': list(dropped),
        'fill_strategy': (
            '与主实验一致：数值列 NaN 在训练折内用中位数填充，'
            '类别列缺失记为 unknown 并单独编码；未做额外插补'
        ),
        'splits_file': 'ml/reports/splits_group_family_5fold.json',
        'cv': {
            'scheme': 'group_family_5fold',
            'model': 'LightGBM',
            **cv_summary,
        },
        'holdout': holdout,
        'random_forest_comparison': {
            'model': 'RandomForest',
            'macro_f1': rf_summary['macro_f1'],
            'accuracy_mean': rf_summary['accuracy_mean'],
        },
        'relative_to_full': {
            'full_macro_f1_mean': full_mean,
            'weak_macro_f1_mean': weak_mean,
            'absolute_drop': drop,
            'relative_drop_pct': drop_pct,
        },
        'permutation_importance_top': json.loads(
            importance_frame.head(12).to_json(orient='records')),
        'confusion_matrix_files': [
            'ml/reports/weak_confusion_matrix_holdout.png',
            'ml/reports/weak_confusion_matrix_oof.png',
        ],
        'permutation_importance_file': 'ml/reports/weak_permutation_importance.csv',
    }
    dump_json(REPORT_DIR / 'weak_ablation.json', payload)

    row = {
        'ablation': WEAK_FEATURE_SET_NAME,
        'feature_set': WEAK_FEATURE_SET_NAME,
        'feature_count': len(X_weak.columns),
        'macro_f1_mean': weak_mean,
        'macro_f1_std': cv_summary['macro_f1']['std'],
        'accuracy_mean': cv_summary['accuracy_mean'],
        'balanced_accuracy_mean': cv_summary['balanced_accuracy_mean'],
        'holdout_macro_f1': holdout['macro_f1'],
        'macro_f1_drop_vs_full': drop,
        'relative_drop_pct_vs_full': drop_pct,
    }
    log(f'弱信息消融完成：macroF1 {weak_mean:.4f}，相对 full 降 {drop_pct:.1f}%')
    return row, payload, cv_summary


def load_splits_payload(trainable: pd.DataFrame, X_all: pd.DataFrame,
                        y5: np.ndarray, groups: Dict, scheme: str):
    """优先读取落盘划分；缺失时按主实验参数重建并写入。"""
    if SPLITS_PATH.exists():
        payload = json.loads(SPLITS_PATH.read_text(encoding='utf-8'))
        splits5 = [
            (np.array(f['train_idx'], dtype=int), np.array(f['test_idx'], dtype=int))
            for f in payload['folds']
        ]
        train_idx = np.array(payload['holdout']['train_idx'], dtype=int)
        test_idx = np.array(payload['holdout']['test_idx'], dtype=int)
        if payload.get('n_rows') != len(trainable):
            raise ValueError(
                f'落盘划分行数 {payload.get("n_rows")} 与当前可训练行数 '
                f'{len(trainable)} 不一致，请先跑默认全流程')
        return splits5, train_idx, test_idx, payload
    splits5 = make_splits(X_all, y5, groups, scheme)
    holdout_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED,
    )
    train_idx, test_idx = next(holdout_splitter.split(
        X_all, y5, groups=groups['family'],
    ))
    payload = write_splits_payload(trainable, splits5, train_idx, test_idx, scheme)
    return splits5, train_idx, test_idx, payload


def run_weak_only(trainable: pd.DataFrame, X_all: pd.DataFrame, y5: np.ndarray,
                  groups: Dict, scheme: str, features: Sequence[str]) -> None:
    """--feature-set weak：只跑弱信息消融，并把结果并入既有 metrics.json。"""
    groups_payload = load_feature_groups(features)
    splits5, train_idx, test_idx, _ = load_splits_payload(
        trainable, X_all, y5, groups, scheme,
    )
    metrics_path = REPORT_DIR / 'metrics.json'
    full_macro = None
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding='utf-8'))
        for row in existing.get('ablation', []):
            if row.get('ablation') == 'full':
                full_macro = float(row['macro_f1_mean'])
                break
    row, payload, _ = run_weak_ablation(
        X_all, y5, splits5, train_idx, test_idx, groups_payload, full_macro,
    )
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding='utf-8'))
        ablation = [
            r for r in existing.get('ablation', [])
            if r.get('ablation') != WEAK_FEATURE_SET_NAME
        ]
        ablation.append(row)
        ablation.sort(key=lambda r: r['macro_f1_mean'], reverse=True)
        existing['ablation'] = ablation
        existing['weak_ablation'] = payload
        dump_json(metrics_path, existing)
        log('已把弱信息消融并入 ml/reports/metrics.json')
    else:
        log('metrics.json 不存在，仅写出 ml/reports/weak_ablation.json')


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_model_comparison(comparison: Dict, path: Path) -> None:
    tasks = list(comparison.keys())
    schemes = []
    for task in tasks:
        for scheme in comparison[task]:
            if scheme not in schemes:
                schemes.append(scheme)
    scheme_labels = {
        'stratified_5fold': 'Stratified 5-fold',
        'group_family_5fold': 'Grouped by family',
        'group_country_5fold': 'Grouped by country',
    }
    width = max(4.2 * len(schemes), 6.0)
    height = max(4.0 * len(tasks), 4.0)
    fig, axes = plt.subplots(len(tasks), len(schemes),
                             figsize=(width, height), squeeze=False)
    for ti, task in enumerate(tasks):
        for si, scheme in enumerate(schemes):
            ax = axes[ti][si]
            entries = comparison[task].get(scheme, {})
            if not entries:
                ax.axis('off')
                continue
            names = list(entries.keys())
            means = [entries[n]['macro_f1']['mean'] for n in names]
            stds = [entries[n]['macro_f1']['std'] for n in names]
            colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
            bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Macro F1')
            ax.set_title(f'{task} · {scheme_labels.get(scheme, scheme)}', fontsize=9)
            ax.grid(axis='y', alpha=0.25)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                        f'{mean:.3f}', ha='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: Sequence[str], path: Path,
                   title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title, fontsize=10)
    max_val = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center',
                    color='white' if cm[i, j] > max_val * 0.55 else '#0f172a', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_importance(frame: pd.DataFrame, path: Path, title: str) -> None:
    top = frame.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.barh(top['label'], top['importance'], color='#0f766e')
    ax.set_xlabel('Mean decrease in Macro F1')
    ax.set_title(title, fontsize=10)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_ablation(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    names = frame['ablation'].tolist()
    means = frame['macro_f1_mean'].tolist()
    stds = frame['macro_f1_std'].tolist()
    bars = ax.bar(
        range(len(names)), means, yerr=stds, capsize=4,
        color=['#0f766e', '#14b8a6', '#f59e0b', '#dc2626', '#7c3aed', '#2563eb'][:len(names)],
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=12, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Macro F1')
    ax.set_title('Ablation study (5-class, stratified CV)', fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.015,
                f'{mean:.3f}', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SHAP 解释
# ---------------------------------------------------------------------------
def transformed_feature_map(pre, features: Sequence[str]) -> Tuple[List[str], List[str]]:
    """把预处理后的列名映射回原始特征名（one-hot 列归并到所属类别特征）。"""
    names = list(pre.get_feature_names_out())
    cat_cols = sorted([c for c in features if c in CATEGORICAL], key=len, reverse=True)
    groups: List[str] = []
    for name in names:
        if name.startswith('num__'):
            groups.append(name[5:])
        elif name.startswith('cat__'):
            rest = name[5:]
            matched = None
            for cat in cat_cols:
                if rest == cat or rest.startswith(cat + '_'):
                    matched = cat
                    break
            groups.append(matched or rest)
        else:
            groups.append(name)
    return names, groups


def _normalize_shap_values(raw, n_features: int) -> np.ndarray:
    """统一不同模型的 SHAP 输出形状为 (样本, 特征, 类别)。"""
    if isinstance(raw, list):
        arr = np.stack([np.asarray(item, dtype=float) for item in raw], axis=-1)
    else:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[1] != n_features and arr.shape[2] == n_features:
            arr = np.transpose(arr, (0, 2, 1))
    if arr.ndim != 3 or arr.shape[1] != n_features:
        raise ValueError(f'无法识别的 SHAP 输出形状: {arr.shape}')
    return arr


def run_shap_analysis(model, X_df: pd.DataFrame, features: Sequence[str],
                      out_dir: Path = REPORT_DIR, max_display: int = 20) -> Dict:
    """对树模型计算 SHAP 全局重要性，并保存汇总图与单例瀑布图。"""
    if not HAS_SHAP:
        return {'available': False}

    try:
        X_df = X_df.reset_index(drop=True)
        names, groups = transformed_feature_map(model.pre, model.feature_columns)
        Xt = model.pre.transform(X_df[model.feature_columns])
        explainer = shap.TreeExplainer(model.estimator)
        raw = explainer.shap_values(Xt)
        values = _normalize_shap_values(raw, len(names))

        column_importance = np.abs(values).mean(axis=(0, 2))
        frame = pd.DataFrame({
            'feature': groups,
            'output_column': names,
            'importance': column_importance,
        })
        grouped = frame.groupby('feature', as_index=False)['importance'].sum()
        grouped['label'] = [FEATURE_LABELS.get(f, f) for f in grouped['feature']]
        grouped = grouped.sort_values('importance', ascending=False)
        grouped.to_csv(out_dir / 'shap_importance.csv',
                       index=False, encoding='utf-8-sig')

        # 全局汇总图（按特征平均绝对贡献）
        plt.figure(figsize=(8.0, 6.4))
        shap.summary_plot(
            np.abs(values).mean(axis=2), Xt,
            feature_names=names, plot_type='bar',
            max_display=max_display, show=False,
        )
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_summary.png', dpi=170, bbox_inches='tight')
        plt.close('all')

        # 单例瀑布图：取预测置信度最高的样本
        proba = model.predict_proba(X_df)
        sample_index = int(np.argmax(proba.max(axis=1)))
        class_index = int(np.argmax(proba[sample_index]))
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            flat = np.asarray(expected).ravel()
            base_value = float(flat[class_index % flat.size])
        else:
            base_value = float(expected)

        plt.figure(figsize=(9.0, 6.0))
        shap.plots.waterfall(
            shap.Explanation(
                values=values[sample_index, :, class_index],
                base_values=base_value,
                data=Xt[sample_index],
                feature_names=names,
            ),
            max_display=12, show=False,
        )
        plt.savefig(out_dir / 'shap_waterfall.png', dpi=170, bbox_inches='tight')
        plt.close('all')

        return {
            'available': True,
            'explainer': 'TreeExplainer',
            'sample_size': int(len(Xt)),
            'top_features': grouped.head(10)[['feature', 'label', 'importance']].to_dict(
                orient='records'),
            'summary_plot': 'shap_summary.png',
            'waterfall_plot': 'shap_waterfall.png',
        }
    except Exception as exc:  # noqa: BLE001
        return {'available': False, 'error': str(exc)}


# ---------------------------------------------------------------------------
# 推理与解释
# ---------------------------------------------------------------------------
def risk_score_from_proba(proba: np.ndarray) -> np.ndarray:
    return proba @ RISK_WEIGHTS_5


def build_baseline(X: pd.DataFrame) -> Dict:
    baseline = {}
    for col in X.columns:
        if col in CATEGORICAL:
            mode = X[col].value_counts(dropna=False)
            baseline[col] = str(mode.index[0]) if len(mode) else 'unknown'
        else:
            value = pd.to_numeric(X[col], errors='coerce').median()
            baseline[col] = 0.0 if pd.isna(value) else float(value)
    return baseline


def local_contributions(model, X: pd.DataFrame, baseline: Dict,
                        columns: Optional[Sequence[str]] = None) -> np.ndarray:
    """逐特征把取值替换为典型值，用预测类别置信度的变化（百分点）近似贡献。"""
    columns = list(columns) if columns is not None else list(X.columns)
    base_scores = model.predict_proba(X).max(axis=1) * 100
    contributions = np.zeros((len(X), len(columns)), dtype=np.float32)
    for idx, col in enumerate(columns):
        perturbed = X.copy()
        perturbed[col] = baseline[col]
        new_scores = model.predict_proba(perturbed).max(axis=1) * 100
        contributions[:, idx] = base_scores - new_scores
    return contributions


def build_predictions(frame: pd.DataFrame, model, features: Sequence[str],
                      explain_features: Optional[Sequence[str]] = None,
                      source: str = 'resubstitution') -> pd.DataFrame:
    """批量推理并生成展示用评分。

    ``source`` 写入 ``prediction_source`` 列，区分「训练集重代入」与「OOF」。
    """
    X = prepare_frame(frame, features)
    proba = model.predict_proba(X)
    scores = risk_score_from_proba(proba)
    pred = np.argmax(proba, axis=1)

    baseline = build_baseline(X)
    # 只对全局重要性较高的特征做局部解释，避免把模型偶然依赖的弱特征当成风险因素
    explain_cols = [c for c in features
                    if explain_features is None or c in explain_features]
    contributions = local_contributions(
        model, X, baseline, explain_cols,
    )
    feature_names = explain_cols
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]

    top_factors = []
    for row_idx in range(len(X)):
        values = contributions[row_idx]
        order = np.argsort(-np.abs(values))[:5]
        factors = [
            {
                'feature': feature_names[j],
                'label': labels[j],
                'delta': round(float(values[j]), 1),
                'direction': 'raises' if values[j] >= 0 else 'lowers',
                'unit': 'ppt',
            }
            for j in order
        ]
        factors = [item for item in factors if abs(item['delta']) >= 0.5] or factors[:1]
        top_factors.append(json.dumps(factors, ensure_ascii=False))

    out = pd.DataFrame({
        'code_id': frame['code_id'].astype(int).values,
        'language_name': frame['language_name'].fillna('').astype(str).values,
        'predicted_level': [LEVEL_ORDER[i] for i in pred],
        'predicted_level_zh': [LEVEL_LABELS_ZH[LEVEL_ORDER[i]] for i in pred],
        'risk_score': np.round(scores, 2),
        'confidence': np.round(proba.max(axis=1), 4),
        'risk_band': pd.cut(
            scores, bins=[-1, 40, 60, 80, 101],
            labels=['low', 'medium', 'high', 'critical'],
        ).astype(str),
        'top_factors': top_factors,
    })
    for i, name in enumerate(LEVEL_ORDER):
        out[f'p_{name}'] = np.round(proba[:, i], 4)
    out['prediction_source'] = source
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='训练濒危语言风险预测模型')
    parser.add_argument('--no-torch', action='store_true', help='跳过 PyTorch 实体嵌入 MLP')
    parser.add_argument('--tabnet', action='store_true',
                        help='加入 TabNet 对比（需安装 pytorch-tabnet，训练明显更慢）')
    parser.add_argument('--tasks', default='5,2', help='参与训练的任务：5=五级，2=二分类')
    parser.add_argument(
        '--feature-set', choices=['full', 'weak'], default='full',
        help=('full=全特征主实验（默认，产物与改动前一致）；'
              'weak=只跑弱信息消融并并入 metrics.json'),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    include_torch = (not args.no_torch) and HAS_TORCH
    include_tabnet = args.tabnet and HAS_TABNET

    log(f'读取数据集 {DATASET_PATH.name}')
    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    features = [f for f in meta['features'] if f in df.columns]

    trainable = df[df['trainable'] == 1].reset_index(drop=True)
    log(f'总记录 {len(df)}，可训练 {len(trainable)}，特征 {len(features)}')

    for col in CATEGORICAL:
        if col not in features:
            features.append(col)

    X_all = prepare_frame(trainable, features)
    groups = {
        'family': trainable['family_top'].fillna('unknown').astype(str),
        'country': trainable['country'].fillna('unknown').astype(str),
    }

    if args.feature_set == 'weak':
        weak_y5 = trainable['target_5'].map(
            {c: i for i, c in enumerate(LEVEL_ORDER)}).astype(int).values
        run_weak_only(
            trainable, X_all, weak_y5, groups, 'group_family_5fold', features,
        )
        log('完成（weak feature-set）')
        return

    task_specs = {
        '5': {
            'name': 'target_5',
            'order': LEVEL_ORDER,
            'y': trainable['target_5'].map({c: i for i, c in enumerate(LEVEL_ORDER)}).astype(int).values,
            'schemes': ['stratified_5fold', 'group_family_5fold', 'group_country_5fold'],
        },
        '2': {
            'name': 'target_binary',
            'order': BINARY_ORDER,
            'y': trainable['target_binary'].map({c: i for i, c in enumerate(BINARY_ORDER)}).astype(int).values,
            'schemes': ['stratified_5fold', 'group_family_5fold'],
        },
    }
    selected_tasks = [t.strip() for t in args.tasks.split(',') if t.strip() in task_specs]

    cv_rows: List[Dict] = []
    comparison: Dict[str, Dict[str, Dict]] = {}
    cv_details: Dict[str, Dict[str, Dict]] = {}

    for task_key in selected_tasks:
        spec = task_specs[task_key]
        task_name = spec['name']
        class_names = spec['order']
        y = spec['y']
        factories = model_factories(
            len(class_names), include_torch, include_tabnet=include_tabnet,
        )
        comparison[task_name] = {}
        cv_details[task_name] = {}

        for scheme in spec['schemes']:
            splits = make_splits(X_all, y, groups, scheme)
            comparison[task_name][scheme] = {}
            cv_details[task_name][scheme] = {}
            for model_name, factory in factories.items():
                log(f'训练 {task_name} / {scheme} / {model_name}')
                rows, raw_rows, _ = run_cv(
                    X_all, y, splits, factory, class_names,
                    task_name, scheme, model_name, verbose=False,
                )
                cv_rows.extend(rows)
                summary = summarize_folds(rows, class_names)
                log(f'  macroF1={summary["macro_f1"]["mean"]:.4f} '
                    f'± {summary["macro_f1"]["std"]:.4f}')
                comparison[task_name][scheme][model_name] = summary
                cv_details[task_name][scheme][model_name] = {
                    'summary': summary,
                    'confusion_matrices': [r['confusion_matrix'] for r in raw_rows],
                }

    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(REPORT_DIR / 'cv_results.csv', index=False, encoding='utf-8-sig')
    dump_json(REPORT_DIR / 'model_comparison.json', comparison)
    plot_model_comparison(comparison, REPORT_DIR / 'model_comparison.png')

    # ---- 选择部署模型（sklearn 系列，指标最优） ----
    primary_task = task_specs[selected_tasks[0]]['name']
    deployable = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']
    selection_scheme = (
        'group_family_5fold'
        if 'group_family_5fold' in comparison[primary_task]
        else 'stratified_5fold'
    )
    scores = {
        name: values['macro_f1']['mean']
        for name, values in comparison[primary_task][selection_scheme].items()
        if name in deployable
    }
    best_name = max(scores, key=scores.get)
    log(f'部署模型选择: {best_name} ({selection_scheme} macroF1={scores[best_name]:.4f})')

    best_factory = model_factories(len(LEVEL_ORDER), False)[best_name]

    # ---- 留出集评估 ----
    y5 = task_specs['5']['y']
    holdout_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED,
    )
    train_idx, test_idx = next(holdout_splitter.split(
        X_all, y5, groups=groups['family'],
    ))
    log(f'留出集训练 {len(train_idx)} / 测试 {len(test_idx)}（按语系分组）')
    final_model = best_factory()
    final_model.fit(X_all.iloc[train_idx].reset_index(drop=True), y5[train_idx])
    test_proba = final_model.predict_proba(X_all.iloc[test_idx].reset_index(drop=True))
    test_pred = np.argmax(test_proba, axis=1)
    holdout_metrics = compute_metrics(y5[test_idx], test_pred, test_proba, LEVEL_ORDER)
    plot_confusion(
        np.array(holdout_metrics['confusion_matrix']), LEVEL_ORDER,
        REPORT_DIR / 'confusion_matrix_holdout.png',
        f'{best_name} · hold-out test',
    )

    # ---- SHAP 解释 ----
    log(f'计算 SHAP 解释（shap={HAS_SHAP}）')
    shap_payload = run_shap_analysis(
        final_model, X_all.iloc[test_idx].reset_index(drop=True), features,
    )
    if shap_payload.get('available'):
        log(f'SHAP 完成，输出 {len(shap_payload["top_features"])} 个重要特征')
    else:
        log(f'SHAP 跳过：{shap_payload.get("error", "shap 未安装")}')

    # ---- 置换重要性 ----
    log('计算置换重要性')
    perm = permutation_importance(
        final_model, X_all.iloc[test_idx].reset_index(drop=True), y5[test_idx],
        scoring=make_scorer(f1_score, average='macro', zero_division=0),
        n_repeats=8, random_state=SEED, n_jobs=1,
    )
    importance_frame = pd.DataFrame({
        'feature': features,
        'label': [FEATURE_LABELS.get(f, f) for f in features],
        'importance': perm.importances_mean,
        'std': perm.importances_std,
    }).sort_values('importance', ascending=False)
    importance_frame.to_csv(REPORT_DIR / 'permutation_importance.csv',
                            index=False, encoding='utf-8-sig')
    plot_importance(importance_frame, REPORT_DIR / 'permutation_importance.png',
                    f'Permutation importance · {best_name}')

    explain_features = importance_frame.head(12)['feature'].tolist()

    # ---- OOF（按语系分组 5 折）----
    # 每条可训练语言只由「没见过它」的那一折模型预测，供页面展示交叉验证估计
    log('生成按语系分组 5 折 out-of-fold 评分')
    oof_splits = make_splits(X_all, y5, groups, 'group_family_5fold')
    oof_proba = np.zeros((len(y5), len(LEVEL_ORDER)))
    oof_pred = np.full(len(y5), -1, dtype=int)
    oof_parts = []
    for fold, (tr, te) in enumerate(oof_splits, start=1):
        fold_model = best_factory()
        fold_model.fit(X_all.iloc[tr].reset_index(drop=True), y5[tr])
        proba = fold_model.predict_proba(X_all.iloc[te].reset_index(drop=True))
        oof_proba[te] = proba
        oof_pred[te] = np.argmax(proba, axis=1)
        oof_parts.append(build_predictions(
            trainable.iloc[te].reset_index(drop=True), fold_model, features,
            explain_features, source='oof',
        ))
        log(f'  OOF fold{fold}: test={len(te)} '
            f'macroF1={compute_metrics(y5[te], oof_pred[te], proba, LEVEL_ORDER)["macro_f1"]:.4f}')
    oof_frame = pd.concat(oof_parts, ignore_index=True).sort_values(
        'code_id').reset_index(drop=True)
    if (len(oof_frame) != len(trainable)
            or set(oof_frame['code_id']) != set(trainable['code_id'])
            or (oof_pred < 0).any()):
        raise RuntimeError('OOF 覆盖与可训练样本不一致')
    oof_metrics = compute_metrics(y5, oof_pred, oof_proba, LEVEL_ORDER)
    tmp_oof = ARTIFACT_DIR / 'language_risk_oof.csv.tmp'
    oof_frame.to_csv(tmp_oof, index=False, encoding='utf-8-sig')
    os.replace(tmp_oof, ARTIFACT_DIR / 'language_risk_oof.csv')
    log(f'  OOF macroF1={oof_metrics["macro_f1"]:.4f} rows={len(oof_frame)}')

    # ---- 消融实验 ----
    sample_cols = [c for c in features if c == 'sample_count' or c == 'sample_type_count'
                   or c.startswith('sample_')]
    literature_cols = ['bibliography_count', 'source_count', 'cited_count', 'source_type_count']
    geo_cols = ['latitude', 'longitude', 'has_location', 'location_count', 'country_count', 'region']

    assessment_cols = ['speaker_trend', 'transmission', 'domain_of_use']
    ablation_sets = {
        'full': list(features),
        'no_literature': [c for c in features if c not in literature_cols],
        'no_samples': [c for c in features if c not in sample_cols],
        'no_geo': [c for c in features if c not in geo_cols],
        'no_assessment': [c for c in features if c not in assessment_cols],
    }
    ablation_rows = []
    ablation_scheme = (
        'group_family_5fold'
        if 'group_family_5fold' in comparison['target_5']
        else 'stratified_5fold'
    )
    splits5 = make_splits(X_all, y5, groups, ablation_scheme)
    for ablation_name, subset in ablation_sets.items():
        subset = [c for c in subset if c in X_all.columns]
        if len(subset) < 5:
            continue
        X_sub = X_all[subset]
        log(f'消融实验 {ablation_name}（特征 {len(subset)}）')
        rows, _, _ = run_cv(
            X_sub, y5, splits5, best_factory, LEVEL_ORDER,
            'target_5', ablation_scheme, f'{best_name}:{ablation_name}',
            verbose=False,
        )
        ablation_rows.append({
            'ablation': ablation_name,
            'feature_count': len(subset),
            'macro_f1_mean': float(np.mean([r['macro_f1'] for r in rows])),
            'macro_f1_std': float(np.std([r['macro_f1'] for r in rows])),
            'accuracy_mean': float(np.mean([r['accuracy'] for r in rows])),
            'balanced_accuracy_mean': float(np.mean([r['balanced_accuracy'] for r in rows])),
        })

    # ---- 弱信息消融（显式分组；与主实验同一折与同一留出集）----
    weak_groups = load_feature_groups(features)
    write_splits_payload(trainable, splits5, train_idx, test_idx, ablation_scheme)
    full_macro = next(
        float(r['macro_f1_mean']) for r in ablation_rows if r['ablation'] == 'full')
    weak_row, weak_payload, _ = run_weak_ablation(
        X_all, y5, splits5, train_idx, test_idx, weak_groups, full_macro,
    )
    ablation_rows.append(weak_row)

    ablation_frame = pd.DataFrame(ablation_rows).sort_values(
        'macro_f1_mean', ascending=False,
    )
    ablation_frame.to_csv(REPORT_DIR / 'ablation.csv', index=False, encoding='utf-8-sig')
    plot_ablation(ablation_frame, REPORT_DIR / 'ablation.png')

    # ---- 全量重训并导出部署模型 ----
    log('在全部可训练数据上重训部署模型')
    deploy_model = best_factory()
    deploy_model.fit(X_all, y5)

    bundle = {
        'task': 'target_5',
        'model_name': best_name,
        'estimator': deploy_model.estimator,
        'preprocessor': deploy_model.pre,
        'feature_columns': list(features),
        'categorical_columns': list(CATEGORICAL),
        'numeric_columns': numeric_columns(features),
        'class_names': LEVEL_ORDER,
        'risk_weights': RISK_WEIGHTS_5.tolist(),
        'trained_rows': int(len(trainable)),
        'trained_at': datetime.now().isoformat(timespec='seconds'),
    }
    tmp_model = ARTIFACT_DIR / 'risk_model.joblib.tmp'
    joblib.dump(bundle, tmp_model)
    os.replace(tmp_model, ARTIFACT_DIR / 'risk_model.joblib')

    dump_json(ARTIFACT_DIR / 'risk_service_meta.json', {
        'model_name': best_name,
        'task': 'target_5',
        'level_order': LEVEL_ORDER,
        'level_labels_zh': LEVEL_LABELS_ZH,
        'risk_weights': RISK_WEIGHTS_5.tolist(),
        'feature_columns': list(features),
        'feature_labels': {f: FEATURE_LABELS.get(f, f) for f in features},
        'categorical_columns': list(CATEGORICAL),
        'trained_rows': int(len(trainable)),
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'shap_available': bool(shap_payload.get('available')),
        # 页面默认展示 OOF（交叉验证估计）而非训练集重代入结果
        'prediction_source': 'oof',
        'oof': {
            'file': 'ml/artifacts/language_risk_oof.csv',
            'scheme': 'group_family_5fold',
            'folds': 5,
            'rows': int(len(oof_frame)),
            'macro_f1': float(oof_metrics['macro_f1']),
            'generated_at': datetime.now().isoformat(timespec='seconds'),
        },
    })

    # ---- 全量语言风险评分（训练集重代入，仅作对照保留）----
    log('生成全量语言风险评分与风险因素（训练集重代入）')
    prediction_rows = df[df['trainable'] == 1].reset_index(drop=True)
    predictions = build_predictions(
        prediction_rows, deploy_model, features, explain_features,
        source='resubstitution',
    )
    tmp_pred = ARTIFACT_DIR / 'language_risk_predictions.csv.tmp'
    predictions.to_csv(tmp_pred, index=False, encoding='utf-8-sig')
    os.replace(tmp_pred, ARTIFACT_DIR / 'language_risk_predictions.csv')

    # ---- OOF 与重代入对照 ----
    resub_proba = predictions[[f'p_{c}' for c in LEVEL_ORDER]].values
    resub_metrics = compute_metrics(
        y5, resub_proba.argmax(axis=1), resub_proba, LEVEL_ORDER)
    joined = predictions[
        ['code_id', 'risk_score', 'predicted_level', 'risk_band', 'confidence']
    ].merge(
        oof_frame[['code_id', 'risk_score', 'predicted_level', 'risk_band', 'confidence']],
        on='code_id', suffixes=('_resub', '_oof'),
    )
    score_delta = joined['risk_score_oof'] - joined['risk_score_resub']
    comparison_stats = {
        'rows': int(len(joined)),
        'score_changed_rows': int((score_delta.abs() > 0.05).sum()),
        'score_change_rate': float((score_delta.abs() > 0.05).mean()),
        'mean_abs_score_delta': float(score_delta.abs().mean()),
        'max_abs_score_delta': float(score_delta.abs().max()),
        'mean_score_delta': float(score_delta.mean()),
        'level_changed_rows': int(
            (joined['predicted_level_oof'] != joined['predicted_level_resub']).sum()),
        'band_changed_rows': int(
            (joined['risk_band_oof'] != joined['risk_band_resub']).sum()),
        'confidence_mean_resubstitution': float(joined['confidence_resub'].mean()),
        'confidence_mean_oof': float(joined['confidence_oof'].mean()),
        'risk_score_mean_resubstitution': float(joined['risk_score_resub'].mean()),
        'risk_score_mean_oof': float(joined['risk_score_oof'].mean()),
        'macro_f1_delta_oof_minus_resubstitution':
            float(oof_metrics['macro_f1'] - resub_metrics['macro_f1']),
    }
    log(f'  OOF 与重代入：分数变化 {comparison_stats["score_changed_rows"]}/'
        f'{len(joined)} 行，macroF1 {resub_metrics["macro_f1"]:.4f} -> '
        f'{oof_metrics["macro_f1"]:.4f}')

    # ---- 汇总报告 ----
    metrics_payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'dataset': {
            # 记录相对项目根目录的路径，避免把开发机绝对路径写进报告
            'path': os.path.relpath(DATASET_PATH, BASE_DIR.parent).replace(os.sep, '/'),
            'rows': int(len(df)),
            'trainable_rows': int(len(trainable)),
            'features': list(features),
        },
        'tasks': {
            task_specs[key]['name']: {
                'class_order': task_specs[key]['order'],
                'schemes': task_specs[key]['schemes'],
            }
            for key in selected_tasks
        },
        'cv_comparison': comparison,
        'best_model': {
            'name': best_name,
            'selection_rule': 'stratified 5-fold macro F1（仅 sklearn 系列可部署模型）',
        },
        'holdout': holdout_metrics,
        'oof': {
            'scheme': 'group_family_5fold',
            'folds': 5,
            'rows': int(len(oof_frame)),
            'prediction_file': 'ml/artifacts/language_risk_oof.csv',
            'prediction_source': 'oof',
            'metrics': oof_metrics,
            'risk_score_mean': float(oof_frame['risk_score'].mean()),
            'confidence_mean': float(oof_frame['confidence'].mean()),
        },
        'resubstitution': {
            'rows': int(len(predictions)),
            'prediction_file': 'ml/artifacts/language_risk_predictions.csv',
            'prediction_source': 'resubstitution',
            'metrics': resub_metrics,
            'risk_score_mean': float(predictions['risk_score'].mean()),
            'confidence_mean': float(predictions['confidence'].mean()),
        },
        'oof_vs_resubstitution': comparison_stats,
        'shap': shap_payload,
        'ablation': ablation_frame.to_dict(orient='records'),
        'weak_ablation': weak_payload,
        'top_permutation_importance': importance_frame.head(12).to_dict(orient='records'),
        'predictions_summary': {
            'rows': int(len(predictions)),
            'risk_band_counts': predictions['risk_band'].value_counts().to_dict(),
            'predicted_level_counts': predictions['predicted_level'].value_counts().to_dict(),
            'risk_score_mean': float(predictions['risk_score'].mean()),
        },
        'notes': [
            '按语言聚合为一行一种语言，训练/测试不会共享同一条语言记录',
            '标签使用 legacy_endangerment_level，特征未包含 level_of_endangerment 与 vitality.endangerment_level',
            'speaker_trends / intergenerational_transmission 属于人工评估字段，仅作为预警输入，泛化能力由分组交叉验证检验',
            '页面展示分数使用按语系分组 5 折 OOF 预测，缺失时回退到训练集重代入结果',
            'OOF 每折模型未见过被预测语言；重代入结果仅保留为对照，不再作为页面默认口径',
            '弱信息消融的分组见 ml/feature_groups.json；与主实验共用同一落盘折索引，'
            '只训练 LightGBM（附 RandomForest 对照），用于量化档案评估类特征的贡献',
            '可解释性：置换重要性 + 树模型 SHAP（TreeExplainer）'
            if shap_payload.get('available')
            else '可解释性：置换重要性（SHAP 未安装或不可用）',
            'TabNet 已参与模型对比' if include_tabnet
            else 'TabNet 未启用（安装 pytorch-tabnet 后加 --tabnet）',
        ],
    }
    dump_json(REPORT_DIR / 'metrics.json', metrics_payload)

    log('完成')
    log(f'部署模型: {ARTIFACT_DIR / "risk_model.joblib"}')
    log(f'风险评分: {ARTIFACT_DIR / "language_risk_predictions.csv"}')
    log(f'评估报告: {REPORT_DIR / "metrics.json"}')


if __name__ == '__main__':
    main()
