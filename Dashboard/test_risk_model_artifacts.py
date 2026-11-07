"""风险模型产物与特征口径的回归测试。

覆盖三件事：
1. 训练报告的关键指标不出现退化；
2. OOF 产物完整覆盖可训练语言、无重复、标签合法；
3. build_dataset 与 risk.py 单条推理的特征组装保持一致（防口径漂移）。
"""
import json
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE_DIR / 'ml' / 'reports' / 'metrics.json'
DATASET_PATH = BASE_DIR / 'ml' / 'endangerment_dataset.csv'
OOF_PATH = BASE_DIR / 'ml' / 'artifacts' / 'language_risk_oof.csv'
LEVEL_ORDER = ['unsafe', 'danger', 'endangered', 'seriously_endangered', 'extinct']
CATEGORICAL = ['family_top', 'country', 'region', 'speaker_trend', 'transmission', 'domain_of_use']


def _metrics():
    return json.loads(METRICS_PATH.read_text(encoding='utf-8'))


def test_lightgbm_metrics_above_threshold():
    metrics = _metrics()
    family_cv = metrics['cv_comparison']['target_5']['group_family_5fold']['LightGBM'][
        'macro_f1']['mean']
    assert family_cv >= 0.85, f'LightGBM family 5-fold macro F1 = {family_cv}'
    assert metrics['holdout']['macro_f1'] >= 0.85
    assert metrics['dataset']['trainable_rows'] >= 3400


def test_oof_artifact_covers_all_trainable_languages():
    metrics = _metrics()
    oof = pd.read_csv(OOF_PATH, encoding='utf-8-sig')
    assert len(oof) == metrics['dataset']['trainable_rows']
    assert oof['code_id'].is_unique
    assert set(oof['predicted_level']).issubset(set(LEVEL_ORDER))
    assert (oof['prediction_source'] == 'oof').all()

    dataset = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    trainable_ids = set(dataset.loc[dataset['trainable'] == 1, 'code_id'].astype(int))
    assert set(oof['code_id'].astype(int)) == trainable_ids


@pytest.mark.django_db
def test_build_dataset_matches_risk_feature_assembly():
    """用测试库 fixture 复算，断言两侧每个特征值都相等。

    脚本导入会执行 django.setup()，此时 settings 已由 pytest 配置为
    language.settings_test，读取的是测试库，不会访问生产库。
    """
    from ml import build_dataset
    from Dashboard.management.commands import risk as risk_command
    from Dashboard.models import (
        BibliographyLink, Country, CountryRegion, Language, LanguageActivity,
        LanguageCode, LanguageContext, LanguageCountry, LanguageDocumentation,
        LanguageLocationDetail, LanguageMedia, LanguageOrthography,
        LanguageResource, LanguageSample, LanguageSource, LanguageVitality,
        Region, SpeakerData,
    )

    region_a = Region.objects.create(id=10, name='Asia')
    region_b = Region.objects.create(id=20, name='Europe')
    region_c = Region.objects.create(id=30, name='Africa')

    country1 = Country.objects.create(name='Country One')
    country2 = Country.objects.create(name='Country Two')
    country3 = Country.objects.create(name='Country Three')
    # 插入顺序与 id 顺序相反，验证两侧都按 id 升序取第一条
    CountryRegion.objects.create(id=11, country_id=country1.id, region_id=region_b.id)
    CountryRegion.objects.create(id=10, country_id=country1.id, region_id=region_a.id)
    CountryRegion.objects.create(id=12, country_id=country3.id, region_id=region_c.id)
    # country2 没有 country_region 记录，区域应为 unknown

    # ---- L1：多国家、多区域、样本两类型、有效坐标、完整关联 ----
    lang1 = Language.objects.create(
        language_name='Lang One', legacy_endangerment_level='danger',
        number_of_speakers=1200, speaker_trends='stable',
        intergenerational_transmission='1',
        language_family='Indo-European; Romance', country='Testland',
        additional_notes='notes', media='video',
    )
    LanguageCode.objects.create(
        id=lang1.language_id, iso_code='abc', alternate_names='a; b',
        classification='x; y; z',
    )
    for sid, stype in [(1, 'A'), (2, 'A'), (3, 'V')]:
        LanguageSample.objects.create(id=sid, code_id=lang1.language_id, sample_type=stype)
    LanguageLocationDetail.objects.create(
        id=1, code_id=lang1.language_id, coordinates='10.0, 20.0', preferred=False)
    LanguageLocationDetail.objects.create(
        id=2, code_id=lang1.language_id, coordinates='11.0, 21.0', preferred=True)
    LanguageLocationDetail.objects.create(
        id=3, code_id=lang1.language_id, coordinates='12.0, 22.0', preferred=True)
    LanguageCountry.objects.create(id=1, language_id=lang1.language_id, country_id=country1.id)
    LanguageCountry.objects.create(id=2, language_id=lang1.language_id, country_id=country3.id)
    LanguageVitality.objects.create(
        id=1, code_id=lang1.language_id, domains_of_use_id=5, preferred=False)
    LanguageVitality.objects.create(
        id=2, code_id=lang1.language_id, domains_of_use_id=7, preferred=True)
    SpeakerData.objects.create(
        id=1, code_id=lang1.language_id, speaker_number=100, ethnic_population=2000,
        semi_speakers=10, children=5, young_adults=6, older_adults=7, elders=8)
    SpeakerData.objects.create(id=2, code_id=lang1.language_id, speaker_number=50)
    LanguageContext.objects.create(
        id=1, code_id=lang1.language_id, government_support='yes',
        institutional_support='school', speaker_attitude='positive')
    LanguageContext.objects.create(id=2, code_id=lang1.language_id)
    LanguageSource.objects.create(id=1, source_type_id=3)
    LanguageSource.objects.create(id=2, source_type_id=3)
    LanguageSource.objects.create(id=3, source_type_id=None)
    BibliographyLink.objects.create(id=1, code_id=lang1.language_id, source_id=1, cited=True)
    BibliographyLink.objects.create(id=2, code_id=lang1.language_id, source_id=2, cited=False)
    BibliographyLink.objects.create(id=3, code_id=lang1.language_id, source_id=3, cited=False)
    LanguageActivity.objects.create(id=1, code_id=lang1.language_id, obj_type='A')
    LanguageActivity.objects.create(id=2, code_id=lang1.language_id, obj_type='B')
    LanguageOrthography.objects.create(id=1, code_id=lang1.language_id, scripts='Latin; Cyrillic')
    LanguageOrthography.objects.create(id=2, code_id=lang1.language_id, scripts='Latin')
    LanguageDocumentation.objects.create(
        id=1, code_id=lang1.language_id, documentation_index=3)
    LanguageDocumentation.objects.create(
        id=2, code_id=lang1.language_id, documentation_index=7)
    LanguageResource.objects.create(id=1, code_id=lang1.language_id)
    LanguageMedia.objects.create(id=1, code_id=lang1.language_id)
    LanguageMedia.objects.create(id=2, code_id=lang1.language_id)

    # ---- L2：无样本、无坐标、无区域、无语言代码 ----
    lang2 = Language.objects.create(
        language_name='Lang Two', legacy_endangerment_level='unsafe',
        number_of_speakers=None, speaker_trends=None,
        intergenerational_transmission=None, language_family=None, country=None,
        additional_notes=None, media='',
    )

    # ---- L3：无区域国家、非法坐标、空国家、单类型边界 ----
    lang3 = Language.objects.create(
        language_name='Lang Three', legacy_endangerment_level='endangered',
        number_of_speakers=80, speaker_trends='', intergenerational_transmission=None,
        language_family='Language isolate', country='', additional_notes='',
        media='',
    )
    LanguageCode.objects.create(id=lang3.language_id, iso_code='', alternate_names='',
                                classification='')
    LanguageSample.objects.create(id=4, code_id=lang3.language_id, sample_type='A')
    LanguageSample.objects.create(id=5, code_id=lang3.language_id, sample_type='T')
    LanguageLocationDetail.objects.create(
        id=4, code_id=lang3.language_id, coordinates='not-a-coord', preferred=True)
    LanguageCountry.objects.create(id=3, language_id=lang3.language_id, country_id=country2.id)
    LanguageVitality.objects.create(
        id=3, code_id=lang3.language_id, domains_of_use_id=None, preferred=True)

    # ---- L4：首选位置非法坐标、多国家与多活力记录 ----
    lang4 = Language.objects.create(
        language_name='Lang Four', legacy_endangerment_level='extinct',
        number_of_speakers=0, speaker_trends='declining',
        intergenerational_transmission='5',
        language_family='Indo-European', country='Testland',
        additional_notes='x', media='',
    )
    LanguageLocationDetail.objects.create(
        id=5, code_id=lang4.language_id, coordinates='30.0, 40.0', preferred=False)
    LanguageLocationDetail.objects.create(
        id=6, code_id=lang4.language_id, coordinates='bad', preferred=True)
    LanguageCountry.objects.create(id=4, language_id=lang4.language_id, country_id=country1.id)
    LanguageCountry.objects.create(id=5, language_id=lang4.language_id, country_id=country2.id)
    LanguageCountry.objects.create(id=6, language_id=lang4.language_id, country_id=country3.id)
    LanguageVitality.objects.create(
        id=4, code_id=lang4.language_id, domains_of_use_id=2, preferred=False)
    LanguageVitality.objects.create(
        id=5, code_id=lang4.language_id, domains_of_use_id=9, preferred=True)

    df, feature_columns = build_dataset.build_frame()
    by_id = df.set_index('code_id')

    # sample_type_count 不再把 sample_count 算进去
    assert int(by_id.loc[lang1.language_id, 'sample_type_count']) == 2
    assert int(by_id.loc[lang1.language_id, 'sample_count']) == 3
    assert int(by_id.loc[lang2.language_id, 'sample_type_count']) == 0

    checked = 0
    for code_id in (lang1.language_id, lang2.language_id, lang3.language_id, lang4.language_id):
        expected = by_id.loc[code_id]
        got, _ = risk_command._build_feature_row(code_id, feature_columns, CATEGORICAL)
        assert got is not None, f'risk.py 无法组装 code_id={code_id}'
        for col in feature_columns:
            if col in ('latitude', 'longitude'):
                left, right = got.get(col), expected[col]
                if pd.isna(left) and pd.isna(right):
                    continue
                assert not (pd.isna(left) or pd.isna(right)), f'{code_id}.{col} 缺失性不一致'
                assert float(left) == pytest.approx(float(right))
            elif col in CATEGORICAL:
                assert str(got.get(col)) == str(expected[col]), f'{code_id}.{col}'
            else:
                assert float(got.get(col)) == pytest.approx(float(expected[col])), \
                    f'{code_id}.{col}'
        checked += 1

    assert checked == 4
