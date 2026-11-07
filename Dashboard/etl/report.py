"""status / verify 的统计与断言。"""
from __future__ import annotations

from django.db.models import Count
from django.db.models import CharField, TextField

from Dashboard.models import (
    BibliographyLink, ChangeSuggestion, Country, CountryRegion, DomainOfUse,
    ETLRun, Language, LanguageActivity, LanguageCode, LanguageContext,
    LanguageCountry, LanguageDocumentation, LanguageLocationDetail,
    LanguageMedia, LanguageOrthography, LanguageResource, LanguageSample,
    LanguageSource, LanguageVitality, Notification, NotificationUser, Region,
    SampleCategory, SampleCategoryLink, SampleContributorLink,
    SampleTranslation, ScoreLevel, SourceType, SpeakerData,
    SpeakerNumberTrend, Transmission,
)
from Dashboard.etl.config import (
    KNOWN_ALLOWED_DIFF, SOURCE_TARGET_PAIRS, STATUS_FIELDS,
    SUPPRESSED_SAMPLE_LINK_SOURCES, SUPPRESSED_SOURCE_REFS,
    excluded_ids, hard_excluded_ids, merged_dropped_rows, name_overrides,
    test_records, unresolved_ids,
)
from Dashboard.etl.source import ElcatSource

STATUS_MODELS = [
    Language, LanguageCode, LanguageSource, BibliographyLink, LanguageVitality,
    SpeakerData, LanguageLocationDetail, LanguageContext, LanguageOrthography,
    LanguageResource, LanguageMedia, ChangeSuggestion, Notification,
    NotificationUser, LanguageDocumentation, LanguageActivity, LanguageSample,
    SampleCategory, SampleCategoryLink, SampleContributorLink, SampleTranslation,
    Country, Region, CountryRegion, LanguageCountry, Transmission,
    SpeakerNumberTrend, DomainOfUse, SourceType, ScoreLevel,
]


def collect_status(include_source=True):
    tables = []
    for model in STATUS_MODELS:
        table = model._meta.db_table
        row = {'table': table, 'rows': model.objects.count(), 'fill': {}}
        for field in STATUS_FIELDS.get(table, []):
            total = row['rows'] or 1
            field_obj = model._meta.get_field(field)
            qs = model.objects.filter(**{f'{field}__isnull': False})
            if isinstance(field_obj, (CharField, TextField)):
                qs = qs.exclude(**{field: ''})
            non_empty = qs.count()
            row['fill'][field] = {'count': non_empty, 'rate': round(non_empty / total * 100, 2)}
        tables.append(row)

    duplicates = list(
        Language.objects.exclude(language_name__isnull=True).exclude(language_name='')
        .values('language_name').annotate(n=Count('language_id'))
        .filter(n__gt=1).order_by('-n')[:20]
    )
    missing_names = Language.objects.exclude(language_name__isnull=True).exclude(
        language_name='').count()
    total_languages = Language.objects.count()
    tests = test_records()
    test_state = []
    for lang_id, record in tests.items():
        exists = Language.objects.filter(language_id=lang_id).exists()
        test_state.append({'language_id': lang_id, 'exists': exists,
                           'delete': bool(record.get('delete')),
                           'note': record.get('note', '')})

    payload = {
        'tables': tables,
        'duplicate_names': duplicates,
        'missing_names': {'named': missing_names, 'total': total_languages,
                          'rate': round(missing_names / (total_languages or 1) * 100, 2)},
        'test_records': test_state,
        'unresolved': [{'language_id': k, 'reason': v}
                       for k, v in unresolved_ids().items()],
        'last_runs': [
            {'id': run.id, 'command': run.command, 'status': run.status,
             'dry_run': run.dry_run, 'started_at': run.started_at.isoformat(timespec='seconds'),
             'finished_at': run.finished_at.isoformat(timespec='seconds') if run.finished_at else None,
             'error': run.error[:200] if run.error else ''}
            for run in ETLRun.objects.order_by('-id')[:5]
        ],
    }

    if include_source:
        try:
            with ElcatSource() as source:
                excluded = sorted(excluded_ids())
                hard_excluded = sorted(hard_excluded_ids())
                dropped = merged_dropped_rows()
                suppressed = {}
                if excluded:
                    # language 主表：所有抑制 ID（含合并别名）都不再存在
                    suppressed['language'] = source.count_where(
                        'language_language', 'code_id', excluded)
                    # 其他明细表：只扣除「从未导入」的抑制行；合并行已迁移保留
                    for target_table, (source_table, column) in SUPPRESSED_SOURCE_REFS.items():
                        if target_table == 'language':
                            continue
                        suppressed[target_table] = source.count_where(
                            source_table, column, hard_excluded)
                    sample_ids = source.ids_where(
                        'language_sample', 'code_id', hard_excluded, 'id')
                    for target_table, source_table in SUPPRESSED_SAMPLE_LINK_SOURCES.items():
                        suppressed[target_table] = source.count_where(
                            source_table, 'sample_id', sample_ids)
                    for table, count in dropped.items():
                        suppressed[table] = suppressed.get(table, 0) + count
                comparisons = []
                for source_table, target_table in SOURCE_TARGET_PAIRS:
                    model = next(m for m in STATUS_MODELS if m._meta.db_table == target_table)
                    source_rows_total = source.count(source_table)
                    suppressed_rows = suppressed.get(target_table, 0)
                    source_rows = source_rows_total - suppressed_rows
                    target_rows = model.objects.count()
                    comparisons.append({
                        'source_table': source_table, 'target_table': target_table,
                        'source_rows': source_rows,
                        'source_rows_total': source_rows_total,
                        'suppressed_rows': suppressed_rows,
                        'target_rows': target_rows,
                        'diff': target_rows - source_rows,
                        'ratio': round(target_rows / (source_rows or 1) * 100, 2),
                    })
                payload['source_comparison'] = comparisons
                payload['source_available'] = True
        except Exception as exc:
            payload['source_available'] = False
            payload['source_error'] = str(exc)
    else:
        payload['source_available'] = False
    return payload


def format_status(payload):
    lines = ['=== 数据表行数 ===']
    for row in payload['tables']:
        lines.append(f'{row["table"]:<32} {row["rows"]:>8}')
    lines.append('')
    lines.append('=== 关键字段填充率 ===')
    for row in payload['tables']:
        if not row['fill']:
            continue
        for field, info in row['fill'].items():
            lines.append(f'{row["table"]}.{field:<32} {info["count"]:>8}/{row["rows"]:<8} '
                         f'{info["rate"]:>6.2f}%')
    lines.append('')
    lines.append('=== 重复名称（Top 20） ===')
    if payload['duplicate_names']:
        for item in payload['duplicate_names']:
            lines.append(f'{item["language_name"]}: {item["n"]}')
    else:
        lines.append('无')
    lines.append('')
    named = payload['missing_names']
    lines.append(f'=== 名称 === 已命名 {named["named"]}/{named["total"]} '
                 f'({named["rate"]}%)')
    for item in payload['unresolved']:
        lines.append(f'  未解决（确认不要名称）#{item["language_id"]}: {item["reason"]}')
    lines.append('')
    lines.append('=== 测试记录 ===')
    for item in payload['test_records']:
        lines.append(f'#{item["language_id"]}: 存在={item["exists"]} '
                     f'删除={item["delete"]} {item["note"]}')
    lines.append('')
    if payload.get('source_available'):
        lines.append('=== 源库 / 目标库对照 ===')
        for item in payload.get('source_comparison', []):
            note = (f' 抑制 -{item["suppressed_rows"]}'
                    if item.get('suppressed_rows') else '')
            lines.append(
                f'{item["source_table"]:<32} {item["source_rows"]:>8} -> '
                f'{item["target_table"]:<30} {item["target_rows"]:>8} '
                f'diff={item["diff"]:+d} ({item["ratio"]}%){note}')
    else:
        lines.append(f'源库不可用：{payload.get("source_error", "未连接")}')
    lines.append('')
    lines.append('=== 最近 ETL 运行 ===')
    if payload['last_runs']:
        for run in payload['last_runs']:
            lines.append(f'#{run["id"]} {run["command"]} {run["status"]} '
                         f'dry_run={run["dry_run"]} {run["started_at"]} {run["error"]}')
    else:
        lines.append('无')
    return '\n'.join(lines)


def run_verify():
    """返回 (ok, checks, payload)。只读，不写库。"""
    checks = []
    payload = collect_status(include_source=True)

    # 1. 行数对照
    if payload.get('source_available'):
        for item in payload.get('source_comparison', []):
            allowed = KNOWN_ALLOWED_DIFF.get(item['target_table'], 0)
            ok = abs(item['diff']) <= allowed
            checks.append({
                'name': f'行数 {item["target_table"]}',
                'ok': ok,
                'detail': (f'源 {item["source_rows"]} / 目标 {item["target_rows"]} '
                           f'(diff {item["diff"]:+d}, 允许 ±{allowed})'),
            })
    else:
        checks.append({'name': '源库连接', 'ok': True,
                       'detail': '源库不可用，跳过源/目标行数断言'})

    # 2. 名称完整性
    named = payload['missing_names']
    checks.append({'name': 'language_name 填充率', 'ok': named['rate'] >= 95.0,
                   'detail': f'{named["named"]}/{named["total"]} ({named["rate"]}%)'})

    # 3. 未解决项必须是显式空值标记
    unresolved = unresolved_ids()
    for lang_id in unresolved:
        obj = Language.objects.filter(language_id=lang_id).values(
            'language_name', 'name_source').first()
        if obj is None:
            checks.append({'name': f'unresolved #{lang_id}', 'ok': False, 'detail': '记录不存在'})
        else:
            marker_ok = (obj['language_name'] or '') == '' and \
                        (obj['name_source'] or '').startswith('manual:unresolved')
            checks.append({'name': f'unresolved #{lang_id}', 'ok': marker_ok,
                           'detail': f'name={obj["language_name"]!r} source={obj["name_source"]!r}'})

    # 4. 人工名称与配置一致
    mismatched = []
    for lang_id, (name, source) in name_overrides().items():
        obj = Language.objects.filter(language_id=lang_id).values(
            'language_name', 'name_source').first()
        if obj is None:
            continue
        if (obj['language_name'] or '') != name:
            mismatched.append(lang_id)
    checks.append({'name': '人工核实名称', 'ok': not mismatched,
                   'detail': f'不一致 {len(mismatched)} 条: {mismatched[:10]}'})

    # 5. 测试记录清理
    for lang_id, record in test_records().items():
        if not record.get('delete'):
            continue
        exists = Language.objects.filter(language_id=lang_id).exists()
        checks.append({'name': f'测试记录 #{lang_id} 已清理', 'ok': not exists,
                       'detail': '仍存在' if exists else '不存在'})

    # 6. 抑制名单必须统一不在目标库
    excluded = sorted(excluded_ids())
    present = sorted(Language.objects.filter(language_id__in=excluded)
                     .values_list('language_id', flat=True))
    checks.append({
        'name': f'抑制名单 {len(excluded)} 条不存在',
        'ok': not present,
        'detail': (f'仍存在 {len(present)} 条: {present[:10]}'
                   if present else '全部不存在'),
    })

    ok = all(item['ok'] for item in checks)
    return ok, checks, payload
