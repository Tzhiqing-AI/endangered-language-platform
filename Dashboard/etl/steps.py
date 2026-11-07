"""ETL 各阶段实现：rebuild / names / backfill / ml。"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.core.cache import cache

from Dashboard.models import (
    BibliographyLink, ChangeSuggestion, Country, CountryRegion, DomainOfUse,
    Language, LanguageActivity, LanguageCode, LanguageContext, LanguageCountry,
    LanguageDocumentation, LanguageLocation, LanguageLocationDetail,
    LanguageMedia, LanguageOrthography, LanguageResource, LanguageSample,
    LanguageSource, LanguageVitality, Notification, NotificationUser, Region,
    SampleCategory, SampleCategoryLink, SampleContributorLink,
    SampleTranslation, ScoreLevel, SourceType, SpeakerData,
    SpeakerNumberTrend, Transmission,
)
from Dashboard.etl.config import (
    MIRROR_TABLES, excluded_ids, name_overrides, test_records, unresolved_ids,
)
from Dashboard.etl.source import (
    ElcatSource, as_date, aware_dt, bo, f, i, parse_coordinates, s,
)
from Dashboard.etl.upsert import rebuild_derived, upsert_rows

RAW_LEVEL_MAP = {
    'vulnerable': 'unsafe',
    'at risk': 'unsafe',
    'threatened': 'danger',
    'endangered': 'endangered',
    'severely endangered': 'seriously_endangered',
    'critically endangered': 'seriously_endangered',
    'dormant': 'extinct',
    'extinct': 'extinct',
    'awakening': 'awakening',
    'safe': 'safe',
}

SIMPLIFIED_LEVEL_MAP = {
    'high': 'seriously_endangered',
    'mid': 'endangered',
    'low': 'danger',
    'dormant': 'extinct',
    'awakening': 'awakening',
}


class StepContext:
    def __init__(self, run_id=None, dry_run=False, source=None):
        self.run_id = run_id
        self.dry_run = dry_run
        self.source = source
        self.stats = {}

    def write(self, model, incoming, fields, *, prune=True):
        is_mirror = model._meta.db_table in MIRROR_TABLES
        result = upsert_rows(
            model, incoming, fields,
            run_id=self.run_id,
            prune=prune and is_mirror,
            dry_run=self.dry_run,
        )
        self.stats[model._meta.db_table] = result
        return result


def parse_raw_level(value):
    if not value:
        return ''
    base = value.split('(')[0].strip().lower()
    return RAW_LEVEL_MAP.get(base, '')


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------

def rebuild_elcat(ctx):
    """以 elcat_source 全量 upsert 规范化表；只裁剪 ETL 台账内的过期行。"""
    source = ctx.source or ElcatSource()
    unresolved = unresolved_ids()
    tests = test_records()
    excluded = excluded_ids()
    skip_ids = {cid for cid, record in tests.items() if record.get('delete')} | set(excluded)

    def keep_code(value):
        return i(value, 0) not in skip_ids

    def exclude_rows(table, *columns):
        """读取源表时即剔除测试/抑制记录，确保它们不进入任何写入路径。"""
        rows = source.rows(table)
        if not skip_ids:
            return rows
        return [r for r in rows
                if not any(i(r.get(column), 0) in skip_ids for column in columns)]

    code_rows = source.rows('language_codes')  # 语言代码目录不受抑制名单影响
    raw_language_rows = source.rows('language_language')
    language_rows = [r for r in raw_language_rows
                     if i(r.get('code_id'), 0) not in skip_ids]
    skipped_tests = sorted({i(r.get('code_id')) for r in raw_language_rows
                            if i(r.get('code_id'), 0) in skip_ids})
    raw_sample_rows = source.rows('language_sample')
    skip_sample_ids = {i(r.get('id')) for r in raw_sample_rows
                       if i(r.get('id')) and i(r.get('code_id'), 0) in skip_ids}
    sample_rows = [r for r in raw_sample_rows if i(r.get('id')) not in skip_sample_ids]
    source_rows = source.rows('language_sources')
    bib_rows = exclude_rows('language_bibliography', 'code_id')
    vitality_rows = exclude_rows('language_vitality', 'code_id')
    speaker_rows = exclude_rows('language_speakers', 'code_id')
    location_rows = exclude_rows('language_locations', 'code_id')
    context_rows = exclude_rows('language_context', 'code_id')
    orthography_rows = exclude_rows('language_orthography', 'code_id')
    resource_rows = exclude_rows('language_resources', 'code_id')
    media_rows = exclude_rows('language_languagemedia', 'code_id')
    change_rows = source.rows('language_changesuggestion')
    notification_rows = source.rows('language_notification')
    notification_user_rows = source.rows('language_notification_users')
    documentation_rows = exclude_rows('language_documentation', 'code_id')
    activity_rows = exclude_rows('language_activity', 'code_id')
    sample_category_rows = source.rows('language_samplecategory')
    sample_contributor_rows = source.rows('language_samplecontributor')
    sample_translation_rows = source.rows('language_sampletranslation')
    category_rows = source.rows('categories_category')

    # 1. 查找表 ------------------------------------------------------------
    lookup_specs = [
        (Region, 'language_region', lambda r: {
            'id': i(r['id']), 'name': s(r.get('name'), 100)}),
        (Transmission, 'language_transmission', lambda r: {
            'id': i(r['id']), 'level': i(r.get('transmission_level'), 0),
            'description': s(r.get('description'))}),
        (SpeakerNumberTrend, 'language_speakernumbertrends', lambda r: {
            'id': i(r['id']), 'level': i(r.get('speaker_number_trend'), 0),
            'description': s(r.get('description'))}),
        (DomainOfUse, 'language_domainsofuse', lambda r: {
            'id': i(r['id']), 'level': i(r.get('domains_of_use_level'), 0),
            'description': s(r.get('description'))}),
        (SourceType, 'language_sourcetypes', lambda r: {
            'id': i(r['id']), 'name': s(r.get('source_type'), 100),
            'description': s(r.get('description'))}),
    ]
    for model, table, builder in lookup_specs:
        incoming = {}
        for r in source.rows(table):
            payload = builder(r)
            if payload['id']:
                incoming[payload['id']] = payload
        fields = [k for k in next(iter(incoming.values())).keys() if k != 'id'] if incoming else []
        ctx.write(model, incoming, fields)

    # 评分等级：自增主键 + (score_type, original_id) 唯一
    _rebuild_score_levels(ctx, source)

    # 2. 语言代码 ----------------------------------------------------------
    code_incoming = {}
    code_map = {}
    for r in code_rows:
        cid = i(r.get('id'))
        if not cid:
            continue
        code_map[cid] = r
        code_incoming[cid] = {
            'id': cid,
            'iso_code': s(r.get('code_val'), 150),
            'code_authorities': s(r.get('code_authorities'), 50),
            'language_name': s(r.get('primary_name'), 255),
            'alternate_names': s(r.get('alternate_names')),
            'classification': s(r.get('classification'), 500),
        }
    ctx.write(LanguageCode, code_incoming,
              ['iso_code', 'code_authorities', 'language_name',
               'alternate_names', 'classification'])

    pycountry = _load_pycountry()

    def resolve_name(code_id, code_val):
        code = code_map.get(code_id)
        if code and s(code.get('primary_name')):
            return s(code['primary_name']), 'ELCat SQL: language_codes.primary_name'
        if code_val:
            lang = pycountry.languages.get(alpha_3=code_val) if pycountry else None
            if lang and getattr(lang, 'name', None):
                return lang.name, 'ELCat SQL: code_val_cache + pycountry'
            return code_val, 'ELCat SQL: code_val_cache'
        return '', ''

    def resolve_family(code_id):
        code = code_map.get(code_id)
        return s(code.get('classification'))[:255] if code else ''

    # 3. 语言主表：绝不覆盖已有名称；测试记录跳过 ------------------------
    existing = {lang.language_id: lang for lang in Language.objects.all()}
    language_incoming = {}
    for r in language_rows:
        cid = i(r.get('code_id'))
        if not cid:
            continue
        auto_name, auto_source = resolve_name(cid, s(r.get('code_val_cache'), 150))
        if cid in unresolved:
            auto_name = ''
            auto_source = 'manual:unresolved'
        old = existing.get(cid)
        if old is not None:
            if old.language_name:
                # 已有名称连同来源原样保留，绝不改写人工核实结果
                name, name_source = old.language_name, old.name_source
            else:
                name, name_source = auto_name, auto_source
            level = s(r.get('simplified_level'), 21) or (old.level_of_endangerment or '')
            speakers = i(r.get('speakers_worldwide'))
            if speakers is None:
                speakers = old.number_of_speakers
            family = resolve_family(cid) or (old.language_family or '')
        else:
            name, name_source = auto_name, auto_source
            level = s(r.get('simplified_level'), 21)
            speakers = i(r.get('speakers_worldwide'))
            family = resolve_family(cid)
        language_incoming[cid] = {
            'language_id': cid,
            'language_name': name,
            'name_source': name_source,
            'level_of_endangerment': level,
            'number_of_speakers': speakers,
            'language_family': family,
        }
    ctx.write(Language, language_incoming,
              ['language_name', 'name_source', 'level_of_endangerment',
               'number_of_speakers', 'language_family'], prune=False)

    # 4. 国家 / 大区 / 语言-国家 ------------------------------------------
    def simple_id_name(rows):
        return {i(r['id']): {'id': i(r['id']), 'name': s(r.get('name'), 100)}
                for r in rows if i(r.get('id'))}

    ctx.write(Country, simple_id_name(source.rows('language_country')), ['name'])
    ctx.write(CountryRegion, {
        i(r['id']): {'id': i(r['id']), 'country_id': i(r.get('country_id'), 0),
                     'region_id': i(r.get('region_id'), 0)}
        for r in source.rows('language_country_region') if i(r.get('id'))
    }, ['country_id', 'region_id'])
    ctx.write(LanguageCountry, {
        i(r['id']): {'id': i(r['id']), 'language_id': i(r.get('language_id'), 0),
                     'country_id': i(r.get('country_id'), 0)}
        for r in source.rows('language_language_country')
        if i(r.get('id')) and keep_code(r.get('language_id'))
    }, ['language_id', 'country_id'])

    # 5. 文献来源与关联 ----------------------------------------------------
    source_incoming = {}

    def source_payload(sid, r=None):
        r = r or {}
        return {
            'id': sid,
            'source_type_id': i(r.get('source_type_id')),
            'address': s(r.get('address'), 255),
            'author': s(r.get('author'), 255),
            'booktitle': s(r.get('booktitle'), 255),
            'chapter': s(r.get('chapter'), 100),
            'edition': s(r.get('edition'), 100),
            'editor': s(r.get('editor'), 255),
            'translator': s(r.get('translator'), 255),
            'howpublished': s(r.get('howpublished'), 255),
            'institution': s(r.get('institution'), 255),
            'journal': s(r.get('journal'), 255),
            'month': s(r.get('month'), 50),
            'note': s(r.get('note'), 500),
            'num': s(r.get('num'), 50),
            'pages': s(r.get('pages'), 50),
            'publisher': s(r.get('publisher'), 255),
            'school': s(r.get('school'), 255),
            'series': s(r.get('series'), 255),
            'title': s(r.get('title'), 500),
            'url': s(r.get('url'), 1000),
            'volume': s(r.get('volume'), 50),
            'year': i(r.get('year')),
            'isbn': s(r.get('isbn'), 255),
            'free_text_citation': s(r.get('free_text_citation')),
            'copied_from': s(r.get('copied_from'), 255),
            'is_placeholder': bool(r.get('is_placeholder')),
        }

    for r in source_rows:
        sid = i(r.get('id'))
        if sid:
            source_incoming[sid] = source_payload(sid, r)
    cited_ids = {i(r.get('source_id'), 0) for r in bib_rows if i(r.get('source_id'))}
    for sid in cited_ids - set(source_incoming):
        source_incoming[sid] = source_payload(sid, {'is_placeholder': True})
    source_fields = [k for k in source_payload(1).keys() if k != 'id']
    ctx.write(LanguageSource, source_incoming, source_fields)

    bib_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'source_id': i(r.get('source_id'), 0),
            'code_id': i(r.get('code_id'), 0), 'cited': bo(r.get('cited')),
            'documentation_index': i(r.get('documentation_index'), 0),
            'internal_status': s(r.get('internal_status'), 50),
        }
        for r in bib_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(BibliographyLink, bib_incoming,
              ['source_id', 'code_id', 'cited', 'documentation_index', 'internal_status'])

    # 6. 活力 / 使用者 / 位置 / 语境 / 正字法 / 资源 / 媒体 ---------------
    vitality_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'endangerment_level': s(r.get('endangerment_level'), 100),
            'transmission_id': i(r.get('transmission_id')),
            'speaker_number_trends_id': i(r.get('speaker_number_trends_id')),
            'domains_of_use_id': i(r.get('domains_of_use_id')),
            'public_comment': s(r.get('public_comment')),
            'private_comment': s(r.get('private_comment')),
            'source_id': i(r.get('source_id')),
            'preferred': bo(r.get('preferred')),
        } for r in vitality_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(LanguageVitality, vitality_incoming,
              ['code_id', 'endangerment_level', 'transmission_id',
               'speaker_number_trends_id', 'domains_of_use_id', 'public_comment',
               'private_comment', 'source_id', 'preferred'])

    speaker_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'speaker_number': i(r.get('speaker_number')),
            'speaker_number_text': s(r.get('speaker_number_text')),
            'second_language_speakers': i(r.get('second_language_speakers')),
            'semi_speakers': i(r.get('semi_speakers')),
            'children': i(r.get('children')),
            'young_adults': i(r.get('young_adults')),
            'older_adults': i(r.get('older_adults')),
            'elders': i(r.get('elders')),
            'ethnic_population': i(r.get('ethnic_population')),
            'date_of_info': s(r.get('date_of_info'), 50),
            'public_comment': s(r.get('public_comment')),
            'private_comment': s(r.get('private_comment')),
            'source_id': i(r.get('source_id')),
            'preferred': bo(r.get('preferred')),
        } for r in speaker_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(SpeakerData, speaker_incoming,
              ['code_id', 'speaker_number', 'speaker_number_text',
               'second_language_speakers', 'semi_speakers', 'children',
               'young_adults', 'older_adults', 'elders', 'ethnic_population',
               'date_of_info', 'public_comment', 'private_comment',
               'source_id', 'preferred'])

    location_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'places': s(r.get('places'), 500),
            'coordinates': s(r.get('coordinates'), 500),
            'description': s(r.get('description')),
            'public_comment': s(r.get('public_comment')),
            'private_comment': s(r.get('private_comment'), 100),
            'source_id': i(r.get('source_id')),
            'preferred': bo(r.get('preferred')),
        } for r in location_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(LanguageLocationDetail, location_incoming,
              ['code_id', 'places', 'coordinates', 'description',
               'public_comment', 'private_comment', 'source_id', 'preferred'])

    context_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'government_support': s(r.get('government_support')),
            'institutional_support': s(r.get('institutional_support')),
            'speaker_attitude': s(r.get('speaker_attitude')),
            'other_languages_used': s(r.get('other_languages_used'), 255),
            'number_speaker_other_languages': s(r.get('number_speaker_other_languages'), 50),
            'domains_other_langs': s(r.get('domains_other_langs'), 255),
            'public_comment': s(r.get('public_comment')),
            'private_comment': s(r.get('private_comment')),
            'source_id': i(r.get('source_id')),
            'preferred': bo(r.get('preferred')),
        } for r in context_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(LanguageContext, context_incoming,
              ['code_id', 'government_support', 'institutional_support',
               'speaker_attitude', 'other_languages_used',
               'number_speaker_other_languages', 'domains_other_langs',
               'public_comment', 'private_comment', 'source_id', 'preferred'])

    orthography_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'standard_orthography': s(r.get('standard_orthography')),
            'scripts': s(r.get('scripts'), 150),
            'competing_orthographies': s(r.get('competing_orthographies'), 150),
            'public_comment': s(r.get('public_comment')),
            'private_comment': s(r.get('private_comment'), 100),
            'source_id': i(r.get('source_id')),
            'preferred': bo(r.get('preferred')),
        } for r in orthography_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(LanguageOrthography, orthography_incoming,
              ['code_id', 'standard_orthography', 'scripts',
               'competing_orthographies', 'public_comment', 'private_comment',
               'source_id', 'preferred'])

    resource_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'resource_type': s(r.get('resource_type'), 100),
            'description': s(r.get('description')),
            'url': s(r.get('url'), 500),
            'public_comment': s(r.get('public_comment')),
            'private_comment': s(r.get('private_comment')),
            'source_id': i(r.get('source_id')),
        } for r in resource_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(LanguageResource, resource_incoming,
              ['code_id', 'resource_type', 'description', 'url',
               'public_comment', 'private_comment', 'source_id'])

    media_incoming = {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'url': s(r.get('url'), 500), 'text': s(r.get('text'), 50),
        } for r in media_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }
    ctx.write(LanguageMedia, media_incoming, ['code_id', 'url', 'text'])

    # 7. 运行期数据：只 upsert，不裁剪 ------------------------------------
    ctx.write(ChangeSuggestion, {
        i(r['id']): {
            'id': i(r['id']), 'content_type_id': i(r.get('content_type_id')),
            'object_id': i(r.get('object_id')),
            'reason_for_change': s(r.get('reason_for_change'), 200),
            'topic': s(r.get('topic'), 200),
            'explanation': s(r.get('explanation')),
            'source': s(r.get('source'), 1000),
            'user_id': i(r.get('user_id')),
            'give_permission': bo(r.get('give_permission')),
        } for r in change_rows if i(r.get('id'))
    }, ['content_type_id', 'object_id', 'reason_for_change', 'topic',
        'explanation', 'source', 'user_id', 'give_permission'], prune=False)

    ctx.write(Notification, {
        i(r['id']): {'id': i(r['id']), 'activity_id': i(r.get('activity_id'), 0)}
        for r in notification_rows if i(r.get('id'))
    }, ['activity_id'], prune=False)

    ctx.write(NotificationUser, {
        i(r['id']): {'id': i(r['id']), 'notification_id': i(r.get('notification_id'), 0),
                     'user_id': i(r.get('user_id'), 0)}
        for r in notification_user_rows if i(r.get('id'))
    }, ['notification_id', 'user_id'], prune=False)

    ctx.write(LanguageDocumentation, {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'grammar_score_id': i(r.get('grammar_score_id')),
            'dictionary_score_id': i(r.get('dictionary_score_id')),
            'text_score_id': i(r.get('text_score_id')),
            'public_comment': s(r.get('public_comment')),
            'private_comment': s(r.get('private_comment')),
            'source_id': i(r.get('source_id')),
            'preferred': bo(r.get('preferred')),
            'documentation_index': i(r.get('documentation_index'), 0),
        } for r in documentation_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }, ['code_id', 'grammar_score_id', 'dictionary_score_id', 'text_score_id',
        'public_comment', 'private_comment', 'source_id', 'preferred',
        'documentation_index'])

    ctx.write(LanguageActivity, {
        i(r['id']): {
            'id': i(r['id']), 'date': aware_dt(r.get('date')),
            'code_id': i(r.get('code_id'), 0),
            'user_id': i(r.get('user_id')),
            'url': s(r.get('url'), 500),
            'obj_description': s(r.get('obj_description'), 500),
            'obj_type': s(r.get('obj_type'), 50),
            'level': i(r.get('level')),
            'type': i(r.get('type')),
            'obj_id': i(r.get('obj_id')),
            'content_type_id': i(r.get('content_type_id')),
            'object_id': i(r.get('object_id')),
        } for r in activity_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }, ['date', 'code_id', 'user_id', 'url', 'obj_description', 'obj_type',
        'level', 'type', 'obj_id', 'content_type_id', 'object_id'], prune=False)

    # 8. 样本 --------------------------------------------------------------
    ctx.write(SampleCategory, {
        i(r['id']): {'id': i(r['id']), 'name': s(r.get('name'), 100),
                     'description': s(r.get('description'))}
        for r in category_rows if i(r.get('id'))
    }, ['name', 'description'])

    ctx.write(LanguageSample, {
        i(r['id']): {
            'id': i(r['id']), 'code_id': i(r.get('code_id'), 0),
            'url': s(r.get('url'), 500),
            'title': s(r.get('title'), 500),
            'recorded_by': s(r.get('recorded_by'), 255),
            'date': as_date(r.get('date')),
            'description': s(r.get('description')),
            'location': s(r.get('location'), 500),
            'sample_type': s(r.get('type'), 10),
            'user_id': i(r.get('user_id')),
            'original_publication': s(r.get('original_publication')),
            'document_language': s(r.get('document_language'), 100),
            'guide_type': s(r.get('guide_type'), 100),
            'rating_votes': i(r.get('rating_votes'), 0),
            'rating_score': f(r.get('rating_score')),
            'thumbnail_img': s(r.get('thumbnail_img')),
            'youtube_video_embed': s(r.get('youtube_video_embed')),
            'contributor': s(r.get('contributor'), 255),
            'coverage': s(r.get('coverage')),
            'creator': s(r.get('creator')),
            'publisher': s(r.get('publisher')),
            'relation': s(r.get('relation')),
            'rights': s(r.get('rights')),
            'source': s(r.get('source')),
            'subject': s(r.get('subject')),
            'view_count': i(r.get('view_count'), 0),
            'transcription': s(r.get('transcription')),
            'featured': bo(r.get('featured')),
            'removed': bo(r.get('removed')),
            'added_at': aware_dt(r.get('added_at')),
            'updated_at': aware_dt(r.get('updated_at')),
        } for r in sample_rows if i(r.get('id')) and keep_code(r.get('code_id'))
    }, ['code_id', 'url', 'title', 'recorded_by', 'date', 'description',
        'location', 'sample_type', 'user_id', 'original_publication',
        'document_language', 'guide_type', 'rating_votes', 'rating_score',
        'thumbnail_img', 'youtube_video_embed', 'contributor', 'coverage',
        'creator', 'publisher', 'relation', 'rights', 'source', 'subject',
        'view_count', 'transcription', 'featured', 'removed', 'added_at',
        'updated_at'])

    skip_sample_ids = {i(r['id']) for r in raw_sample_rows
                       if i(r.get('id')) and i(r.get('code_id')) in skip_ids}
    ctx.write(SampleCategoryLink, {
        i(r['id']): {'id': i(r['id']), 'category_id': i(r.get('category_id'), 0),
                     'sample_id': i(r.get('sample_id'), 0)}
        for r in sample_category_rows
        if i(r.get('id')) and i(r.get('sample_id'), 0) not in skip_sample_ids
    }, ['category_id', 'sample_id'])

    ctx.write(SampleContributorLink, {
        i(r['id']): {'id': i(r['id']), 'sample_id': i(r.get('sample_id'), 0),
                     'contributor': s(r.get('contributor'), 255),
                     'role': s(r.get('role'), 100)}
        for r in sample_contributor_rows
        if i(r.get('id')) and i(r.get('sample_id'), 0) not in skip_sample_ids
    }, ['sample_id', 'contributor', 'role'])

    ctx.write(SampleTranslation, {
        i(r['id']): {'id': i(r['id']), 'sample_id': i(r.get('sample_id'), 0),
                     'name': s(r.get('name'), 255), 'url': s(r.get('url'), 1000)}
        for r in sample_translation_rows
        if i(r.get('id')) and i(r.get('sample_id'), 0) not in skip_sample_ids
    }, ['sample_id', 'name', 'url'])

    # 9. 兼容层：国家/趋势/传承 + 地图点 ----------------------------------
    _rebuild_compat_fields(vitality_rows, dry_run=ctx.dry_run)
    _rebuild_language_locations(location_rows, dry_run=ctx.dry_run)

    ctx.stats['skipped_tests'] = skipped_tests
    ctx.stats['source_rows_not_in_target'] = len(set(existing) - set(language_incoming))
    return ctx.stats


def _load_pycountry():
    try:
        import pycountry
        return pycountry
    except ImportError:
        return None


def _rebuild_score_levels(ctx, source):
    incoming = {}
    for score_type, table in [
            ('dictionary', 'language_dictionaryscore'),
            ('grammar', 'language_grammarscore'),
            ('text', 'language_textscore')]:
        for r in source.rows(table):
            key = (score_type, i(r.get('id')))
            if key[1]:
                incoming[key] = {'score_type': score_type, 'original_id': key[1],
                                 'score': i(r.get('score'), 0),
                                 'description': s(r.get('description'))}
    existing = {(obj.score_type, obj.original_id): obj for obj in ScoreLevel.objects.all()}
    created = updated = deleted = 0
    for key, payload in incoming.items():
        obj = existing.pop(key, None)
        if obj is None:
            created += 1
            if not ctx.dry_run:
                ScoreLevel.objects.create(**payload)
        elif any(getattr(obj, name) != payload[name] for name in payload):
            updated += 1
            if not ctx.dry_run:
                for name, value in payload.items():
                    setattr(obj, name, value)
                obj.save()
    deleted = len(existing)
    if not ctx.dry_run and existing:
        ScoreLevel.objects.filter(id__in=[obj.id for obj in existing.values()]).delete()
    ctx.stats['score_level'] = {'table': 'score_level', 'created': created,
                                'updated': updated, 'deleted': deleted,
                                'source_rows': len(incoming)}


def _rebuild_compat_fields(vitality_rows, dry_run=False):
    country_names = dict(Country.objects.values_list('id', 'name'))
    lang_countries = defaultdict(list)
    for m in LanguageCountry.objects.values('language_id', 'country_id'):
        name = country_names.get(m['country_id'])
        if name and name not in lang_countries[m['language_id']]:
            lang_countries[m['language_id']].append(name)

    transmission_levels = dict(Transmission.objects.values_list('id', 'level'))
    trend_levels = dict(SpeakerNumberTrend.objects.values_list('id', 'level'))

    preferred_vitality = {}
    for r in vitality_rows:
        cid = i(r.get('code_id'))
        if not cid:
            continue
        if cid not in preferred_vitality or bo(r.get('preferred')):
            preferred_vitality[cid] = r

    languages = list(Language.objects.all())
    changed = []
    for lang in languages:
        countries = lang_countries.get(lang.language_id, [])
        if countries:
            lang.country = ', '.join(countries[:3])[:255]
        v = preferred_vitality.get(lang.language_id)
        if v:
            trend = i(v.get('speaker_number_trends_id'))
            if trend in trend_levels:
                lang.speaker_trends = str(trend_levels[trend])[:10]
            trans = i(v.get('transmission_id'))
            if trans in transmission_levels:
                lang.intergenerational_transmission = str(transmission_levels[trans])[:4]
        changed.append(lang)
    if changed and not dry_run:
        Language.objects.bulk_update(
            changed,
            ['country', 'speaker_trends', 'intergenerational_transmission'],
            batch_size=1000)


def _rebuild_language_locations(location_rows, dry_run=False):
    names_map = dict(Language.objects.values_list('language_id', 'language_name'))
    location_best = {}
    for r in location_rows:
        cid = i(r.get('code_id'))
        coords = parse_coordinates(r.get('coordinates'))
        if not cid or cid not in names_map or not coords:
            continue
        if cid not in location_best or bo(r.get('preferred')):
            location_best[cid] = coords
    objects = [LanguageLocation(language_name=names_map.get(cid, ''),
                                longitude=lng, latitude=lat)
               for cid, (lat, lng) in location_best.items()]
    rebuild_derived(LanguageLocation, objects, dry_run=dry_run)


# ---------------------------------------------------------------------------
# enrich: names
# ---------------------------------------------------------------------------

def apply_names(ctx):
    """应用人工核实名称；未解决项写入显式标记，避免被自动补全回退。"""
    overrides = name_overrides()
    unresolved = unresolved_ids()
    tests = test_records()

    seen = set()
    updated = 0
    languages = {lang.language_id: lang for lang in Language.objects.all()}
    changed = []

    for lang_id, (name, source) in overrides.items():
        lang = languages.get(lang_id)
        if lang is None:
            continue
        seen.add(lang_id)
        if lang.language_name != name or lang.name_source != source:
            lang.language_name = name
            lang.name_source = source
            changed.append(lang)
            updated += 1

    for lang_id, reason in unresolved.items():
        lang = languages.get(lang_id)
        if lang is None:
            continue
        seen.add(lang_id)
        marker = f'manual:unresolved: {reason}'[:500]
        if lang.language_name != '' or lang.name_source != marker:
            lang.language_name = ''
            lang.name_source = marker
            changed.append(lang)
            updated += 1

    deleted = []
    for lang_id, record in tests.items():
        if not record.get('delete'):
            continue
        lang = languages.get(lang_id)
        if lang is not None:
            deleted.append(lang_id)
            if not ctx.dry_run:
                lang.delete()
    if changed and not ctx.dry_run:
        Language.objects.bulk_update(changed, ['language_name', 'name_source'],
                                     batch_size=1000)
    ctx.stats['names'] = {'updated': updated, 'deleted_test_records': deleted,
                          'configured': len(overrides), 'unresolved': len(unresolved)}
    return ctx.stats['names']


# ---------------------------------------------------------------------------
# enrich: backfill
# ---------------------------------------------------------------------------

def backfill_language_fields(ctx):
    """回填展示字段；legacy 只填空值，绝不覆盖人工核实结果。"""
    languages = list(Language.objects.all())

    raw_by_code = {}
    trans_by_code = {}
    for row in LanguageVitality.objects.order_by('-preferred', 'id').values(
            'code_id', 'endangerment_level', 'transmission_id'):
        cid = row['code_id']
        if not cid:
            continue
        if row['endangerment_level'] and cid not in raw_by_code:
            raw_by_code[cid] = row['endangerment_level']
        if row['transmission_id'] and cid not in trans_by_code:
            trans_by_code[cid] = row['transmission_id']

    transmission_levels = dict(Transmission.objects.values_list('id', 'level'))

    family_by_code = {}
    family_by_name = {}
    for row in LanguageCode.objects.exclude(classification='').exclude(
            classification__isnull=True).values('id', 'language_name', 'classification'):
        family_by_code[row['id']] = row['classification'].strip()
        name = (row['language_name'] or '').strip()
        if name:
            family_by_name.setdefault(name, row['classification'].strip())

    country_names = dict(Country.objects.values_list('id', 'name'))
    countries_by_code = defaultdict(list)
    for row in LanguageCountry.objects.order_by('id').values('language_id', 'country_id'):
        name = country_names.get(row['country_id'])
        if name and name not in countries_by_code[row['language_id']]:
            countries_by_code[row['language_id']].append(name)

    codes_by_name = defaultdict(list)
    for code_id, code_name in LanguageCode.objects.values_list('id', 'language_name'):
        name = (code_name or '').strip()
        if name:
            codes_by_name[name].append(code_id)

    code_rows = {
        row['id']: row for row in LanguageCode.objects.values(
            'id', 'classification')
    }

    def code_fields(code_id):
        code = code_rows.get(code_id) or {}
        return {
            'country': ', '.join(countries_by_code.get(code_id, [])[:3])[:255],
            'family': (code.get('classification') or '')[:255],
            'transmission': str(transmission_levels.get(trans_by_code.get(code_id), ''))[:4],
            'legacy': parse_raw_level(raw_by_code.get(code_id, '')),
            'level': '',
        }

    code_cache = {}

    def best_code(language_id, name):
        direct = code_fields(language_id)
        if any(direct.values()):
            return direct
        candidates = codes_by_name.get(name, [])
        if not candidates:
            return direct
        best = None
        for cid in candidates:
            if cid not in code_cache:
                code_cache[cid] = code_fields(cid)
            fields = code_cache[cid]
            score = sum(bool(v) for v in fields.values())
            if best is None or score > best[0]:
                best = (score, fields)
        return best[1] if best else direct

    name_best = {}
    for lang in languages:
        name = (lang.language_name or '').strip()
        if not name:
            continue
        fields = best_code(lang.language_id, name)
        fields = {
            'country': fields['country'] or (lang.country or ''),
            'family': fields['family'] or (lang.language_family or ''),
            'transmission': fields['transmission'] or
            str(transmission_levels.get(trans_by_code.get(lang.language_id), ''))[:4],
            'legacy': fields['legacy'] or parse_raw_level(raw_by_code.get(lang.language_id, '')) or
            SIMPLIFIED_LEVEL_MAP.get(lang.level_of_endangerment or '', ''),
            'level': fields['level'] or (lang.level_of_endangerment or ''),
        }
        score = sum(bool(v) for v in fields.values())
        current = name_best.get(name)
        if current is None or score > current['score'] or (
                score == current['score'] and lang.language_id <= 10000 < current['id']):
            name_best[name] = {'score': score, 'id': lang.language_id, **fields}

    stats = defaultdict(int)
    changed = []
    for lang in languages:
        name = (lang.language_name or '').strip()
        code = best_code(lang.language_id, name)
        fallback = name_best.get(name) if name else None
        if fallback and fallback['id'] == lang.language_id:
            fallback = None
        before = (lang.country, lang.language_family, lang.intergenerational_transmission,
                  lang.level_of_endangerment, lang.legacy_endangerment_level)

        if not (lang.country or '').strip():
            country = ', '.join(countries_by_code.get(lang.language_id, [])[:3])[:255]
            country = country or code['country'] or (fallback['country'] if fallback else '')
            if country:
                lang.country = country[:255]
                stats['country'] += 1

        if not (lang.language_family or '').strip():
            family = family_by_code.get(lang.language_id)
            family = family or code['family'] or (family_by_name.get(name) if name else '')
            if not family and fallback:
                family = fallback['family']
            if family:
                lang.language_family = family[:255]
                stats['family'] += 1

        if not (lang.intergenerational_transmission or '').strip():
            level = transmission_levels.get(trans_by_code.get(lang.language_id))
            transmission = str(level) if level is not None else ''
            transmission = transmission or code['transmission']
            if not transmission and fallback:
                transmission = fallback['transmission']
            if transmission:
                lang.intergenerational_transmission = transmission[:4]
                stats['transmission'] += 1

        if not (lang.level_of_endangerment or '').strip():
            level = code['level'] or (fallback['level'] if fallback else '')
            if level:
                lang.level_of_endangerment = level[:21]
                stats['level'] += 1

        # 关键修正：只有空值才回填，人工核实值永不被重算覆盖
        if not (lang.legacy_endangerment_level or '').strip():
            legacy = parse_raw_level(raw_by_code.get(lang.language_id, ''))
            legacy = legacy or code['legacy']
            if not legacy:
                legacy = SIMPLIFIED_LEVEL_MAP.get(lang.level_of_endangerment or '', '')
            if not legacy and fallback:
                legacy = fallback['legacy']
            if legacy:
                lang.legacy_endangerment_level = legacy[:30]
                stats['legacy'] += 1

        after = (lang.country, lang.language_family, lang.intergenerational_transmission,
                 lang.level_of_endangerment, lang.legacy_endangerment_level)
        if after != before:
            changed.append(lang)

    if changed and not ctx.dry_run:
        Language.objects.bulk_update(
            changed,
            ['country', 'language_family', 'intergenerational_transmission',
             'level_of_endangerment', 'legacy_endangerment_level'],
            batch_size=1000,
        )
    ctx.stats['backfill'] = {'scanned': len(languages), 'changed': len(changed), **stats}
    return ctx.stats['backfill']


# ---------------------------------------------------------------------------
# ml
# ---------------------------------------------------------------------------

def build_dataset(ctx):
    return _run_script(ctx, 'build_dataset.py', 'dataset')


def train_models(ctx):
    return _run_script(ctx, 'train_models.py', 'train')


def _run_script(ctx, name, stage):
    script = Path(settings.BASE_DIR) / 'ml' / name
    if not script.exists():
        raise RuntimeError(f'缺少脚本: {script}')
    if ctx.dry_run:
        return {'stage': stage, 'script': str(script), 'executed': False}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(Path(settings.BASE_DIR)),
        capture_output=True,
        text=True,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'{name} 执行失败（{result.returncode}）\n'
            f'STDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}')
    return {'stage': stage, 'script': str(script), 'executed': True,
            'output_tail': result.stdout[-1000:]}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def invalidate_caches():
    for key in ('chart_data_v2', 'chart_data'):
        cache.delete(key)
    try:
        from Dashboard.risk_service import reload_cache
        reload_cache()
    except Exception:
        pass
    try:
        from Dashboard.common import invalidate_language_stats_cache
        invalidate_language_stats_cache()
    except Exception:
        pass
