"""
Dashboard AI 功能视图

包含翻译、可视化分析、AI聊天等功能
"""

import logging
import time
from typing import Tuple, List

from django.shortcuts import render
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q
from django.views.decorators.csrf import csrf_protect

from Dashboard.models import Language, LEGACY_LEVEL_LABELS, ELCAT_LEVEL_LABELS
from User.models import ChatHistory, User
from Dashboard.common import (
    get_user_context, get_chart_data, get_openai_client, build_country_counter,
    display_language_name,
    ENDANGERMENT_LEVELS, CHAT_HISTORY_LIMIT, CHAT_RECENT_LIMIT,
    OPENAI_MODEL, OPENAI_TEMPERATURE,
)
from Dashboard.constants import (
    ELCAT_SIMPLE_LEVEL_KEYS, EXCLUDED_MAJOR_LANGUAGES,
    KNOWN_LEGACY_LEVEL_KEYS, MAP_SNAPSHOT_CACHE_KEY, MAP_SNAPSHOT_TTL,
    UNESCO_LEVEL_KEYS,
)
from language.utils import login_required_redirect, sanitize_ai_output, rate_limit

logger = logging.getLogger(__name__)

# AI API 重试配置
AI_MAX_RETRIES = 3
AI_RETRY_DELAY = 1.0  # 初始重试延迟（秒）


def _retry_on_transient_error(func, *args, max_retries=AI_MAX_RETRIES, **kwargs):
    """
    对 AI API 调用进行指数退避重试。
    仅对网络超时、限频等临时错误重试，认证错误等直接抛出。
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_msg = str(e).lower()
            # 认证错误不重试
            if any(kw in err_msg for kw in ('api', 'key', 'auth', 'invalid')):
                raise
            # 限频或超时则重试
            if any(kw in err_msg for kw in ('timeout', 'rate', 'quota', '529', '503', '502')):
                delay = AI_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"AI API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}，{delay}s 后重试")
                time.sleep(delay)
                continue
            raise
    raise last_exc


@rate_limit(max_requests=10, window_seconds=60)
@login_required_redirect
def language_translation(request):
    """
    语言翻译视图 - 使用 AI 进行濒危语言翻译（支持联网搜索、翻译记忆）

    URL: /Dashboard/language_translation/
    Methods: GET, POST
    """
    from Dashboard.models import TranslationMemory, Glossary
    from django.db.models import Q

    user, name, _ = get_user_context(request)
    result = ''
    memory_match = None
    glossary_terms = []

    if request.method == 'POST':
        language_text = request.POST.get('language_text', '').strip()

        if len(language_text) > 5000:
            messages.error(request, "单次翻译内容不能超过 5000 个字符")
            return render(request, 'language_translation.html', {
                'name': name,
                'username': user.username,
                'result': '',
                'memory_match': None,
                'glossary_terms': []
            })

        if not language_text:
            messages.warning(request, "请输入需要翻译的内容")
            return render(request, 'language_translation.html', {
                'name': name,
                'username': user.username,
                'result': '',
                'memory_match': None,
                'glossary_terms': []
            })

        # 1. 先查询翻译记忆库（精确匹配）
        memory_match = TranslationMemory.objects.filter(
            source_text=language_text, username=user.username
        ).first()

        if memory_match:
            # 更新使用次数
            memory_match.usage_count += 1
            memory_match.save()
            result = memory_match.translated_text
            logger.info(f"用户 {user.username} 使用翻译记忆匹配")
        else:
            # 2. 查询术语表，获取相关术语
            glossary_terms = Glossary.objects.filter(
                Q(term__icontains=language_text) | Q(translation__icontains=language_text),
                is_approved=True
            )[:5]

            # 3. 使用 AI 翻译
            try:
                # 构建包含术语的提示
                glossary_context = ""
                if glossary_terms.exists():
                    glossary_context = "\n\n参考术语表：\n"
                    for term in glossary_terms:
                        glossary_context += f"- {term.term} → {term.translation}\n"

                system_prompt = f"你是一个专业的濒危语言翻译助手，将以下文本翻译成中文。如果涉及专业术语或需要查证的内容，可以联网搜索相关信息。{glossary_context}"
                result = chat_with_web_search(language_text, [], system_prompt)
                logger.info(f"用户 {user.username} 使用 AI 翻译功能")

                # 4. 保存到翻译记忆库
                TranslationMemory.objects.create(
                    source_text=language_text,
                    translated_text=result,
                    source_language='auto',
                    target_language='zh',
                    username=user.username if user else None
                )

            except ValueError as ve:
                messages.error(request, str(ve))
                result = ""
            except Exception as e:
                logger.error(f"翻译请求失败: {str(e)}")
                messages.error(request, "翻译服务暂时不可用，请稍后重试")
                result = ""

    # 获取最近的翻译记录
    recent_translations = TranslationMemory.objects.filter(
        username=user.username if user else None
    ).order_by('-created_at')[:10]

    # 获取常用术语
    popular_glossary = Glossary.objects.filter(
        is_approved=True
    ).order_by('-id')[:20]

    return render(request, 'language_translation.html', {
        'name': name,
        'username': user.username,
        'result': result,
        'memory_match': memory_match,
        'glossary_terms': glossary_terms,
        'recent_translations': recent_translations,
        'popular_glossary': popular_glossary,
    })


@rate_limit(max_requests=8, window_seconds=60)
def language_visualization(request):
    """
    语言数据洞察 - 用真实数据讲述濒危语言的故事

    URL: /Dashboard/language_visualization/
    Methods: GET, POST (AJAX AI 解读)
    """
    from collections import Counter, defaultdict
    from django.db.models import Count, Sum, Q
    from Dashboard.models import (
        LanguageVitality, Transmission, SpeakerNumberTrend, DomainOfUse,
        SpeakerData, BibliographyLink, LanguageSource, LanguageOrthography,
        LanguageLocationDetail, LanguageContext, LanguageSample,
        Region, CountryRegion, LanguageCountry,
    )
    from Dashboard.sample_views import SAMPLE_TYPES, sample_media

    user, name, _ = get_user_context(request)
    username = user.username if user else ''
    result = ""

    if request.method == 'POST':
        if not user:
            return JsonResponse({'status': 'error', 'message': '请先登录后再使用 AI 解读'}, status=401)

        language_text = request.POST.get('language_text', '').strip()
        if len(language_text) > 5000:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': '分析文本不能超过 5000 个字符'})
            messages.warning(request, "分析文本不能超过 5000 个字符")
        elif not language_text:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': '请输入需要分析的文本'})
            messages.warning(request, "请输入需要分析的文本")
        else:
            try:
                system_prompt = "你是一个专业的濒危语言分析助手，请根据以下文本，结合真实的数据进行合理的分析。请尽量给出具体的数字和百分比,使用具体数据给出回答。如果需要实时数据或最新信息，可以联网搜索。"
                result = chat_with_web_search(language_text, [], system_prompt)
                result = sanitize_ai_output(result)
                logger.info(f"用户 {username} 使用可视化分析功能")

                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'status': 'success', 'answer': result})
            except ValueError as ve:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'status': 'error', 'message': str(ve)})
                messages.error(request, str(ve))
                result = ""
            except Exception as e:
                logger.error(f"分析请求失败: {str(e)}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'status': 'error', 'message': '分析服务暂时不可用，请稍后重试'})
                messages.error(request, "分析服务暂时不可用，请稍后重试")

    chart_data = get_chart_data()

    # 排除主流大语种，聚焦濒危语言（共享常量，见 Dashboard.constants）
    total_languages = Language.objects.count()
    named_languages = Language.objects.exclude(language_name='').exclude(
        language_name__startswith='未命名语言').count()
    focus = Language.objects.exclude(language_name='').exclude(
        language_name__startswith='未命名语言'
    ).exclude(language_name__in=EXCLUDED_MAJOR_LANGUAGES)

    # 统一使用 UNESCO 五级，避免首页/洞察页出现两套分级口径
    level_order = [(key, LEGACY_LEVEL_LABELS[key]) for key in UNESCO_LEVEL_KEYS]
    level_data = [{'key': key, 'name': label, 'value': chart_data.get(key, 0)}
                  for key, label in level_order]
    level_data.append({'key': 'awakening', 'name': '复苏',
                       'value': chart_data.get('legacy_awakening', 0)})
    level_data.append({'key': 'safe', 'name': '安全',
                       'value': chart_data.get('legacy_safe', 0)})
    known_total = sum(item['value'] for item in level_data)
    unknown_total = max(total_languages - known_total, 0)
    if unknown_total:
        level_data.append({'key': 'unknown', 'name': '未分类', 'value': unknown_total})
    level_total = total_languages or 1
    for item in level_data:
        item['pct'] = round(item['value'] / level_total * 100, 1)

    # 使用者最多 / 最后的声音
    top_speakers = list(focus.exclude(number_of_speakers__isnull=True).filter(
        number_of_speakers__gt=0
    ).order_by('-number_of_speakers').values(
        'language_id', 'language_name', 'number_of_speakers',
        'level_of_endangerment', 'country'
    )[:12])

    last_speakers = list(focus.filter(
        number_of_speakers__gt=0, number_of_speakers__lte=10
    ).order_by('number_of_speakers').values(
        'language_id', 'language_name', 'number_of_speakers',
        'level_of_endangerment', 'country'
    )[:12])

    awakening = list(Language.objects.filter(
        level_of_endangerment='awakening'
    ).exclude(language_name='').order_by('language_name').values(
        'language_id', 'language_name', 'number_of_speakers', 'country'
    )[:12])
    dormant_count = Language.objects.filter(level_of_endangerment='dormant').count()

    # 语系版图
    family_stats = list(Language.objects.exclude(language_name='').filter(
        language_family__isnull=False
    ).exclude(
        language_family=''
    ).values('language_family').annotate(
        count=Count('language_id')
    ).order_by('-count')[:10])
    family_data = [{'name': f['language_family'][:30], 'full': f['language_family'],
                    'value': f['count']} for f in family_stats]

    # 国家分布（Language.country 可能是逗号分隔的多国）
    country_counter = build_country_counter()
    country_data = [{'name': name_, 'value': value}
                    for name_, value in country_counter.most_common(10)]

    # 大区分布（按 country_region + language_country_map 聚合）
    country_regions = defaultdict(list)
    for cr in CountryRegion.objects.values('country_id', 'region_id'):
        country_regions[cr['country_id']].append(cr['region_id'])
    region_names = dict(Region.objects.values_list('id', 'name'))
    region_langs = defaultdict(set)
    for m in LanguageCountry.objects.values('language_id', 'country_id'):
        for region_id in country_regions.get(m['country_id'], []):
            region_langs[region_id].add(m['language_id'])
    region_data = [
        {'name': region_names.get(region_id, f'大区 {region_id}'),
         'value': len(language_ids)}
        for region_id, language_ids in region_langs.items()
    ]
    region_data.sort(key=lambda item: item['value'], reverse=True)

    # 使用者人口结构
    demo_agg = SpeakerData.objects.aggregate(
        children=Sum('children'), young=Sum('young_adults'),
        older=Sum('older_adults'), elders=Sum('elders'),
        ethnic=Sum('ethnic_population'))
    demo_items = [
        ('儿童', demo_agg['children'] or 0),
        ('青年', demo_agg['young'] or 0),
        ('老年', demo_agg['older'] or 0),
        ('长老', demo_agg['elders'] or 0),
    ]
    demo_max = max([value for _, value in demo_items] or [1]) or 1
    speaker_data = [{'name': label, 'value': value,
                     'pct': round(value / demo_max * 100, 1)}
                    for label, value in demo_items]
    speaker_numeric_count = SpeakerData.objects.filter(
        speaker_number__isnull=False).count()

    # 活力的三个维度
    def build_dimension(model, field, title):
        counts = Counter(LanguageVitality.objects.exclude(
            **{field: None}).values_list(field, flat=True))
        rows = []
        for obj in model.objects.all().order_by('id'):
            count = counts.get(obj.id, 0)
            rows.append({
                'id': obj.id,
                'level': obj.level,
                'description': obj.description,
                'count': count,
            })
        max_count = max([row['count'] for row in rows] or [1]) or 1
        for row in rows:
            row['pct'] = round(row['count'] / max_count * 100, 1)
        return {'title': title, 'rows': rows}

    vitality_dimensions = [
        build_dimension(Transmission, 'transmission_id', '代际传承'),
        build_dimension(SpeakerNumberTrend, 'speaker_number_trends_id', '使用者趋势'),
        build_dimension(DomainOfUse, 'domains_of_use_id', '使用领域'),
    ]

    # 文献关联（只统计真实来源，排除占位；只保留 language 表中存在的语言）
    valid_language_ids = set(Language.objects.values_list('language_id', flat=True))
    real_source_ids = set(LanguageSource.objects.filter(
        is_placeholder=False).values_list('id', flat=True))
    bib_counter = Counter(
        code_id for code_id in BibliographyLink.objects.filter(
            source_id__in=real_source_ids).values_list('code_id', flat=True)
        if code_id in valid_language_ids
    )
    name_map = dict(Language.objects.values_list('language_id', 'language_name'))
    top_bib = [{'id': code_id,
                'name': display_language_name({'language_id': code_id,
                                               'language_name': name_map.get(code_id)}),
                'value': count} for code_id, count in bib_counter.most_common(10)]
    bib_data = [{'name': item['name'][:30], 'value': item['value']}
                for item in top_bib]
    bib_total = BibliographyLink.objects.count()
    bib_real_links = sum(bib_counter.values())

    # 数据覆盖度（只统计 language 表中真实存在的语言 ID）
    orthography_rows = list(LanguageOrthography.objects.values(
        'code_id', 'scripts', 'public_comment', 'competing_orthographies',
        'standard_orthography'))
    orth_langs = len({
        row['code_id'] for row in orthography_rows
        if row['code_id'] in valid_language_ids
        and any([row['scripts'], row['public_comment'],
                 row['competing_orthographies'], row['standard_orthography']])
    })
    location_langs = len(set(
        LanguageLocationDetail.objects.values_list('code_id', flat=True)
    ) & valid_language_ids)
    speaker_langs = len(set(
        SpeakerData.objects.filter(
            speaker_number__isnull=False).values_list('code_id', flat=True)
    ) & valid_language_ids)
    bib_langs = len(set(
        BibliographyLink.objects.values_list('code_id', flat=True)
    ) & valid_language_ids)
    coverage = [
        ('名称', named_languages),
        ('国家', Language.objects.filter(
            country__isnull=False).exclude(country='').count()),
        ('地理坐标', location_langs),
        ('使用者数据', speaker_langs),
        ('文献关联', bib_langs),
        ('正字法', orth_langs),
    ]
    coverage_data = [{
        'label': label,
        'value': value,
        'total': total_languages,
        'pct': round(value / total_languages * 100, 1) if total_languages else 0,
    } for label, value in coverage]

    # 语言故事：挑选数据较完整的语言
    story_ids = [2089, 4570, 1135, 2088, 3943, 4612]
    stories = []
    for story_id in story_ids:
        lang = Language.objects.filter(language_id=story_id).first()
        if not lang:
            continue
        vitality = LanguageVitality.objects.filter(
            code_id=story_id).order_by('-preferred', 'id').first()
        speaker = SpeakerData.objects.filter(
            code_id=story_id).order_by('-preferred', 'id').first()
        context = LanguageContext.objects.filter(
            code_id=story_id).order_by('-preferred', 'id').first()
        stories.append({
            'id': story_id,
            'name': lang.language_name,
            'level_key': lang.level_of_endangerment or 'unknown',
            'level_label': ENDANGERMENT_LEVELS.get(lang.level_of_endangerment, '未知'),
            'country': lang.country or '',
            'speakers': lang.number_of_speakers,
            'speaker_text': speaker.speaker_number_text if speaker else '',
            'vitality': (vitality.endangerment_level[:110]
                         if vitality and vitality.endangerment_level else ''),
            'context': (context.public_comment[:110]
                        if context and context.public_comment else ''),
            'source': lang.name_source or '',
        })

    # 语言样本：视频、音频、词汇、语法、词典、图片
    sample_qs = LanguageSample.objects.filter(
        removed=False).exclude(title='')
    sample_total = sample_qs.count()
    sample_type_counts = Counter(dict(
        sample_qs.values('sample_type').annotate(count=Count('id'))
        .values_list('sample_type', 'count')
    ))
    sample_type_stats = [
        {'key': key, 'label': label, 'count': sample_type_counts.get(key, 0)}
        for key, label in SAMPLE_TYPES.items()
    ]
    featured_samples = list(sample_qs.order_by(
        '-featured', '-view_count', '-id')[:6])
    for sample in featured_samples:
        sample.language_name = display_language_name({
            'language_id': sample.code_id,
            'language_name': name_map.get(sample.code_id),
        })
        sample.thumbnail, sample.play_url = sample_media(sample)
        sample.type_label = SAMPLE_TYPES.get(sample.sample_type, '其他')

    stats = {
        'total': total_languages,
        'named': named_languages,
        'unsafe': chart_data.get('unsafe', 0),
        'danger': chart_data.get('danger', 0),
        'endangered': chart_data.get('endangered', 0),
        'seriously_endangered': chart_data.get('seriously_endangered', 0),
        'extinct': chart_data.get('extinct', 0),
        'awakening': chart_data.get('legacy_awakening', 0),
        'vitality': LanguageVitality.objects.count(),
        'bibliography': bib_total,
        'bibliography_real': bib_real_links,
        'orthography': LanguageOrthography.objects.count(),
        'countries': len(country_counter),
        'speaker_records': SpeakerData.objects.count(),
        'speaker_numeric': speaker_numeric_count,
        'locations': LanguageLocationDetail.objects.count(),
        'samples': sample_total,
        'sample_videos': sample_type_counts.get('V', 0),
        'sample_audio': sample_type_counts.get('A', 0),
        'sample_lexical': sample_type_counts.get('L', 0),
    }

    context = {
        'data': chart_data,
        'stats': stats,
        'level_data': level_data,
        'level_total': level_total,
        'top_speakers': top_speakers,
        'last_speakers': last_speakers,
        'awakening': awakening,
        'dormant_count': dormant_count,
        'family_data': family_data,
        'country_data': country_data,
        'region_data': region_data,
        'speaker_data': speaker_data,
        'vitality_dimensions': vitality_dimensions,
        'bib_data': bib_data,
        'top_bib': top_bib,
        'coverage_data': coverage_data,
        'stories': stories,
        'featured_samples': featured_samples,
        'sample_type_stats': sample_type_stats,
        'sample_total': sample_total,
        'name': name,
        'username': username,
        'result': result,
        'endangerment_levels': ENDANGERMENT_LEVELS,
    }

    return render(request, 'language_visualization.html', context)


# =============================================================================
# Kimi 联网搜索工具定义
# =============================================================================

def get_web_search_tools():
    """
    获取 Kimi 联网搜索工具定义
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web-search",
                "description": "搜索互联网获取最新信息。当问题涉及实时数据、最新新闻、当前事件或需要最新信息时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词，应简洁明了，包含关键信息"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]


def execute_web_search(query: str) -> str:
    """
    执行联网搜索

    Args:
        query: 搜索关键词

    Returns:
        str: 搜索结果
    """
    from Dashboard.common import get_openai_client

    client = get_openai_client()

    try:
        def _call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": f"请搜索以下内容并总结关键信息，返回简洁的中文摘要（100字以内）：{query}"}
                ],
                temperature=OPENAI_TEMPERATURE,
            )
        response = _retry_on_transient_error(_call)
        return response.choices[0].message.content if response.choices else "搜索无结果"
    except Exception as e:
        logger.error(f"联网搜索执行失败: {str(e)}")
        return f"搜索服务暂时不可用: {str(e)}"


def chat_with_web_search(query: str, history: List[dict], system_prompt: str = None) -> str:
    """
    带联网搜索功能的 AI 对话助手（基于 Kimi Function Calling）

    Args:
        query: 用户输入的问题
        history: 对话历史记录
        system_prompt: 系统提示词（可选）

    Returns:
        str: AI 的回复内容
    """
    from Dashboard.common import get_openai_client, OPENAI_API_KEY

    if not OPENAI_API_KEY:
        raise ValueError(
            "AI 服务 API 密钥未配置！请在项目根目录的 .env 文件中设置 OPENAI_API_KEY "
            "(申请地址：https://platform.moonshot.cn/console/api-keys)，然后重启服务。"
        )

    # 构建消息
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    try:
        client = get_openai_client()
        tools = get_web_search_tools()

        # 第一轮：让 Kimi 决定是否需要联网搜索
        completion = _retry_on_transient_error(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools,
            temperature=OPENAI_TEMPERATURE,
        )

        message = completion.choices[0].message

        # 检查是否需要调用工具（联网搜索）
        if message.tool_calls:
            # 处理工具调用
            messages.append(message)

            for tool_call in message.tool_calls:
                if tool_call.function.name == "web-search":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    search_query = args.get("query", "")

                    # 执行搜索
                    search_result = execute_web_search(search_query)

                    # 将搜索结果返回给 Kimi
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_result
                    })

            # 第二轮：用搜索结果继续对话
            completion = _retry_on_transient_error(
                client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
            )
            result = completion.choices[0].message.content if completion.choices and completion.choices[0].message.content else "回答失败！"
        else:
            result = message.content if message.content else "回答失败！"

    except Exception as e:
        error_msg = str(e)
        logger.error(f"AI 对话请求失败: {error_msg}")
        if "api" in error_msg.lower() or "key" in error_msg.lower() or "auth" in error_msg.lower():
            raise ValueError("AI 服务认证失败，请检查 API 配置")
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            raise ValueError("AI 服务响应超时，请稍后重试")
        elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
            raise ValueError("AI 服务请求频率超限，请稍后重试")
        else:
            raise

    return result


# =============================================================================
# AI 聊天功能（支持联网搜索）
# =============================================================================

def chat(query: str, history: List[dict]) -> str:
    """
    AI 聊天助手核心函数（基于 Kimi 联网搜索）

    Args:
        query: 用户输入的问题
        history: 对话历史记录（包含之前的消息）

    Returns:
        str: AI 的回复内容
    """
    system_prompt = "你是一个濒危语言方面的顶尖学者，你也是一个优秀的老师，你能够用简洁易懂的语言去教授学生如何高效快速的学习一门濒危语言。请回答以下问题。如果问题与语言无关，请礼貌说明。回答时可以联网搜索最新信息来提供更准确的答案。"

    return chat_with_web_search(query, history, system_prompt)


def get_user_chat_context(username: str) -> Tuple[List, List, List, any]:
    """
    获取用户聊天历史上下文（用于构建 AI 对话历史）

    Args:
        username: 用户名

    Returns:
        Tuple: (questions, answers, times, user_obj)
    """
    from User.models import User as LocalUser

    questions = []
    answers = []
    times = []
    user_obj = None

    if username:
        try:
            user_obj = LocalUser.objects.get(username=username)
            chat_history = ChatHistory.objects.filter(
                username=username
            ).order_by('-timestamp')[:CHAT_HISTORY_LIMIT]

            for item in chat_history:
                times.append(item.timestamp)
                questions.append(item.question)
                answers.append(sanitize_ai_output(item.answer or ''))
        except LocalUser.DoesNotExist:
            pass

    return questions, answers, times, user_obj


def build_chat_history(questions: List[str], answers: List[str]) -> List[dict]:
    """
    从问题/回答列表构建 AI 对话历史格式

    Args:
        questions: 问题列表（按时间正序）
        answers: 回答列表（按时间正序）

    Returns:
        List[dict]: AI 对话历史消息列表
    """
    history = []
    for q, a in zip(questions, answers):
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": a})
    return history


@rate_limit(max_requests=10, window_seconds=60)
@login_required_redirect
def language_chat(request):
    """
    智能问答学习助手视图

    URL: /Dashboard/language_chat/
    Methods: GET, POST (AJAX)
    """
    from django.http import JsonResponse

    username = request.session.get('username', '')
    result = ''
    language_text = ''

    user, name, _ = get_user_context(request)
    questions, answers, times, user_obj = get_user_chat_context(username)

    if request.method == 'POST':
        language_text = request.POST.get('language_text', '').strip()

        if len(language_text) > 5000:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': '单个问题不能超过 5000 个字符'})
            messages.error(request, '单个问题不能超过 5000 个字符')
        elif not language_text:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': '请输入您的问题'})
            messages.warning(request, "请输入您的问题")
        elif not user_obj:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': '用户会话异常，请重新登录'})
            messages.error(request, "用户会话异常，请重新登录")
        else:
            try:
                # 构建带历史记录的对话上下文
                history = [{
                    "role": "system",
                    "content": "你是一个濒危语言方面的顶尖学者，你也是一个优秀的老师，你能够用简洁易懂的语言去教授学生如何高效快速的学习一门濒危语言。请回答以下问题。如果问题与语言无关，请礼貌说明。"
                }]

                # 添加最近的对话历史（保留最近10轮）
                recent_q = list(reversed(questions[:CHAT_RECENT_LIMIT]))
                recent_a = list(reversed(answers[:CHAT_RECENT_LIMIT]))
                for q, a in zip(recent_q, recent_a):
                    history.append({"role": "user", "content": q})
                    history.append({"role": "assistant", "content": a})

                result = chat(language_text, history)

                ChatHistory.objects.create(
                    username=user_obj,
                    question=language_text,
                    answer=result
                )

                # AJAX 请求返回 JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'status': 'success',
                        'answer': sanitize_ai_output(result),
                        'time': timezone.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                times.insert(0, timezone.now())
                questions.insert(0, language_text)
                answers.insert(0, sanitize_ai_output(result))
                result = sanitize_ai_output(result)
                logger.info(f"用户 {username} 使用聊天助手功能")
            except ValueError as ve:
                error_message = str(ve)
                logger.error(f"聊天处理失败: {error_message}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'status': 'error', 'message': error_message})
                messages.error(request, error_message)
            except Exception as e:
                logger.error(f"聊天处理失败: {str(e)}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'status': 'error', 'message': '处理请求时发生错误，请稍后重试'})
                messages.error(request, "处理请求时发生错误，请稍后重试")

    context = {
        'username': username,
        'name': name,
        'result': result,
        'questions': questions,
        'answers': answers,
        'times': times,
        'chat_history': zip(times, questions, answers),
        'length': len(times),
        'list_length': range(len(times)),
        'language_text': language_text
    }

    return render(request, 'language_chat.html', context)


def get_chat_history(request):
    """获取指定时间戳之前的聊天历史（用于切换会话）"""
    from django.http import JsonResponse

    username = request.session.get('username', '')
    if not username:
        return JsonResponse({'status': 'error', 'message': '请先登录'}, status=401)

    timestamp_str = request.GET.get('before', '')
    if not timestamp_str:
        return JsonResponse({'status': 'error', 'message': '缺少时间参数'})

    try:
        before_time = timezone.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        return JsonResponse({'status': 'error', 'message': '无效的时间格式'})

    try:
        before_time = timezone.make_aware(before_time)
    except Exception:
        pass

    # 获取该时间及之后的记录（会话内的完整对话）
    history = ChatHistory.objects.filter(
        username=username,
        timestamp__gte=before_time
    ).order_by('timestamp')[:CHAT_HISTORY_LIMIT]

    messages_data = []
    for item in history:
        messages_data.append({
            'question': item.question,
            'answer': sanitize_ai_output(item.answer or ''),
            'time': item.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    return JsonResponse({
        'status': 'success',
        'messages': messages_data
    })


def _parse_coordinates(coordinates: str):
    """解析 location 明细中的坐标字符串，返回 (lng, lat)，无效时返回 (None, None)。"""
    first = (coordinates or '').split(';')[0].strip()
    parts = [part.strip() for part in first.split(',')]
    if len(parts) >= 2:
        try:
            return float(parts[1]), float(parts[0])
        except ValueError:
            return None, None
    return None, None


def _build_map_snapshot():
    """构建地图全量快照：{'points': [...], 'stats': {...}, 'total': int}。

    只取地图渲染必需字段，位置明细用迭代器避免一次性缓存大结果集。
    点 dict 带 ``id``，供前端把视口返回的副本映射回全局列表。
    """
    from django.db.models import Count

    from Dashboard.models import Language, LanguageLocation, LanguageLocationDetail
    from Dashboard.risk_service import get_risk_profile

    language_rows = Language.objects.values(
        'language_id', 'language_name', 'number_of_speakers',
        'country', 'language_family', 'legacy_endangerment_level',
    )
    total_languages = language_rows.count()

    detail_rows = LanguageLocationDetail.objects.exclude(
        coordinates=''
    ).order_by('-preferred', 'id').values_list('code_id', 'coordinates')
    coordinates_by_code = {}
    for code_id, coordinates in detail_rows.iterator():
        coordinates_by_code.setdefault(code_id, coordinates)

    # 兼容层位置：按 language_name 匹配，作为兜底
    legacy_rows = LanguageLocation.objects.exclude(
        language_name__isnull=True
    ).exclude(language_name='').values_list(
        'language_name', 'longitude', 'latitude')
    legacy_by_name = {}
    for language_name, longitude, latitude in legacy_rows.iterator():
        legacy_by_name[language_name] = (longitude, latitude)

    points = []
    for lang in language_rows.iterator():
        lng, lat = _parse_coordinates(coordinates_by_code.get(lang['language_id']))
        if lng is None:
            legacy = legacy_by_name.get(lang['language_name'])
            if legacy and legacy[0] and legacy[1]:
                try:
                    lng = float(legacy[0])
                    lat = float(legacy[1])
                except (ValueError, TypeError):
                    lng = lat = None
        if lng is not None and lat is not None:
            # 本系统 UNESCO 五级作为地图主分类，ELCat 原始分级保留在详情里
            legacy_key = lang['legacy_endangerment_level'] or 'unknown'
            level_cn = LEGACY_LEVEL_LABELS.get(legacy_key, '未分类')
            risk = get_risk_profile(lang['language_id'])
            points.append({
                'id': lang['language_id'],
                'name': display_language_name(lang),
                'longitude': lng,
                'latitude': lat,
                'endangerment': level_cn,
                'speakers': lang['number_of_speakers'] or 0,
                'region': lang['country'] or '未知',
                'family': lang['language_family'] or '未知',
                'riskScore': risk['risk_score'] if risk else None,
                'riskLabel': risk['risk_band_label'] if risk else '暂无预测',
            })

    # 濒危等级统计：两条分组聚合替代 11 条 COUNT 查询
    legacy_counts = dict(Language.objects.values_list(
        'legacy_endangerment_level'
    ).annotate(count=Count('language_id')))
    elcat_counts = dict(Language.objects.values_list(
        'level_of_endangerment'
    ).annotate(count=Count('language_id')))

    stats = {
        key: legacy_counts.get(key, 0) for key in UNESCO_LEVEL_KEYS
    }
    stats['awakening'] = legacy_counts.get('awakening', 0)
    # “未分类”口径保持不变：总记录数减去全部已知 legacy 分级（含 NULL 行）
    stats['unknown'] = total_languages - sum(
        legacy_counts.get(key, 0) for key in KNOWN_LEGACY_LEVEL_KEYS)
    for key in ELCAT_SIMPLE_LEVEL_KEYS:
        stats[key] = elcat_counts.get(key, 0)

    return {'points': points, 'stats': stats, 'total': total_languages}


def _get_map_snapshot():
    """读取地图快照缓存，未命中时构建并写入（TTL 与统计缓存一致）。

    语言/位置明细写入后由 signals 调用 invalidate_language_stats_cache 清空。
    """
    from django.core.cache import cache

    cached = cache.get(MAP_SNAPSHOT_CACHE_KEY)
    # 结构校验：旧版本/损坏缓存视为未命中，避免前端拿到不完整数据
    if (isinstance(cached, dict) and 'points' in cached
            and 'stats' in cached and 'total' in cached):
        return cached

    snapshot = _build_map_snapshot()
    cache.set(MAP_SNAPSHOT_CACHE_KEY, snapshot, MAP_SNAPSHOT_TTL)
    return snapshot


def _parse_bbox(raw):
    """解析 'minLng,minLat,maxLng,maxLat'，非法时抛 ValueError。"""
    parts = [part.strip() for part in str(raw).split(',')]
    if len(parts) != 4:
        raise ValueError('bbox 需要 4 个数值：minLng,minLat,maxLng,maxLat')
    try:
        min_lng, min_lat, max_lng, max_lat = (float(part) for part in parts)
    except ValueError:
        raise ValueError('bbox 含非数值')
    if not (-180 <= min_lng <= 180 and -180 <= max_lng <= 180):
        raise ValueError('bbox 经度需在 -180~180')
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError('bbox 纬度需在 -90~90')
    if min_lng > max_lng or min_lat > max_lat:
        raise ValueError('bbox 顺序错误：需 minLng,minLat,maxLng,maxLat')
    return min_lng, min_lat, max_lng, max_lat


def _resolve_level(raw):
    """把 legacy 键（如 endangered）或中文标签（如 濒危）统一为中文标签。"""
    value = str(raw or '').strip()
    if not value:
        return None
    if value in LEGACY_LEVEL_LABELS:
        return LEGACY_LEVEL_LABELS[value]
    if value == 'unknown':
        return '未分类'
    if value in set(LEGACY_LEVEL_LABELS.values()) or value == '未分类':
        return value
    raise ValueError('未知的 level：%s' % value)


def _filter_points(points, bbox=None, level=None):
    """按视口/分级裁剪快照点集，字段原样保留（含 riskScore/riskLabel）。"""
    if bbox is None and level is None:
        return points
    min_lng, min_lat, max_lng, max_lat = bbox if bbox else (None,) * 4
    return [
        point for point in points
        if (level is None or point['endangerment'] == level)
        and (bbox is None or (
            min_lng <= point['longitude'] <= max_lng
            and min_lat <= point['latitude'] <= max_lat))
    ]


def language_map(request):
    """
    互动式语言地图视图 - 展示濒危语言地理分布与濒危程度热力图

    URL: /Dashboard/language_map/
    Methods: GET

    Query:
        format=json            返回 JSON（与 bbox/level 一起使用时自动生效）
        bbox=minLng,minLat,maxLng,maxLat   只返回可视范围内的点
        level=endangered|濒危     只返回指定濒危等级
        mode=cluster|marker|heatmap       请求来源标记，原样回显
    """
    user, name, _ = get_user_context(request)
    snapshot = _get_map_snapshot()

    raw_bbox = (request.GET.get('bbox') or '').strip()
    raw_level = (request.GET.get('level') or '').strip()
    mode = (request.GET.get('mode') or '').strip()
    wants_json = request.GET.get('format') == 'json' or bool(raw_bbox) or bool(raw_level)

    if wants_json:
        try:
            bbox = _parse_bbox(raw_bbox) if raw_bbox else None
            level = _resolve_level(raw_level) if raw_level else None
        except ValueError as exc:
            return JsonResponse(
                {'status': 'error', 'message': str(exc)}, status=400,
                json_dumps_params={'ensure_ascii': False},
            )

        points = _filter_points(snapshot['points'], bbox=bbox, level=level)
        return JsonResponse({
            'status': 'success',
            'mode': mode or 'cluster',
            'points': points,
            'count': len(points),
            'total': snapshot['total'],
            'bbox': list(bbox) if bbox else None,
            'level': level,
        }, json_dumps_params={'ensure_ascii': False})

    # 默认行为保持不变：整页渲染仍使用全量点集，仅数据来源改为缓存快照
    context = {
        'name': name,
        'username': user.username if user else '',
        'map_data': snapshot['points'],
        'endangerment_stats': snapshot['stats'],
        'endangerment_levels': ENDANGERMENT_LEVELS,
        'total_languages': snapshot['total'],
    }

    return render(request, 'language_map.html', context)


@login_required_redirect
def translation_memory_list(request):
    """
    翻译记忆库列表视图

    URL: /Dashboard/translation_memory/
    Method: GET
    """
    from Dashboard.models import TranslationMemory

    user, name, _ = get_user_context(request)

    # 获取用户的翻译记忆
    memories = TranslationMemory.objects.filter(
        username=user.username
    ).order_by('-created_at')[:50]

    context = {
        'name': name,
        'username': user.username,
        'memories': memories,
    }

    return render(request, 'translation_memory.html', context)


@login_required_redirect
@csrf_protect
def translation_rate(request):
    """
    翻译评分 API

    URL: /Dashboard/translation/rate/
    Method: POST
    """
    from Dashboard.models import TranslationMemory

    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '无效的请求方法'})

    memory_id = request.POST.get('id')
    score = request.POST.get('score')

    try:
        score = int(score)
        if score < 1 or score > 5:
            raise ValueError
    except (ValueError, TypeError):
        return JsonResponse({'status': 'error', 'message': '评分必须在1-5之间'})

    user, name, _ = get_user_context(request)

    try:
        memory = TranslationMemory.objects.get(id=memory_id, username=user.username)
        memory.quality_score = score
        memory.save()
        return JsonResponse({'status': 'success', 'message': '评分成功'})
    except TranslationMemory.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '翻译记录不存在'})


@login_required_redirect
@csrf_protect
def glossary_manage(request):
    """
    术语表管理视图

    URL: /Dashboard/glossary/
    Methods: GET, POST
    """
    from Dashboard.models import Glossary

    user, name, is_admin = get_user_context(request)

    if request.method == 'POST':
        action = request.POST.get('action', '')

        if action == 'add':
            term = request.POST.get('term', '').strip()
            translation = request.POST.get('translation', '').strip()
            language = request.POST.get('language', '').strip()
            category = request.POST.get('category', '').strip()
            description = request.POST.get('description', '').strip()

            if not term or not translation:
                return JsonResponse({'status': 'error', 'message': '术语和翻译不能为空'})

            Glossary.objects.create(
                term=term,
                translation=translation,
                language=language,
                category=category,
                description=description,
                username=user.username,
                is_approved=is_admin == 1  # 管理员添加的自动审核
            )
            return JsonResponse({'status': 'success', 'message': '术语添加成功'})

        elif action == 'delete':
            glossary_id = request.POST.get('id')
            try:
                glossary = Glossary.objects.get(id=glossary_id)
                if glossary.username == user.username or is_admin == 1:
                    glossary.delete()
                    return JsonResponse({'status': 'success', 'message': '术语删除成功'})
                else:
                    return JsonResponse({'status': 'error', 'message': '权限不足'})
            except Glossary.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': '术语不存在'})

        elif action == 'approve':
            if is_admin != 1:
                return JsonResponse({'status': 'error', 'message': '权限不足'})
            glossary_id = request.POST.get('id')
            try:
                glossary = Glossary.objects.get(id=glossary_id)
                glossary.is_approved = True
                glossary.save()
                return JsonResponse({'status': 'success', 'message': '术语审核通过'})
            except Glossary.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': '术语不存在'})

    # GET 请求
    category_filter = request.GET.get('category', '')
    search = request.GET.get('search', '')

    glossaries = Glossary.objects.all()

    if category_filter:
        glossaries = glossaries.filter(category=category_filter)

    if search:
        glossaries = glossaries.filter(
            Q(term__icontains=search) | Q(translation__icontains=search)
        )

    # 获取所有分类
    categories = Glossary.objects.values_list('category', flat=True).distinct()

    context = {
        'name': name,
        'username': user.username,
        'is_admin': is_admin == 1,
        'glossaries': glossaries[:100],
        'categories': [c for c in categories if c],
        'category_filter': category_filter,
        'search': search,
    }

    return render(request, 'glossary.html', context)
