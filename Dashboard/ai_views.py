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
from django.views.decorators.csrf import csrf_protect

from Dashboard.models import Language
from User.models import ChatHistory, User
from Dashboard.common import get_user_context, get_chart_data, get_openai_client, ENDANGERMENT_LEVELS, CHAT_HISTORY_LIMIT, CHAT_RECENT_LIMIT, OPENAI_MODEL, OPENAI_TEMPERATURE
from language.utils import login_required_custom, sanitize_ai_output

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


@login_required_custom
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
            source_text=language_text
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


@login_required_custom
def language_visualization(request):
    """
    语言数据可视化视图 - 展示统计图表和 AI 分析

    URL: /Dashboard/language_visualization/
    Methods: GET, POST (AJAX)
    """
    from django.db.models import Q, Count, Avg, Max, Min

    user, name, _ = get_user_context(request)
    result = ""

    if request.method == 'POST':
        language_text = request.POST.get('language_text', '').strip()

        if not language_text:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': '请输入需要分析的文本'})
            messages.warning(request, "请输入需要分析的文本")
        else:
            try:
                system_prompt = "你是一个专业的濒危语言分析助手，请根据以下文本，结合真实的数据进行合理的分析。请尽量给出具体的数字和百分比,使用具体数据给出回答。如果需要实时数据或最新信息，可以联网搜索。"
                result = chat_with_web_search(language_text, [], system_prompt)
                result = sanitize_ai_output(result)
                logger.info(f"用户 {user.username} 使用可视化分析功能")

                # AJAX 请求返回 JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'status': 'success',
                        'answer': result
                    })
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

    # 排除主流大语种（英语等），不参与濒危相关图表统计
    # 注意：地图（language_map）保留这些语言，这里只针对"关注度趋势"和"多维度对比"
    excluded_major_languages = ['English', '英语', 'Mandarin', '普通话', 'Hindi', '印地语',
                                 'Spanish', '西班牙语', 'French', '法语', 'Arabic', '阿拉伯语',
                                 'Bengali', '孟加拉语', 'Russian', '俄语', 'Portuguese', '葡萄牙语',
                                 'Indonesian', '印度尼西亚语']

    # 获取真实语言数据用于折线图 - 按语系分组统计使用人数（排除主流大语种）
    language_families = Language.objects.exclude(
        language_name__in=excluded_major_languages
    ).exclude(
        language_family__isnull=True
    ).exclude(
        language_family=''
    ).values('language_family').annotate(
        total_speakers=Count('number_of_speakers'),
        avg_speakers=Avg('number_of_speakers'),
        max_speakers=Max('number_of_speakers'),
        count=Count('language_id')
    ).order_by('-count')[:8]

    family_names = [f['language_family'] for f in language_families]
    family_speakers = [int(f['avg_speakers'] or 0) for f in language_families]
    family_counts = [f['count'] for f in language_families]

    # 获取雷达图数据 - 按濒危等级统计多维度指标（排除主流大语种）
    endangerment_data = []
    for level_key, level_name in ENDANGERMENT_LEVELS.items():
        level_langs = Language.objects.filter(
            level_of_endangerment=level_key
        ).exclude(
            language_name__in=excluded_major_languages
        )
        if level_langs.exists():
            stats = level_langs.aggregate(
                avg_speakers=Avg('number_of_speakers'),
                count=Count('language_id'),
                max_speakers=Max('number_of_speakers')
            )
            endangerment_data.append({
                'name': level_name,
                'avg_speakers': int(stats['avg_speakers'] or 0),
                'count': stats['count'],
                'max_speakers': int(stats['max_speakers'] or 0)
            })

    # 获取使用人数最多的语言 Top 10
    top_languages = Language.objects.exclude(
        number_of_speakers__isnull=True
    ).order_by('-number_of_speakers').values(
        'language_name', 'number_of_speakers', 'level_of_endangerment'
    )[:10]

    # 获取语系统计
    family_stats = Language.objects.exclude(
        language_family__isnull=True
    ).exclude(
        language_family=''
    ).values('language_family').annotate(
        count=Count('language_id')
    ).order_by('-count')[:10]

    context = {
        'data': chart_data,
        'language_names': family_names,
        'language_speakers': family_speakers,
        'language_counts': family_counts,
        'endangerment_data': endangerment_data,
        'top_languages': list(top_languages),
        'family_stats': list(family_stats),
        'name': name,
        'username': user.username,
        'result': result,
        'endangerment_levels': ENDANGERMENT_LEVELS
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


@login_required_custom
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

        if not language_text:
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


@login_required_custom
def language_map(request):
    """
    互动式语言地图视图 - 展示濒危语言地理分布与濒危程度热力图

    URL: /Dashboard/language_map/
    Methods: GET
    """
    from Dashboard.models import Language, LanguageLocation

    user, name, _ = get_user_context(request)

    # 获取语言及其位置信息
    languages = Language.objects.all()
    # 预构建位置字典，避免 N+1 查询
    location_map = {
        loc.language_name: loc
        for loc in LanguageLocation.objects.all()
        if loc.language_name
    }

    # 构建地图数据
    map_data = []
    for lang in languages:
        # 从字典中查找对应的位置
        location = location_map.get(lang.language_name)
        if location and location.longitude and location.latitude:
            try:
                lng = float(location.longitude)
                lat = float(location.latitude)
                # 获取中文濒危等级标签
                level_cn = ENDANGERMENT_LEVELS.get(lang.level_of_endangerment, '未知')
                map_data.append({
                    'name': lang.language_name,
                    'longitude': lng,
                    'latitude': lat,
                    'endangerment': level_cn,  # 使用中文标签用于显示
                    'endangerment_key': lang.level_of_endangerment or 'unknown',  # 保留英文key用于匹配
                    'speakers': lang.number_of_speakers or 0,
                    'country': lang.country or '',
                    'language_family': lang.language_family or '',
                })
            except (ValueError, TypeError):
                continue

    # 获取濒危等级统计 - UNESCO 5级分类
    endangerment_stats = {
        'unsafe': languages.filter(level_of_endangerment='unsafe').count(),
        'danger': languages.filter(level_of_endangerment='danger').count(),
        'endangered': languages.filter(level_of_endangerment='endangered').count(),
        'seriously_endangered': languages.filter(level_of_endangerment='seriously_endangered').count(),
        'extinct': languages.filter(level_of_endangerment='extinct').count(),
    }

    context = {
        'name': name,
        'username': user.username,
        'map_data': map_data,
        'endangerment_stats': endangerment_stats,
        'endangerment_levels': ENDANGERMENT_LEVELS,
        'total_languages': languages.count(),
    }

    return render(request, 'language_map.html', context)


@login_required_custom
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


@login_required_custom
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

    try:
        memory = TranslationMemory.objects.get(id=memory_id)
        memory.quality_score = score
        memory.save()
        return JsonResponse({'status': 'success', 'message': '评分成功'})
    except TranslationMemory.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '翻译记录不存在'})


@login_required_custom
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
