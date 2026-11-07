"""
Dashboard 语言管理视图

包含濒危语言数据的增删改查、列表展示、搜索功能
"""

import csv
import os
import re
import logging
from io import StringIO
from typing import Tuple

from django.conf import settings
from django.shortcuts import render, redirect
from django.db import transaction
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse

from Dashboard.models import (
    Language, LEGACY_LEVEL_LABELS, ELCAT_LEVEL_LABELS,
)
from Dashboard.forms import LanguageForm
from Dashboard.common import (
    get_user_context, get_chart_data, get_paginated_languages,
    validate_csv_file, parse_csv_row, check_duplicate_language,
    invalidate_language_stats_cache, count_distinct_countries,
    count_distinct_families, pagination_window,
    ENDANGERMENT_LEVELS, MAX_SEARCH_LENGTH, CSV_BATCH_SIZE
)
from Dashboard.constants import EXCLUDED_MAJOR_LANGUAGES
from Dashboard.risk_service import (
    get_alerts, get_risk_profile, get_risk_stats, get_service_meta,
)
from language.utils import login_required_redirect

logger = logging.getLogger(__name__)


def language(request):
    """
    首页视图 - 展示系统概览

    URL: /Dashboard/language/
    Method: GET
    """
    from django.db.models import Sum, Avg
    from Community.models import Post, Reply
    from User.models import User

    user, name, is_admin = get_user_context(request)

    # 获取语言统计（总数按实际记录数，不因名称为空而漏计）
    chart_data = get_chart_data()
    total_languages = Language.objects.count()

    # 获取濒危语言数量（只排除英语等主流大语种，不安全和危险也算濒危）
    endangered_count = Language.objects.exclude(
        language_name__in=EXCLUDED_MAJOR_LANGUAGES
    ).count()

    # 获取使用人数统计（只统计濒危语言，排除英语等主流大语种）
    speaker_stats = Language.objects.exclude(
        language_name__in=EXCLUDED_MAJOR_LANGUAGES
    ).exclude(
        number_of_speakers__isnull=True
    ).aggregate(
        total_speakers=Sum('number_of_speakers'),
        avg_speakers=Avg('number_of_speakers')
    )

    # 语系 / 国家统计：数据库先去重，再按既有口径拆分
    family_count = count_distinct_families()
    country_count = count_distinct_countries()

    # 获取社区统计
    # 注意：Post 模型的状态值为 'draft'、'published'、'hidden'、'banned'，没有 'approved'
    total_posts = Post.objects.filter(status='published').count()
    total_replies = Reply.objects.filter(is_deleted=False).count()
    total_users = User.objects.filter(is_active=True).count()

    # 获取使用人数最多的濒危语言 Top 5（排除英语等主流大语种，且必须有分级数据）
    top_languages = Language.objects.exclude(
        language_name=''
    ).exclude(
        language_name__isnull=True
    ).exclude(
        language_name__startswith='未命名语言'
    ).exclude(
        language_name__in=EXCLUDED_MAJOR_LANGUAGES
    ).exclude(
        number_of_speakers__isnull=True
    ).exclude(
        level_of_endangerment__isnull=True
    ).exclude(
        level_of_endangerment=''
    ).order_by('-number_of_speakers').values(
        'language_name', 'number_of_speakers', 'level_of_endangerment',
        'legacy_endangerment_level', 'country'
    )

    # 按名称去重，避免重复导入的记录占据榜单
    top_list = []
    seen_names = set()
    for item in top_languages:
        language_name = (item['language_name'] or '').strip()
        if not language_name or language_name in seen_names:
            continue
        seen_names.add(language_name)
        item['legacy_label'] = LEGACY_LEVEL_LABELS.get(
            item['legacy_endangerment_level'], '未分类')
        item['elcat_label'] = ELCAT_LEVEL_LABELS.get(
            item['level_of_endangerment'], '未分类')
        top_list.append(item)
        if len(top_list) >= 5:
            break

    # 获取最新添加的语言（只取有分级数据、名称不重复的记录）
    latest_languages = Language.objects.exclude(
        language_name=''
    ).exclude(
        language_name__isnull=True
    ).exclude(
        language_name__startswith='未命名语言'
    ).exclude(
        level_of_endangerment__isnull=True
    ).exclude(
        level_of_endangerment=''
    ).order_by('-language_id').values(
        'language_name', 'level_of_endangerment', 'legacy_endangerment_level', 'country'
    )
    latest_list = []
    seen_names = set()
    for item in latest_languages:
        language_name = (item['language_name'] or '').strip()
        if not language_name or language_name in seen_names:
            continue
        seen_names.add(language_name)
        item['legacy_label'] = LEGACY_LEVEL_LABELS.get(
            item['legacy_endangerment_level'], '未分类')
        latest_list.append(item)
        if len(latest_list) >= 5:
            break

    context = {
        'name': name,
        'username': user.username if user else '',
        'is_admin': is_admin == 1 if user else False,
        'chart_data': chart_data,
        'total_languages': total_languages,
        'endangered_count': endangered_count,
        'total_speakers': int(speaker_stats['total_speakers'] or 0),
        'avg_speakers': int(speaker_stats['avg_speakers'] or 0),
        'family_count': family_count,
        'country_count': country_count,
        'total_posts': total_posts,
        'total_replies': total_replies,
        'total_users': total_users,
        'top_languages': top_list,
        'latest_languages': latest_list,
    }

    return render(request, 'language.html', context)


@login_required_redirect
@csrf_protect
def language_list(request):
    """
    语言数据列表视图 - 支持增删改查和 CSV 导入

    URL: /Dashboard/language_list/
    Methods: GET, POST
    """
    user, name, _ = get_user_context(request)

    if request.method == 'GET':
        return _handle_language_list_get(request, user, name)
    else:
        response = _handle_language_list_post(request, user, name)
        invalidate_language_stats_cache()
        return response


def _handle_language_list_get(request, user, name):
    """处理语言列表 GET 请求"""
    sort_field = request.GET.get('sort', 'language_id')
    order = request.GET.get('order', 'asc')
    page_number = request.GET.get('page', 1)

    allowed_sort_fields = [
        'language_id', 'language_name', 'level_of_endangerment',
        'number_of_speakers', 'speaker_trends', 'country'
    ]
    if sort_field not in allowed_sort_fields:
        sort_field = 'language_id'

    queryset = Language.objects.all()
    if order == 'desc':
        queryset = queryset.order_by(f'-{sort_field}')
    else:
        queryset = queryset.order_by(sort_field)

    paginator, page_obj = get_paginated_languages(queryset, page_number)

    # 获取用户的管理员标识
    is_admin = getattr(user, 'admin', 0) == 1 if user else False

    csv_result = request.session.pop('csv_import_result', None)

    context = {
        'languages': page_obj,
        'paginator': paginator,
        'page_obj': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
        'query_prefix': f'sort={sort_field}&order={order}&',
        'csv_result': csv_result,
        'data': get_chart_data(),
        'name': name,
        'username': user.username if user else '',
        'is_admin': is_admin,
        'sort_field': sort_field,
        'order': order,
        'endangerment_levels': ENDANGERMENT_LEVELS,
        'risk_alerts': get_alerts(limit=8, min_score=75),
        'risk_stats': get_risk_stats(),
    }

    return render(request, 'language_list.html', context)


def _handle_language_list_post(request, user, name):
    """处理语言列表 POST 请求（增删改查操作）"""
    # 检查是否为管理员
    is_admin = getattr(user, 'admin', 0) == 1 if user else False
    if not is_admin:
        messages.error(request, "权限不足，只有管理员才能进行此操作")
        return redirect('/Dashboard/language_list/')

    try:
        if 'addBtn' in request.POST:
            return _handle_add_language(request, user)
        elif 'updateBtn' in request.POST:
            return _handle_update_language(request, user)
        elif 'deleteBtn' in request.POST:
            return _handle_delete_language(request, user)
        elif 'fileBtn' in request.POST:
            return _handle_csv_import(request, user)
        else:
            messages.warning(request, "未知的操作类型")
            return redirect('/Dashboard/language_list/')

    except Exception as e:
        logger.error(f"语言列表操作异常: {str(e)}", exc_info=True)
        messages.error(request, f"操作失败: {str(e)}")
        return redirect('/Dashboard/language_list/')


def _handle_add_language(request, user):
    """处理新增语言"""
    form = LanguageForm(request.POST)

    if not form.is_valid():
        error_msg = '; '.join([
            f"{field}: {', '.join(errors)}"
            for field, errors in form.errors.items()
        ])
        messages.error(request, f"表单验证失败: {error_msg}")
        return redirect('/Dashboard/language_list/')

    language_name = form.cleaned_data['language_name']

    if check_duplicate_language(language_name):
        messages.warning(request, f"语言 '{language_name}' 已存在，请勿重复添加")
        return redirect('/Dashboard/language_list/')

    with transaction.atomic():
        language_obj = form.save()
        logger.info(f"用户 {user.username} 新增语言: {language_name} (ID: {language_obj.language_id})")

    messages.success(request, f"成功添加语言: {language_name}")
    return redirect('/Dashboard/language_list/')


def _handle_update_language(request, user):
    """处理更新语言"""
    language_id = request.POST.get('language_id', '').strip()

    if not language_id:
        messages.error(request, "更新失败: 未提供语言 ID")
        return redirect('/Dashboard/language_list/')

    try:
        language_id = int(language_id)
    except ValueError:
        messages.error(request, "更新失败: ID 格式错误")
        return redirect('/Dashboard/language_list/')

    try:
        language_obj = Language.objects.get(language_id=language_id)
    except Language.DoesNotExist:
        messages.error(request, f"更新失败: 未找到 ID 为 {language_id} 的语言")
        return redirect('/Dashboard/language_list/')

    form = LanguageForm(request.POST, instance=language_obj)

    if not form.is_valid():
        error_msg = '; '.join([
            f"{field}: {', '.join(errors)}"
            for field, errors in form.errors.items()
        ])
        messages.error(request, f"表单验证失败: {error_msg}")
        return redirect('/Dashboard/language_list/')

    new_name = form.cleaned_data['language_name']

    if check_duplicate_language(new_name, exclude_id=language_id):
        messages.warning(request, f"语言名称 '{new_name}' 已被其他语言使用")
        return redirect('/Dashboard/language_list/')

    with transaction.atomic():
        form.save()
        logger.info(f"用户 {user.username} 更新语言: {new_name} (ID: {language_id})")

    messages.success(request, f"成功更新语言: {new_name}")
    return redirect('/Dashboard/language_list/')


def _archive_deleted_language(language_obj):
    """删除前把整行数据写入本地审计文件，便于误删恢复。"""
    import json
    from datetime import datetime
    from django.conf import settings
    from django.forms.models import model_to_dict

    archive_dir = os.path.join(settings.BASE_DIR, 'backups')
    os.makedirs(archive_dir, exist_ok=True)
    archive_file = os.path.join(archive_dir, 'deleted_languages.jsonl')
    payload = {
        'deleted_at': datetime.now().isoformat(timespec='seconds'),
        'deleted_by': getattr(language_obj, '_deleted_by', ''),
        'record': model_to_dict(language_obj),
    }
    with open(archive_file, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + '\n')


def _handle_delete_language(request, user):
    """处理删除语言"""
    language_id = request.POST.get('language_id', '').strip()

    if not language_id:
        messages.error(request, "删除失败: 未提供语言 ID")
        return redirect('/Dashboard/language_list/')

    try:
        language_id = int(language_id)
    except ValueError:
        messages.error(request, "删除失败: ID 格式错误")
        return redirect('/Dashboard/language_list/')

    try:
        language_obj = Language.objects.get(language_id=language_id)
    except Language.DoesNotExist:
        messages.error(request, f"删除失败: 未找到 ID 为 {language_id} 的语言")
        return redirect('/Dashboard/language_list/')

    language_name = language_obj.language_name
    language_obj._deleted_by = user.username

    with transaction.atomic():
        _archive_deleted_language(language_obj)
        language_obj.delete()
        logger.info(f"用户 {user.username} 删除语言: {language_name} (ID: {language_id})")

    messages.success(request, f"成功删除语言: {language_name}")
    return redirect('/Dashboard/language_list/')


def _handle_csv_import(request, user):
    """处理 CSV 批量导入"""
    uploaded_file = request.FILES.get('myFile')

    is_valid, error_msg = validate_csv_file(uploaded_file)
    if not is_valid:
        messages.error(request, f"文件验证失败: {error_msg}")
        return redirect('/Dashboard/language_list/')

    try:
        file_content = uploaded_file.read()
        try:
            data = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                data = file_content.decode('gbk')
            except UnicodeDecodeError:
                messages.error(request, "文件编码错误: 请使用 UTF-8 或 GBK 编码的 CSV 文件")
                return redirect('/Dashboard/language_list/')
    except Exception as e:
        logger.error(f"文件读取失败: {str(e)}")
        messages.error(request, f"文件读取失败: {str(e)}")
        return redirect('/Dashboard/language_list/')

    reader = csv.reader(StringIO(data))
    header = next(reader, None)

    success_count = 0
    error_count = 0
    error_messages = []
    languages_to_create = []
    seen_names = set()
    line_number = 1

    for row in reader:
        line_number += 1

        if not row or all(not cell.strip() for cell in row):
            continue

        is_valid_row, parsed_data, error_msg = parse_csv_row(row, line_number)

        if not is_valid_row:
            error_count += 1
            error_messages.append(error_msg)
            continue

        normalized_name = parsed_data['language_name'].strip().lower()
        if normalized_name in seen_names:
            error_count += 1
            error_messages.append(f"第 {line_number} 行: 语言 '{parsed_data['language_name']}' 在文件中重复")
            continue
        if check_duplicate_language(parsed_data['language_name']):
            error_count += 1
            error_messages.append(f"第 {line_number} 行: 语言 '{parsed_data['language_name']}' 已存在")
            continue

        seen_names.add(normalized_name)
        languages_to_create.append(Language(**parsed_data))

        if len(languages_to_create) >= CSV_BATCH_SIZE:
            batch_size = len(languages_to_create)
            try:
                with transaction.atomic():
                    Language.objects.bulk_create(languages_to_create)
                success_count += batch_size
                logger.info(f"批量导入 {batch_size} 条语言数据")
            except Exception as e:
                error_count += batch_size
                error_messages.append(f"批量插入失败: {str(e)}")
            finally:
                languages_to_create = []

    if languages_to_create:
        batch_size = len(languages_to_create)
        try:
            with transaction.atomic():
                Language.objects.bulk_create(languages_to_create)
            success_count += batch_size
            logger.info(f"批量导入 {batch_size} 条语言数据")
        except Exception as e:
            error_count += batch_size
            error_messages.append(f"批量插入失败: {str(e)}")

    request.session['csv_import_result'] = {
        'success': success_count,
        'failed': error_count,
        'errors': error_messages[:500],
    }

    if success_count > 0:
        messages.success(request, f"成功导入 {success_count} 条语言数据")

    if error_count > 0:
        display_errors = error_messages[:5]
        if len(error_messages) > 5:
            display_errors.append(f"... 还有 {len(error_messages) - 5} 条错误")
        error_text = "; ".join(display_errors)
        messages.warning(request, f"导入完成，{error_count} 条失败: {error_text}")

    if success_count == 0 and error_count == 0:
        messages.info(request, "文件为空或未包含有效数据")

    return redirect('/Dashboard/language_list/')


@login_required_redirect
def language_search(request):
    """
    语言数据搜索视图 - 支持多条件模糊搜索

    URL: /Dashboard/language_search/
    Method: GET
    """
    from django.db.models import Q

    user, name, is_admin = get_user_context(request)

    search_query = request.GET.get('q', '').strip()
    search_type = request.GET.get('search_type', 'all')
    page_number = request.GET.get('page', 1)

    # 搜索关键词只做长度限制；不按字符集过滤，否则西里尔文、阿拉伯文、
    # 带变音符的拉丁文等原生文字会被直接清空。
    if len(search_query) > MAX_SEARCH_LENGTH:
        search_query = search_query[:MAX_SEARCH_LENGTH]

    allowed_types = ['name', 'family', 'country', 'all']
    if search_type not in allowed_types:
        search_type = 'all'

    if search_query:
        if search_type == 'name':
            queryset = Language.objects.filter(Q(language_name__icontains=search_query))
        elif search_type == 'family':
            queryset = Language.objects.filter(Q(language_family__icontains=search_query))
        elif search_type == 'country':
            queryset = Language.objects.filter(Q(country__icontains=search_query))
        else:
            queryset = Language.objects.filter(
                Q(language_name__icontains=search_query) |
                Q(language_family__icontains=search_query) |
                Q(country__icontains=search_query) |
                Q(additional_notes__icontains=search_query)
            ).distinct()
    else:
        queryset = Language.objects.all()

    queryset = queryset.order_by('-language_id')

    paginator, page_obj = get_paginated_languages(queryset, page_number)

    context = {
        'languages': page_obj,
        'paginator': paginator,
        'page_obj': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
        'search_query': search_query,
        'search_type': search_type,
        'name': name,
        'username': user.username if user else '',
        'is_admin': is_admin == 1 if user else False,
        'endangerment_levels': ENDANGERMENT_LEVELS,
        'result_count': queryset.count()
    }

    return render(request, 'language_search.html', context)


@login_required_redirect
def user_management(request):
    """
    用户管理页面 - 仅管理员可见

    URL: /Dashboard/user_management/
    Method: GET
    """
    from django.db.models import Count, Q

    user, name, is_admin = get_user_context(request)

    # 检查是否为管理员
    if str(is_admin) != '1':
        messages.error(request, "权限不足，只有管理员才能访问此页面")
        return redirect('/Dashboard/language/')

    from User.models import User, UserProfile
    from Community.models import Post, Reply

    # 引导账号从环境变量读取，不再把用户名写死在代码里
    super_admin_username = getattr(settings, 'SUPER_ADMIN_USERNAME', '')
    is_super_admin = bool(
        user and user.is_admin and super_admin_username and user.username == super_admin_username
    )

    # 权限筛选：all / admin / expert / normal
    current_role = request.GET.get('role', 'all')
    if current_role not in ('all', 'admin', 'expert', 'normal'):
        current_role = 'all'

    # 使用 annotation 一次性查询所有用户的帖子和回复数，避免 N+1
    users_data = User.objects.filter(is_active=True).annotate(
        post_count=Count('posts', filter=Q(posts__status='published')),
        reply_count=Count('replies', filter=Q(replies__is_deleted=False))
    ).order_by('username').values(
        'username', 'name', 'regtime', 'admin', 'expert', 'post_count', 'reply_count'
    )

    users_list = []
    for u in users_data:
        users_list.append({
            'user': User(username=u['username'], name=u['name'],
                         regtime=u['regtime'], admin=u['admin'],
                         expert=u.get('expert', 0)),
            'post_count': u['post_count'],
            'reply_count': u['reply_count'],
        })

    # 统计始终按全量用户计算，不受当前筛选影响
    total_count = len(users_list)
    admin_count = sum(1 for u in users_list if u['user'].admin == 1)
    expert_count = sum(1 for u in users_list if u['user'].expert == 1)
    normal_count = sum(
        1 for u in users_list
        if u['user'].admin != 1 and u['user'].expert != 1
    )

    if current_role == 'admin':
        filtered_users = [u for u in users_list if u['user'].admin == 1]
    elif current_role == 'expert':
        filtered_users = [u for u in users_list if u['user'].expert == 1]
    elif current_role == 'normal':
        filtered_users = [
            u for u in users_list
            if u['user'].admin != 1 and u['user'].expert != 1
        ]
    else:
        filtered_users = users_list

    from django.core.paginator import Paginator
    paginator = Paginator(filtered_users, 20)
    page_obj = paginator.get_page(request.GET.get('page', 1))

    context = {
        'name': name,
        'username': user.username if user else '',
        'is_admin': True,
        'is_super_admin': is_super_admin,
        'super_admin_username': super_admin_username,
        'users_data': page_obj,
        'page_obj': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
        'current_role': current_role,
        'total_count': total_count,
        'admin_count': admin_count,
        'normal_count': normal_count,
        'expert_count': expert_count,
    }
    return render(request, 'user_management.html', context)


@login_required_redirect
@csrf_protect
def user_management_post(request):
    """
    用户管理操作 - 仅超级管理员可用

    URL: /Dashboard/user_management/ POST
    """
    user, name, is_admin = get_user_context(request)

    # 检查是否为管理员
    if str(is_admin) != '1':
        return JsonResponse({'success': False, 'message': '权限不足'}, status=403)

    from User.models import User
    action = request.POST.get('action', '')
    target_username = request.POST.get('username', '').strip()

    if not target_username:
        return JsonResponse({'success': False, 'message': '未指定用户'})

    try:
        target_user = User.objects.get(username=target_username)
    except User.DoesNotExist:
        return JsonResponse({'success': False, 'message': '用户不存在'})

    super_admin_username = getattr(settings, 'SUPER_ADMIN_USERNAME', '')
    is_super_admin = bool(
        user and user.is_admin and super_admin_username and user.username == super_admin_username
    )

    # 防止操作自己
    if user is None or target_username == user.username:
        if action == 'delete':
            return JsonResponse({'success': False, 'message': '不能删除自己'})
        return JsonResponse({'success': False, 'message': '不能修改自己的权限'})

    # 管理员账号互相保护，只有引导账号可以调整管理员角色
    if target_user.admin == 1 and not is_super_admin:
        return JsonResponse({'success': False, 'message': '管理员账号只能由超级管理员调整'}, status=403)

    if action in ('set_admin', 'set_normal'):
        if not is_super_admin:
            return JsonResponse({'success': False, 'message': '只有超级管理员才能调整管理员角色'}, status=403)
    elif action in ('set_expert', 'unset_expert', 'delete'):
        if target_user.admin == 1:
            return JsonResponse({'success': False, 'message': '不能修改管理员账号'}, status=403)
    else:
        return JsonResponse({'success': False, 'message': '未知操作'})

    if action == 'set_admin':
        target_user.admin = 1
        target_user.save(update_fields=['admin'])
        return JsonResponse({'success': True, 'message': f'已将 {target_username} 设为管理员'})
    elif action == 'set_normal':
        target_user.admin = 0
        target_user.save(update_fields=['admin'])
        return JsonResponse({'success': True, 'message': f'已将 {target_username} 设为普通用户'})
    elif action == 'set_expert':
        target_user.expert = 1
        target_user.save(update_fields=['expert'])
        return JsonResponse({'success': True, 'message': f'已将 {target_username} 设为专家'})
    elif action == 'unset_expert':
        target_user.expert = 0
        target_user.save(update_fields=['expert'])
        return JsonResponse({'success': True, 'message': f'已取消 {target_username} 的专家资格'})
    elif action == 'delete':
        target_user.delete()
        return JsonResponse({'success': True, 'message': f'已删除用户 {target_username}'})


def _fetch_vitality_dimensions(vitality):
    """一次查询取回代际传承、使用者趋势、使用领域三张查找表的记录。

    三张表都是 6 行的静态参考表，通过 UNION 合并成一条 SQL，
    避免每个详情页请求额外产生 3 条独立查询。
    """
    from django.db.models import Value, CharField
    from Dashboard.models import (
        Transmission, SpeakerNumberTrend, DomainOfUse,
    )

    specs = (
        ('transmission', Transmission, vitality.transmission_id),
        ('trend', SpeakerNumberTrend, vitality.speaker_number_trends_id),
        ('domain', DomainOfUse, vitality.domains_of_use_id),
    )
    querysets = [
        model.objects.filter(id=pid).annotate(
            kind=Value(kind, output_field=CharField()),
        ).values('kind', 'id', 'level', 'description')
        for kind, model, pid in specs if pid is not None
    ]
    if not querysets:
        return None, None, None

    rows = querysets[0].union(*querysets[1:]) if len(querysets) > 1 else querysets[0]
    models_by_kind = {kind: model for kind, model, _ in specs}
    records = {}
    for row in rows:
        model = models_by_kind[row['kind']]
        records[row['kind']] = model(
            id=row['id'], level=row['level'], description=row['description'])
    return (
        records.get('transmission'),
        records.get('trend'),
        records.get('domain'),
    )


def language_detail(request, language_id):
    """
    语言详情页 - 展示语言的详细信息、相关帖子、统计数据

    URL: /Dashboard/language/<int:language_id>/
    Method: GET
    """
    from django.db.models import Q
    from Dashboard.models import (
        LanguageLocation, LanguageVitality, SpeakerData, BibliographyLink,
        LanguageSource, LanguageOrthography, LanguageContext,
        LanguageLocationDetail, LanguageMedia, LanguageActivity, LanguageSample,
        Country, LanguageCountry,
    )
    from Dashboard.sample_views import SAMPLE_TYPES, sample_media
    from Community.models import Post, Like

    user, name, is_admin = get_user_context(request)

    # 获取语言详情
    try:
        lang = Language.objects.get(language_id=language_id)
    except Language.DoesNotExist:
        from django.http import Http404
        raise Http404('语言不存在')

    code_id = lang.language_id

    # 获取位置信息（兼容层），并校验经纬度可用于地图
    location = LanguageLocation.objects.filter(language_name=lang.language_name).first()
    location_valid = False
    if location:
        try:
            float(location.longitude)
            float(location.latitude)
            location_valid = True
        except (TypeError, ValueError):
            location_valid = False

    # 获取濒危等级中文标签（本系统 UNESCO 五级）
    endangerment_label = lang.legacy_level_display

    # 活力评估（含三个维度）
    # 一次取回全部活力记录：既作为档案计数，也用于选出首选记录，避免额外 COUNT 查询
    vitality_records = list(LanguageVitality.objects.filter(
        code_id=code_id).order_by('-preferred', 'id'))
    vitality = vitality_records[0] if vitality_records else None
    vitality_info = None
    if vitality:
        transmission, trend, domain = _fetch_vitality_dimensions(vitality)
        vitality_info = {
            'record': vitality,
            'transmission': transmission,
            'trend': trend,
            'domain': domain,
            'source': None,
        }

    # 使用者记录：优先 preferred，缺失字段时选信息最全的一条
    speaker_records = list(SpeakerData.objects.filter(
        code_id=code_id).order_by('-preferred', 'id'))
    speaker = None
    if speaker_records:
        fields = ['speaker_number', 'children', 'young_adults',
                  'older_adults', 'elders', 'ethnic_population']
        speaker = max(speaker_records, key=lambda r: sum(
            1 for f in fields if getattr(r, f) is not None))

    speaker_demographics = []
    if speaker:
        demo_items = [
            ('儿童', speaker.children),
            ('青年', speaker.young_adults),
            ('老年', speaker.older_adults),
            ('长老', speaker.elders),
        ]
        max_demo = max([value for _, value in demo_items if value] or [1]) or 1
        speaker_demographics = [{
            'label': label,
            'value': value or 0,
            'pct': round((value or 0) / max_demo * 100, 1),
        } for label, value in demo_items]

    # 文献：只展示真实来源，占位行不展示
    # 一次取回全部关联来源：关联数即 source_ids 长度，真实条数即有效来源数，
    # 无需再发两条 COUNT 查询
    source_ids = list(BibliographyLink.objects.filter(
        code_id=code_id).values_list('source_id', flat=True))
    bibliography_count = len(source_ids)
    real_sources = list(LanguageSource.objects.filter(
        id__in=source_ids, is_placeholder=False
    ).exclude(title='').order_by('year'))
    bibliography = real_sources[:10]
    real_bibliography_count = len(real_sources)

    # 活力评估来源通常也在文献关联中，优先复用已加载的来源，避免重复查询
    if vitality_info is not None and vitality.source_id is not None:
        vitality_source = next(
            (s for s in real_sources if s.id == vitality.source_id), None)
        if vitality_source is None:
            vitality_source = LanguageSource.objects.filter(
                id=vitality.source_id).first()
        vitality_info['source'] = vitality_source

    # 正字法：只展示有内容的记录
    # 一次取回全部记录，展示仍按原规则只扫描前 30 条，总数直接用长度
    orthography_records = list(LanguageOrthography.objects.filter(
        code_id=code_id).order_by('-preferred', 'id'))
    orthography = []
    for record in orthography_records[:30]:
        if (record.scripts or record.public_comment
                or record.competing_orthographies or record.standard_orthography):
            orthography.append(record)
        if len(orthography) >= 5:
            break
    orthography_count = len(orthography_records)

    language_context = LanguageContext.objects.filter(
        code_id=code_id).order_by('-preferred', 'id').first()
    location_records = list(LanguageLocationDetail.objects.filter(
        code_id=code_id).order_by('-preferred', 'id'))
    locations = location_records[:5]
    media_records = list(LanguageMedia.objects.filter(code_id=code_id))
    media = media_records[:6]
    activity_records = list(LanguageActivity.objects.filter(
        code_id=code_id).order_by('-date', '-id'))
    activities = activity_records[:10]

    # 一次取回全部未移除样本：展示仍取前 6 条有标题的，总数直接用长度
    sample_records = list(LanguageSample.objects.filter(
        code_id=code_id, removed=False).order_by(
        '-featured', '-view_count', '-id'))
    samples = [s for s in sample_records if s.title][:6]
    for sample in samples:
        sample.thumbnail, sample.play_url = sample_media(sample)
        sample.type_label = SAMPLE_TYPES.get(sample.sample_type, '其他')

    data_counts = {
        'vitality': len(vitality_records),
        'speaker': len(speaker_records),
        'bibliography': bibliography_count,
        'bibliography_real': real_bibliography_count,
        'orthography': orthography_count,
        'location': len(location_records),
        'activity': len(activity_records),
        'media': len(media_records),
        'samples': len(sample_records),
    }

    # 获取相关帖子（名称为空时不能用空串做 icontains，否则会匹配全部帖子）
    language_name = (lang.language_name or '').strip()
    related_post_filter = Q(related_language=lang)
    if language_name:
        related_post_filter |= Q(content__icontains=language_name)
    related_posts = Post.objects.filter(
        related_post_filter, status='published'
    ).order_by('-created_at')[:10]

    # 获取同语系的语言
    related_languages = []
    if lang.language_family:
        related_languages = Language.objects.filter(
            language_family=lang.language_family
        ).exclude(language_id=lang.language_id)[:6]

    # 获取同国家的语言（基于 language_country_map 的完整多国关系）
    # 国家 ID 用子查询直接过滤，省去单独物化 ID 列表的查询
    country_id_subquery = LanguageCountry.objects.filter(
        language_id=code_id).values_list('country_id', flat=True)
    countries = list(Country.objects.filter(
        id__in=country_id_subquery).values_list('name', flat=True))
    same_country_languages = []
    if countries:
        # 先取其他语言 ID（保持原有 LIMIT 6 的选取顺序），再取语言对象，
        # 保证与原实现的条目选择完全一致
        other_ids = list(LanguageCountry.objects.filter(
            country_id__in=country_id_subquery
        ).exclude(language_id=code_id).values_list(
            'language_id', flat=True).distinct()[:6])
        same_country_languages = Language.objects.filter(
            language_id__in=other_ids)

    # 统计信息
    post_count = Post.objects.filter(
        related_post_filter, status='published'
    ).count()

    context = {
        'language': lang,
        'location': location,
        'location_valid': location_valid,
        'endangerment_label': endangerment_label,
        'endangerment_class': lang.legacy_endangerment_level or 'unknown',
        'vitality_info': vitality_info,
        'speaker': speaker,
        'speaker_demographics': speaker_demographics,
        'bibliography': bibliography,
        'orthography': orthography,
        'language_context': language_context,
        'locations': locations,
        'media': media,
        'activities': activities,
        'samples': samples,
        'data_counts': data_counts,
        'related_posts': related_posts,
        'related_languages': related_languages,
        'same_country_languages': same_country_languages,
        'countries': countries,
        'post_count': post_count,
        'risk': get_risk_profile(code_id),
        'risk_meta': get_service_meta(),
        'name': name,
        'username': user.username if user else '',
        'is_admin': is_admin == 1 if user else False,
    }

    return render(request, 'language_detail.html', context)


@login_required_redirect
def admin_dashboard(request):
    """
    管理员仪表盘 - 展示系统关键指标和统计数据

    URL: /Dashboard/admin_dashboard/
    Method: GET
    """
    from django.db.models import Count, Sum, Avg, Q
    from django.utils import timezone
    from datetime import timedelta
    from Community.models import Post, Reply, Like, Report
    from User.models import User

    user, name, is_admin = get_user_context(request)

    # 检查管理员权限
    if is_admin != 1:
        messages.error(request, '权限不足')
        return redirect('Dashboard:language')

    # 时间范围
    now = timezone.now()
    today = now.date()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # 语言统计
    total_languages = Language.objects.count()
    endangered_languages = Language.objects.exclude(
        language_name__in=EXCLUDED_MAJOR_LANGUAGES
    ).count()

    # 用户统计
    total_users = User.objects.filter(is_active=True).count()
    active_users_week = User.objects.filter(is_active=True, last_login__gte=week_ago).count()
    new_users_week = User.objects.filter(is_active=True, regtime__gte=week_ago).count()
    expert_users = User.objects.filter(is_active=True, expert=1).count()

    # 帖子统计
    total_posts = Post.objects.count()
    posts_week = Post.objects.filter(created_at__gte=week_ago).count()
    posts_pending = Post.objects.filter(needs_moderation=True, moderation_queue='pending').count()
    posts_today = Post.objects.filter(created_at__date=today).count()

    # 回复统计
    total_replies = Reply.objects.filter(is_deleted=False).count()
    replies_week = Reply.objects.filter(created_at__gte=week_ago, is_deleted=False).count()

    # 举报统计
    reports_pending = Report.objects.filter(status='pending').count()
    reports_total = Report.objects.count()

    # 热门帖子
    hot_posts = Post.objects.filter(
        status='published'
    ).order_by('-view_count')[:5]

    # 最新注册用户
    new_users = User.objects.order_by('-regtime')[:5]

    # 语言濒危等级分布（本系统 UNESCO 五级）
    endangerment_stats = {
        'unsafe': Language.objects.filter(legacy_endangerment_level='unsafe').count(),
        'danger': Language.objects.filter(legacy_endangerment_level='danger').count(),
        'endangered': Language.objects.filter(legacy_endangerment_level='endangered').count(),
        'seriously_endangered': Language.objects.filter(legacy_endangerment_level='seriously_endangered').count(),
        'extinct': Language.objects.filter(legacy_endangerment_level='extinct').count(),
        'awakening': Language.objects.filter(legacy_endangerment_level='awakening').count(),
    }

    context = {
        'name': name,
        'username': user.username,
        'is_admin': True,
        # 语言
        'total_languages': total_languages,
        'endangered_languages': endangered_languages,
        # 用户
        'total_users': total_users,
        'active_users_week': active_users_week,
        'new_users_week': new_users_week,
        'expert_users': expert_users,
        # 帖子
        'total_posts': total_posts,
        'posts_week': posts_week,
        'posts_pending': posts_pending,
        'posts_today': posts_today,
        # 回复
        'total_replies': total_replies,
        'replies_week': replies_week,
        # 举报
        'reports_pending': reports_pending,
        'reports_total': reports_total,
        # 列表
        'hot_posts': hot_posts,
        'new_users': new_users,
        'endangerment_stats': endangerment_stats,
    }

    return render(request, 'admin_dashboard.html', context)
