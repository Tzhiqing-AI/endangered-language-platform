"""
Dashboard 语言管理视图

包含濒危语言数据的增删改查、列表展示、搜索功能
"""

import csv
import re
import logging
from io import StringIO
from typing import Tuple

from django.shortcuts import render, redirect
from django.db import transaction
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse

from Dashboard.models import Language
from Dashboard.forms import LanguageForm
from Dashboard.common import (
    get_user_context, get_chart_data, get_paginated_languages,
    validate_csv_file, parse_csv_row, check_duplicate_language,
    ENDANGERMENT_LEVELS, MAX_SEARCH_LENGTH, CSV_BATCH_SIZE
)
from language.utils import login_required_custom

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

    # 获取语言统计
    chart_data = get_chart_data()
    total_languages = Language.objects.count()

    # 获取濒危语言数量（只排除英语等主流大语种，不安全和危险也算濒危）
    endangered_count = Language.objects.exclude(
        language_name__in=['English', '英语', 'Mandarin', '普通话', 'Hindi', '印地语', 'Spanish', '西班牙语',
                          'French', '法语', 'Arabic', '阿拉伯语', 'Bengali', '孟加拉语', 'Russian', '俄语',
                          'Portuguese', '葡萄牙语', 'Indonesian', '印度尼西亚语']
    ).count()

    # 获取使用人数统计（只统计濒危语言，排除英语等主流大语种）
    speaker_stats = Language.objects.exclude(
        language_name__in=['English', '英语', 'Mandarin', '普通话', 'Hindi', '印地语', 'Spanish', '西班牙语',
                          'French', '法语', 'Arabic', '阿拉伯语', 'Bengali', '孟加拉语', 'Russian', '俄语',
                          'Portuguese', '葡萄牙语', 'Indonesian', '印度尼西亚语']
    ).exclude(
        number_of_speakers__isnull=True
    ).aggregate(
        total_speakers=Sum('number_of_speakers'),
        avg_speakers=Avg('number_of_speakers')
    )

    # 获取语系统计
    family_count = Language.objects.exclude(
        language_family__isnull=True
    ).exclude(
        language_family=''
    ).values('language_family').distinct().count()

    # 获取国家分布统计
    country_count = Language.objects.exclude(
        country__isnull=True
    ).exclude(
        country=''
    ).values('country').distinct().count()

    # 获取社区统计
    # 注意：Post 模型的状态值为 'draft'、'published'、'hidden'、'banned'，没有 'approved'
    total_posts = Post.objects.filter(status='published').count()
    total_replies = Reply.objects.filter(is_deleted=False).count()
    total_users = User.objects.count()

    # 获取使用人数最多的濒危语言 Top 5（排除英语等主流大语种）
    top_languages = Language.objects.exclude(
        language_name__in=['English', '英语', 'Mandarin', '普通话', 'Hindi', '印地语', 'Spanish', '西班牙语',
                          'French', '法语', 'Arabic', '阿拉伯语', 'Bengali', '孟加拉语', 'Russian', '俄语',
                          'Portuguese', '葡萄牙语', 'Indonesian', '印度尼西亚语']
    ).exclude(
        number_of_speakers__isnull=True
    ).order_by('-number_of_speakers').values(
        'language_name', 'number_of_speakers', 'level_of_endangerment'
    )[:5]

    # 获取最新添加的语言
    latest_languages = Language.objects.order_by('-language_id').values(
        'language_name', 'level_of_endangerment', 'country'
    )[:5]

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
        'top_languages': list(top_languages),
        'latest_languages': list(latest_languages),
    }

    return render(request, 'language.html', context)


@login_required_custom
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
        return _handle_language_list_post(request, user, name)


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

    context = {
        'languages': page_obj,
        'paginator': paginator,
        'page_obj': page_obj,
        'data': get_chart_data(),
        'name': name,
        'username': user.username if user else '',
        'is_admin': is_admin,
        'sort_field': sort_field,
        'order': order,
        'endangerment_levels': ENDANGERMENT_LEVELS
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

    with transaction.atomic():
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

        if check_duplicate_language(parsed_data['language_name']):
            error_count += 1
            error_messages.append(f"第 {line_number} 行: 语言 '{parsed_data['language_name']}' 已存在")
            continue

        languages_to_create.append(Language(**parsed_data))
        success_count += 1

        if len(languages_to_create) >= CSV_BATCH_SIZE:
            try:
                with transaction.atomic():
                    Language.objects.bulk_create(languages_to_create)
                logger.info(f"批量导入 {len(languages_to_create)} 条语言数据")
                languages_to_create = []
            except Exception as e:
                error_count += len(languages_to_create)
                error_messages.append(f"批量插入失败: {str(e)}")
                languages_to_create = []

    if languages_to_create:
        try:
            with transaction.atomic():
                Language.objects.bulk_create(languages_to_create)
            logger.info(f"批量导入 {len(languages_to_create)} 条语言数据")
        except Exception as e:
            error_count += len(languages_to_create)
            error_messages.append(f"批量插入失败: {str(e)}")

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


@login_required_custom
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

    # 搜索防注入：限制关键词长度
    if len(search_query) > MAX_SEARCH_LENGTH:
        search_query = search_query[:MAX_SEARCH_LENGTH]

    # 搜索防注入：过滤危险字符
    safe_pattern = re.compile(r'[^a-zA-Z0-9一-鿿\s\-\.\,\。\，\-\+]')
    search_query = safe_pattern.sub('', search_query)

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
        'search_query': search_query,
        'search_type': search_type,
        'name': name,
        'username': user.username if user else '',
        'is_admin': is_admin == 1 if user else False,
        'endangerment_levels': ENDANGERMENT_LEVELS,
        'result_count': queryset.count()
    }

    return render(request, 'language_search.html', context)


@login_required_custom
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

    # 超级管理员：唯一可以修改其他用户权限的账号
    SUPER_ADMIN_USERNAME = 'htd'
    is_super_admin = bool(user and user.username == SUPER_ADMIN_USERNAME)

    # 使用 annotation 一次性查询所有用户的帖子和回复数，避免 N+1
    users_data = User.objects.annotate(
        post_count=Count('posts', filter=Q(posts__status='published')),
        reply_count=Count('replies', filter=Q(replies__is_deleted=False))
    ).order_by('username').values(
        'username', 'name', 'regtime', 'admin', 'expert', 'post_count', 'reply_count'
    )

    admin_count = 0
    normal_count = 0
    expert_count = 0
    users_list = []
    for u in users_data:
        if u['admin'] == 1:
            admin_count += 1
        else:
            normal_count += 1
        if u.get('expert') == 1:
            expert_count += 1
        users_list.append({
            'user': User(username=u['username'], name=u['name'],
                         regtime=u['regtime'], admin=u['admin'],
                         expert=u.get('expert', 0)),
            'post_count': u['post_count'],
            'reply_count': u['reply_count'],
        })

    context = {
        'name': name,
        'username': user.username if user else '',
        'is_admin': True,
        'is_super_admin': is_super_admin,
        'super_admin_username': SUPER_ADMIN_USERNAME,
        'users_data': users_list,
        'admin_count': admin_count,
        'normal_count': normal_count,
        'expert_count': expert_count,
    }
    return render(request, 'user_management.html', context)


@login_required_custom
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

    # 超级管理员（浩天帝）拥有唯一修改他人权限的权限
    SUPER_ADMIN_USERNAME = 'htd'

    # 防止操作自己
    if user is None or target_username == user.username:
        if action == 'delete':
            return JsonResponse({'success': False, 'message': '不能删除自己'})
        # 任何其他动作（设/降级、设/取消专家）一律禁止对自己操作
        return JsonResponse({'success': False, 'message': '不能修改自己的权限'})

    # 保护超级管理员：任何人都不能修改浩天帝的权限或删除浩天帝
    if target_username == SUPER_ADMIN_USERNAME:
        return JsonResponse({'success': False, 'message': '超级管理员的权限不可被修改'}, status=403)

    # 除超级管理员外，其他管理员不允许修改他人权限
    if user.username != SUPER_ADMIN_USERNAME:
        return JsonResponse({'success': False, 'message': '只有超级管理员才能修改其他用户的权限'}, status=403)

    if action == 'set_admin':
        target_user.admin = 1
        target_user.save()
        return JsonResponse({'success': True, 'message': f'已将 {target_username} 设为管理员'})
    elif action == 'set_normal':
        target_user.admin = 0
        target_user.save()
        return JsonResponse({'success': True, 'message': f'已将 {target_username} 设为普通用户'})
    elif action == 'set_expert':
        target_user.expert = 1
        target_user.save()
        return JsonResponse({'success': True, 'message': f'已将 {target_username} 设为专家'})
    elif action == 'unset_expert':
        target_user.expert = 0
        target_user.save()
        return JsonResponse({'success': True, 'message': f'已取消 {target_username} 的专家资格'})
    elif action == 'delete':
        target_user.delete()
        return JsonResponse({'success': True, 'message': f'已删除用户 {target_username}'})
    else:
        return JsonResponse({'success': False, 'message': '未知操作'})


def language_detail(request, language_id):
    """
    语言详情页 - 展示语言的详细信息、相关帖子、统计数据

    URL: /Dashboard/language/<int:language_id>/
    Method: GET
    """
    from django.db.models import Count, Q
    from Dashboard.models import LanguageLocation
    from Community.models import Post, Like

    user, name, is_admin = get_user_context(request)

    # 获取语言详情
    try:
        lang = Language.objects.get(language_id=language_id)
    except Language.DoesNotExist:
        messages.error(request, '语言不存在')
        return redirect('Dashboard:language_list')

    # 获取位置信息
    location = LanguageLocation.objects.filter(language_name=lang.language_name).first()

    # 获取濒危等级中文标签
    endangerment_label = ENDANGERMENT_LEVELS.get(lang.level_of_endangerment, '未知')

    # 获取相关帖子
    related_posts = Post.objects.filter(
        Q(related_language=lang) | Q(content__icontains=lang.language_name),
        status='published'
    ).order_by('-created_at')[:10]

    # 获取同语系的语言
    related_languages = []
    if lang.language_family:
        related_languages = Language.objects.filter(
            language_family=lang.language_family
        ).exclude(language_id=lang.language_id)[:6]

    # 获取同国家的语言
    same_country_languages = []
    if lang.country:
        same_country_languages = Language.objects.filter(
            country=lang.country
        ).exclude(language_id=lang.language_id)[:6]

    # 统计信息
    post_count = Post.objects.filter(
        Q(related_language=lang) | Q(content__icontains=lang.language_name),
        status='published'
    ).count()

    context = {
        'language': lang,
        'location': location,
        'endangerment_label': endangerment_label,
        'endangerment_class': lang.level_of_endangerment or 'unknown',
        'related_posts': related_posts,
        'related_languages': related_languages,
        'same_country_languages': same_country_languages,
        'post_count': post_count,
        'name': name,
        'username': user.username if user else '',
        'is_admin': is_admin == 1 if user else False,
    }

    return render(request, 'language_detail.html', context)


@login_required_custom
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
        language_name__in=['English', '英语', 'Mandarin', '普通话', 'Hindi', '印地语', 'Spanish', '西班牙语',
                          'French', '法语', 'Arabic', '阿拉伯语', 'Bengali', '孟加拉语', 'Russian', '俄语',
                          'Portuguese', '葡萄牙语', 'Indonesian', '印度尼西亚语']
    ).count()

    # 用户统计
    total_users = User.objects.count()
    active_users_week = User.objects.filter(last_login__gte=week_ago).count()
    new_users_week = User.objects.filter(regtime__gte=week_ago).count()
    expert_users = User.objects.filter(expert=1).count()

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

    # 语言濒危等级分布
    endangerment_stats = {
        'unsafe': Language.objects.filter(level_of_endangerment='unsafe').count(),
        'danger': Language.objects.filter(level_of_endangerment='danger').count(),
        'endangered': Language.objects.filter(level_of_endangerment='endangered').count(),
        'seriously_endangered': Language.objects.filter(level_of_endangerment='seriously_endangered').count(),
        'extinct': Language.objects.filter(level_of_endangerment='extinct').count(),
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
