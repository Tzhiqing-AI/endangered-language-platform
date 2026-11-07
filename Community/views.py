import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.db import transaction
from django.db.models import Q, Count
from django.core.paginator import Paginator
from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages as flash_messages
from language.utils import login_required_custom, login_required_redirect, rate_limit
from django.utils import timezone
from Dashboard.common import get_user_context, pagination_window
from Dashboard.models import Language
from Community.constants import (
    COMMUNITY_HOT_POSTS_LIMIT,
    COMMUNITY_LATEST_POSTS_LIMIT,
    COMMUNITY_TOP_USERS_LIMIT,
    COMMUNITY_TOP_TAGS_LIMIT,
    COMMUNITY_RELATED_POSTS_LIMIT,
    COMMUNITY_USER_POSTS_LIMIT,
    COMMUNITY_USER_REPLIES_LIMIT,
    COMMUNITY_USER_COLLECTS_LIMIT,
    COMMUNITY_PAGE_SIZE,
    COMMUNITY_MAX_PAGE_SIZE,
)
from User.models import User
from .models import Post, Reply, Like, Collect, Follow, Message, Report, Category, Tag, ContentModeration
from .forms import PostForm, ReplyForm, ReportForm, SearchForm
from .services import (
    increment_view, save_post, delete_post_data, search_posts_hybrid,
    get_recommendations, get_mutual_follows, get_follow_network, _log_action,
    refresh_tag_counts, _neo4j_ok,
)

logger = logging.getLogger(__name__)


def _get_unread_count(user):
    """获取用户未读消息数（缓存 30 秒）"""
    if not user:
        return 0
    cache_key = f'unread_count:{user.username}'
    count = cache.get(cache_key)
    if count is None:
        from Community.models import Message
        count = Message.objects.filter(recipient=user, is_read=False).count()
        cache.set(cache_key, count, 30)
    return count


# ==================== 社区首页 ====================
def community_home(request):
    user, name, _ = get_user_context(request)

    categories = Category.objects.filter(is_active=True).annotate(
        post_count=Count('posts', filter=Q(posts__status='published'))
    ).order_by('sort_order')
    _post_base = Post.objects.filter(status='published').select_related('author', 'category').prefetch_related('tags')

    # Redis 热门帖子排行榜（优先从 Redis 读取）
    hot_posts = None
    try:
        from .nosql_services import RedisService
        hot_ids = RedisService.get_hot_posts(limit=COMMUNITY_HOT_POSTS_LIMIT)
        if hot_ids:
            ids = [int(pid) for pid, _ in hot_ids]
            hot_posts = list(Post.objects.filter(id__in=ids, status='published')
                           .select_related('author', 'category').prefetch_related('tags'))
            # 按热度排序
            id_order = {pid: i for i, pid in enumerate(ids)}
            hot_posts.sort(key=lambda p: id_order.get(p.id, 999))
    except Exception:
        pass

    # 回退到 MySQL 排序
    if not hot_posts:
        hot_posts = _post_base.order_by('-view_count')[:COMMUNITY_HOT_POSTS_LIMIT]

    latest_posts = _post_base.order_by('-created_at')[:COMMUNITY_LATEST_POSTS_LIMIT]

    # ChromaDB 语义推荐（猜你喜欢）
    recommended_posts = []
    if user:
        try:
            recs = get_recommendations(user_id=user.username, limit=5)
            if recs:
                rec_ids = [r.get('post_id') if isinstance(r, dict) else r.get('post_id') for r in recs]
                rec_ids = [pid for pid in rec_ids if pid]
                if rec_ids:
                    recommended_posts = list(Post.objects.filter(id__in=rec_ids, status='published')
                                           .select_related('author', 'category')[:5])
        except Exception:
            pass

    top_followers = Follow.objects.select_related('following').values(
        'following__username', 'following__name'
    ).annotate(
        count=Count('id')
    ).order_by('-count')[:COMMUNITY_TOP_USERS_LIMIT]
    active_users = []
    for item in top_followers:
        if item['count'] > 0:
            active_users.append({
                'id': item['following__username'],
                'name': item['following__name'] or item['following__username'],
                'followers_count': item['count']
            })

    top_tags = Tag.objects.order_by('-post_count')[:COMMUNITY_TOP_TAGS_LIMIT]

    unread_count = 0
    username = ''
    if user:
        unread_count = _get_unread_count(user)
        username = user.username

    context = {
        'name': name,
        'username': username,
        'categories': categories,
        'hot_posts': hot_posts,
        'latest_posts': latest_posts,
        'recommended_posts': recommended_posts,
        'active_users': active_users,
        'top_tags': top_tags,
        'unread_count': unread_count,
    }
    return render(request, 'community/index.html', context)


# ==================== 帖子详情 ====================
def post_detail(request, post_id):
    user, name, _ = get_user_context(request)

    post = get_object_or_404(Post.objects.select_related('author', 'category').prefetch_related('tags'), id=post_id, status='published')
    increment_view(post)

    # MongoDB 行为日志
    if user:
        _log_action(user, 'view', post_id=post.id)

    # MongoDB 嵌套评论（一次查询拿到帖子+所有评论，无需 JOIN）
    # 包装成模板需要的格式，替换 MySQL 评论
    class _Author:
        def __init__(self, username, name):
            self.username = username
            self.name = name
            self.bio = ''
            self.pk = username

    class _Comment:
        def __init__(self, data):
            self.id = data.get('reply_id', 0)
            self.author = _Author(data.get('user_id', ''), data.get('user_name', ''))
            self.content = data.get('content', '')
            self.created_at = data.get('created_at')
            self.is_liked = False
            self.like_count = data.get('like_count', 0)
            self.children_list = []
            for child_data in data.get('children', []):
                child = _Comment(child_data)
                self.children_list.append(child)

    mongo_replies = []
    try:
        from .nosql_services import MongoService
        mongo_doc = MongoService.get_post_snapshot(post.id)
        if mongo_doc and 'comments' in mongo_doc:
            for c in mongo_doc['comments']:
                mongo_replies.append(_Comment(c))
    except Exception:
        pass

    # 如果 MongoDB 有评论就用 MongoDB，否则回退到 MySQL
    if mongo_replies:
        # 从 MySQL 补充 like_count 和 is_liked（MongoDB 不存储这些动态数据）
        def _collect_reply_ids(comments):
            ids = []
            for c in comments:
                if c.id:
                    ids.append(c.id)
                ids.extend(_collect_reply_ids(c.children_list))
            return ids

        all_mongo_reply_ids = _collect_reply_ids(mongo_replies)
        if user and all_mongo_reply_ids:
            user_likes = set(Like.objects.filter(user=user, reply_id__in=all_mongo_reply_ids).values_list('reply_id', flat=True))
        else:
            user_likes = set()

        # 从 MySQL 获取所有回复的真实 like_count
        from django.db.models import Q
        mysql_like_counts = {}
        if all_mongo_reply_ids:
            for r in Reply.objects.filter(id__in=all_mongo_reply_ids).values('id', 'like_count'):
                mysql_like_counts[r['id']] = r['like_count']

        def _apply_like_data(comments):
            for c in comments:
                c.is_liked = c.id in user_likes
                c.like_count = mysql_like_counts.get(c.id, c.like_count)
                _apply_like_data(c.children_list)

        _apply_like_data(mongo_replies)
        replies_with_children = mongo_replies
    else:
        # 无 MongoDB 时从 MySQL 构建完整回复树（支持任意层级）
        all_replies = list(Reply.objects.filter(
            post=post, is_deleted=False
        ).select_related('author').order_by('created_at'))
        all_reply_ids = [r.id for r in all_replies]
        if user and all_reply_ids:
            user_likes = set(Like.objects.filter(
                user=user, reply_id__in=all_reply_ids
            ).values_list('reply_id', flat=True))
        else:
            user_likes = set()

        children_by_parent = {}
        for reply in all_replies:
            reply.is_liked = reply.id in user_likes
            children_by_parent.setdefault(reply.parent_id, []).append(reply)

        def _build_tree(parent_id=None):
            nodes = []
            for reply in children_by_parent.get(parent_id, []):
                reply.children_list = _build_tree(reply.id)
                nodes.append(reply)
            return nodes

        replies_with_children = _build_tree(None)

    # ChromaDB 语义推荐：相似帖子
    similar_posts = []
    rec_scores = {}
    try:
        from .nosql_services import ChromaService
        recs = ChromaService.get_recommendations(post_id=post.id, n_results=COMMUNITY_RELATED_POSTS_LIMIT)
        if recs:
            rec_ids = [r['post_id'] for r in recs if r.get('post_id') != post.id and r.get('post_id')]
            for r in recs:
                pid = r.get('post_id')
                if pid and pid != post.id:
                    rec_scores[pid] = round(r.get('score', 0) * 100, 1)
            if rec_ids:
                similar_posts = list(Post.objects.filter(id__in=rec_ids, status='published')
                                   .select_related('author', 'category')[:COMMUNITY_RELATED_POSTS_LIMIT])
    except Exception:
        pass
    for sp in similar_posts:
        sp.search_score = rec_scores.get(sp.id, None)
    if not similar_posts:
        similar_posts = Post.objects.filter(
            Q(category=post.category) | Q(tags__in=post.tags.all()),
            status='published'
        ).select_related('author').prefetch_related('tags').exclude(id=post.id).distinct()[:COMMUNITY_RELATED_POSTS_LIMIT]

    is_liked = False
    is_collected = False
    is_following_author = False

    if user:
        is_liked = Like.objects.filter(user=user, post=post).exists()
        is_collected = Collect.objects.filter(user=user, post=post).exists()
        is_following_author = Follow.objects.filter(follower=user, following=post.author).exists()

    if user:
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0

    username = user.username if user else ''

    related_language_label = ''
    if post.related_language:
        related_language_label = post.related_language.legacy_level_display

    context = {
        'name': name,
        'username': username,
        'post': post,
        'replies': replies_with_children,
        'related_posts': similar_posts,
        'is_liked': is_liked,
        'is_collected': is_collected,
        'is_following_author': is_following_author,
        'reply_form': ReplyForm(),
        'unread_count': unread_count,
        'related_language_label': related_language_label,
    }
    return render(request, 'community/post_detail.html', context)


# ==================== 发帖 ====================
@login_required_redirect
def create_post(request):
    user, name, _ = get_user_context(request)

    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = user

            # 专家发布的内容直接发布，非专家需要审核
            if user.expert == 1:
                post.status = 'published'
                post.needs_moderation = False
                post.is_expert_verified = True
                post.verified_by = user
                post.verified_at = timezone.now()
            else:
                post.status = 'draft'
                post.needs_moderation = True
                post.moderation_queue = 'pending'

            if post.related_language:
                post.language_level = post.related_language.legacy_level_display

            post.save()
            form.save_m2m()

            # 同步到所有非关系型数据库
            save_post(post)

            refresh_tag_counts(post.tags.all())

            if user.expert == 1:
                Message.objects.create(
                    recipient=user,
                    msg_type='system',
                    content='您的帖子已发布成功！'
                )
                return redirect('Community:post_detail', post_id=post.id)
            else:
                Message.objects.create(
                    recipient=user,
                    msg_type='system',
                    content='您的帖子已提交，等待审核通过后将自动发布。'
                )
                return redirect('Community:community_home')
    else:
        form = PostForm()

    if user:
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0

    username = user.username if user else ''

    context = {
        'name': name,
        'username': username,
        'form': form,
        'unread_count': unread_count,
    }
    return render(request, 'community/create_post.html', context)


# ==================== 编辑帖子 ====================
@login_required_redirect
def edit_post(request, post_id):
    user, name, _ = get_user_context(request)
    post = get_object_or_404(Post, id=post_id, author=user)
    
    if request.method == 'POST':
        old_tags = list(post.tags.all())
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            form.save()
            post.refresh_from_db()
            refresh_tag_counts(set(old_tags) | set(post.tags.all()))
            save_post(post)
            from .services import clear_recommendation_cache
            clear_recommendation_cache(post.id)
            return redirect('Community:post_detail', post_id=post.id)
    else:
        form = PostForm(instance=post)
    
    if user:
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0
    
    username = user.username if user else ''
    
    context = {
        'name': name,
        'username': username,
        'form': form,
        'post': post,
        'unread_count': unread_count,
    }
    return render(request, 'community/edit_post.html', context)


# ==================== 删除帖子 ====================
@require_POST
@login_required_redirect
def delete_post(request, post_id):
    user, name, _ = get_user_context(request)
    post = get_object_or_404(Post, id=post_id)

    if post.author != user and user.admin != 1:
        return JsonResponse({'success': False, 'message': '无权删除'})

    post.status = 'hidden'
    post.save()

    # 清理非关系型数据库
    delete_post_data(post_id)
    from .services import clear_recommendation_cache
    clear_recommendation_cache(post_id)

    refresh_tag_counts(post.tags.all())

    return JsonResponse({'success': True, 'message': '删除成功'})


# ==================== 点赞帖子 ====================
@require_POST
@login_required_custom
def like_post(request, post_id):
    user, name, _ = get_user_context(request)
    post = get_object_or_404(Post, id=post_id)
    from Community.services import toggle_like
    result = toggle_like(user, post=post)
    return JsonResponse({
        'success': True,
        'is_liked': result['liked'],
        'like_count': result['like_count']
    })


# ==================== 收藏帖子 ====================
@require_POST
@login_required_custom
def collect_post(request, post_id):
    user, name, _ = get_user_context(request)
    post = get_object_or_404(Post, id=post_id)
    from Community.services import toggle_collect
    result = toggle_collect(user, post)
    return JsonResponse({
        'success': True,
        'is_collected': result['collected']
    })


# ==================== 举报帖子 ====================
@login_required_redirect
def report_post(request, post_id):
    user, name, _ = get_user_context(request)
    post = get_object_or_404(Post, id=post_id)

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == 'POST':
        form = ReportForm(request.POST)
        if form.is_valid():
            report = form.save(commit=False)
            report.reporter = user
            report.post = post
            report.save()

            if is_ajax:
                return JsonResponse({'success': True, 'message': '举报已提交，我们会尽快处理'})
            # 非 AJAX 提交：重定向回帖子详情页
            from django.contrib import messages
            messages.success(request, '举报已提交，我们会尽快处理')
            return redirect('Community:post_detail', post_id=post.id)

        if is_ajax:
            errors = {field: [str(e) for e in errs] for field, errs in form.errors.items()}
            return JsonResponse({'success': False, 'message': '请检查举报内容', 'errors': errors}, status=400)
    else:
        form = ReportForm()

    username = user.username if user else ''
    return render(request, 'community/report_post.html', {'form': form, 'post': post, 'name': name, 'username': username})


# ==================== 回复功能 ====================
@login_required_redirect
def create_reply(request, post_id):
    user, name, _ = get_user_context(request)
    post = get_object_or_404(Post, id=post_id, status='published')

    if post.is_closed:
        return JsonResponse({'success': False, 'message': '该帖子已关闭评论'})

    if request.method == 'POST':
        form = ReplyForm(request.POST)
        if form.is_valid():
            reply = form.save(commit=False)
            reply.post = post
            reply.author = user

            parent_id = request.POST.get('parent_id')
            if parent_id:
                try:
                    parent_id = int(parent_id)
                except (TypeError, ValueError):
                    return JsonResponse({'success': False, 'message': '父回复参数不正确'}, status=400)
                parent = Reply.objects.filter(
                    id=parent_id, post=post, is_deleted=False
                ).first()
                if not parent:
                    return JsonResponse({'success': False, 'message': '父回复不存在或不属于该帖子'}, status=400)
                reply.parent = parent

            reply.save()
            post.increment_reply()

            # 以 MySQL 为准重建 MongoDB 嵌套评论，保证深层回复也可见
            try:
                from .nosql_services import MongoService
                MongoService.rebuild_comments(post_id)
            except Exception:
                logger.warning('MongoDB 评论同步失败', exc_info=True)

            # 更新推荐缓存（新评论可能影响帖子热度）
            from .services import clear_recommendation_cache
            clear_recommendation_cache(post_id)
            
            if user != post.author:
                Message.objects.create(
                    recipient=post.author,
                    sender=user,
                    msg_type='reply',
                    content=f'{user.name} 回复了你的帖子「{post.title}」',
                    post=post,
                    reply=reply
                )
            
            if reply.parent and user != reply.parent.author:
                Message.objects.create(
                    recipient=reply.parent.author,
                    sender=user,
                    msg_type='reply',
                    content=f'{user.name} 回复了你的评论',
                    post=post,
                    reply=reply
                )
            
            return redirect('Community:post_detail', post_id=post.id)
    
    return redirect('Community:post_detail', post_id=post.id)


# ==================== 点赞回复 ====================
@require_POST
@login_required_custom
def like_reply(request, reply_id):
    user, name, _ = get_user_context(request)
    reply = get_object_or_404(Reply, id=reply_id)
    from Community.services import toggle_like
    result = toggle_like(user, reply=reply)
    return JsonResponse({
        'success': True,
        'is_liked': result['liked'],
        'like_count': result['like_count']
    })


# ==================== 删除回复 ====================
@require_POST
@login_required_redirect
def delete_reply(request, reply_id):
    user, name, _ = get_user_context(request)
    reply = get_object_or_404(Reply, id=reply_id)

    if reply.author != user and user.admin != 1:
        return JsonResponse({'success': False, 'message': '无权删除'})

    # 检查是否已删除，避免重复递减
    if reply.is_deleted:
        return JsonResponse({'success': False, 'message': '回复已被删除'})

    reply.is_deleted = True
    reply.save()
    reply.post.decrement_reply()

    # 以 MySQL 为准重建 MongoDB 评论
    try:
        from .nosql_services import MongoService
        MongoService.rebuild_comments(reply.post_id)
    except Exception:
        logger.warning('MongoDB 评论同步失败', exc_info=True)

    return JsonResponse({'success': True, 'message': '删除成功'})


# ==================== 用户主页 ====================

def user_profile(request, user_id):
    user, name, _ = get_user_context(request)
    profile_user = get_object_or_404(User, username=user_id, is_active=True)

    is_own_profile = bool(user and user.username == profile_user.username)
    posts = Post.objects.filter(author=profile_user, status='published').order_by('-created_at')
    replies = Reply.objects.filter(author=profile_user, is_deleted=False).order_by('-created_at')
    collects = Collect.objects.filter(user=profile_user).order_by('-created_at') if is_own_profile else Collect.objects.none()

    # 浏览足迹属于隐私数据，仅本人可见
    footprints = []
    if is_own_profile:
        try:
            from .nosql_services import get_mongo_db
            db = get_mongo_db()
            logs = list(db.user_logs.find(
                {'user_id': profile_user.username, 'action': 'view', 'post_id': {'$ne': None}}
            ).sort('created_at', -1).limit(20))
            seen_posts = set()
            for log in logs:
                pid = log.get('post_id')
                if pid and pid not in seen_posts:
                    seen_posts.add(pid)
                    footprints.append({
                        'post_id': pid,
                        'created_at': log.get('created_at'),
                    })
        except Exception:
            pass

    # 批量查询足迹对应的帖子，构建 (post, timestamp) 列表
    footprint_items = []
    if footprints:
        fp_ids = [fp['post_id'] for fp in footprints]
        posts_map = {}
        for p in Post.objects.filter(id__in=fp_ids, status='published').select_related('author'):
            posts_map[p.id] = p
        for fp in footprints:
            post = posts_map.get(fp['post_id'])
            if post:
                footprint_items.append({'post': post, 'viewed_at': fp['created_at']})

    followers_count = Follow.objects.filter(following=profile_user).count()
    following_count = Follow.objects.filter(follower=profile_user).count()
    is_following = False

    if user:
        is_following = Follow.objects.filter(follower=user, following=profile_user).exists()
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0

    paginator = Paginator(posts, 10)

    # 安全分页：限制页码在有效范围内
    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
        if page_number < 1:
            page_number = 1
        elif page_number > COMMUNITY_MAX_PAGE_SIZE:
            page_number = COMMUNITY_MAX_PAGE_SIZE
    except (ValueError, TypeError):
        page_number = 1

    page_obj = paginator.get_page(page_number)

    username = user.username if user else ''

    context = {
        'name': name,
        'username': username,
        'profile_user': profile_user,
        'profile': getattr(profile_user, 'profile', None),
        'is_own_profile': is_own_profile,
        'posts': page_obj,
        'replies': replies[:COMMUNITY_USER_REPLIES_LIMIT],
        'collects': collects[:COMMUNITY_USER_COLLECTS_LIMIT],
        'footprint_items': footprint_items[:15],
        'followers_count': followers_count,
        'following_count': following_count,
        'is_following': is_following,
        'unread_count': unread_count,
    }
    return render(request, 'community/user_profile.html', context)


# ==================== 关注用户 ====================
@require_POST
@login_required_custom
def follow_user(request, user_id):
    user, name, _ = get_user_context(request)
    target_user = get_object_or_404(User, username=user_id, is_active=True)
    from Community.services import toggle_follow
    result = toggle_follow(user, target_user)
    if result['error']:
        return JsonResponse({'success': False, 'message': result['error']})
    return JsonResponse({
        'success': True,
        'is_following': result['following']
    })


# ==================== 发私信 ====================
@rate_limit(max_requests=5, window_seconds=60)
@login_required_redirect
def send_message(request, user_id):
    """给其他用户发送私信。"""
    user, name, _ = get_user_context(request)
    recipient = get_object_or_404(User, username=user_id, is_active=True)

    if recipient.username == user.username:
        flash_messages.error(request, '不能给自己发私信')
        return redirect('Community:user_profile', user_id=user.username)

    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        if not content:
            flash_messages.error(request, '私信内容不能为空')
        elif len(content) > 500:
            flash_messages.error(request, '私信内容不能超过 500 个字符')
        else:
            Message.objects.create(
                recipient=recipient,
                sender=user,
                msg_type='mention',
                content=content,
            )
            flash_messages.success(request, '私信已发送')
            return redirect('Community:user_profile', user_id=recipient.username)

    return render(request, 'community/send_message.html', {
        'name': name,
        'username': user.username,
        'recipient': recipient,
    })


# ==================== 消息中心 ====================
@login_required_redirect
def messages(request):
    user, name, _ = get_user_context(request)

    msg_list = Message.objects.filter(recipient=user).order_by('-created_at')

    paginator = Paginator(msg_list, 20)

    # 安全分页
    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
        if page_number < 1:
            page_number = 1
        elif page_number > COMMUNITY_MAX_PAGE_SIZE:
            page_number = COMMUNITY_MAX_PAGE_SIZE
    except (ValueError, TypeError):
        page_number = 1

    page_obj = paginator.get_page(page_number)
    
    unread_count = _get_unread_count(user)
    username = user.username if user else ''
    
    context = {
        'name': name,
        'username': username,
        'messages': page_obj,
        'unread_count': unread_count,
    }
    return render(request, 'community/messages.html', context)


# ==================== 标记消息已读 ====================
@require_POST
@login_required_custom
def mark_read_messages(request):
    user, name, _ = get_user_context(request)
    Message.objects.filter(recipient=user, is_read=False).update(is_read=True)
    cache.delete(f'unread_count:{user.username}')
    return JsonResponse({'success': True})


# ==================== 搜索帖子 ====================
def search_posts(request):
    user, name, _ = get_user_context(request)
    form = SearchForm(request.GET)

    posts = Post.objects.filter(status='published').select_related('author', 'category').prefetch_related('tags')
    keyword = ''
    search_scores = {}  # post_id -> similarity score
    search_sources = {}  # post_id -> 'semantic' / 'keyword'

    if form.is_valid():
        keyword = form.cleaned_data.get('keyword', '')
        category = form.cleaned_data.get('category')
        tags = form.cleaned_data.get('tags')

        if keyword:
            # 混合搜索：ChromaDB 语义搜索 + MySQL 精确匹配
            try:
                from .services import search_posts_hybrid
                hybrid_results = search_posts_hybrid(keyword, category=category, tag=tags, limit=30)
                if hybrid_results:
                    post_ids = [r['post_id'] for r in hybrid_results]
                    # 记录每个帖子的匹配分数与来源
                    for r in hybrid_results:
                        score = r.get('score', 0)
                        search_scores[r['post_id']] = round(score * 100, 1)
                        search_sources[r['post_id']] = r.get('source', 'keyword')
                    posts = posts.filter(id__in=post_ids)
                    # 按搜索结果排序
                    from django.db.models import Case, When
                    preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(post_ids)])
                    posts = posts.order_by(preserved)
                else:
                    posts = posts.filter(
                        Q(title__icontains=keyword) | Q(content__icontains=keyword)
                    )
            except Exception:
                posts = posts.filter(
                    Q(title__icontains=keyword) | Q(content__icontains=keyword)
                )

        if category:
            posts = posts.filter(category_id=category)

        if tags:
            posts = posts.filter(tags__id=tags).distinct()

    # 记录搜索行为
    if user and keyword:
        _log_action(user, 'search', extra={'keyword': keyword})

    # 只在没有关键词时按时间排序
    if not keyword:
        posts = posts.order_by('-created_at')

    paginator = Paginator(posts, 15)

    # 安全分页
    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
        if page_number < 1:
            page_number = 1
        elif page_number > COMMUNITY_MAX_PAGE_SIZE:
            page_number = COMMUNITY_MAX_PAGE_SIZE
    except (ValueError, TypeError):
        page_number = 1

    page_obj = paginator.get_page(page_number)

    # 给分页后的帖子附加搜索分数与来源
    for post in page_obj:
        post.search_score = search_scores.get(post.id, None)
        post.search_source = search_sources.get(post.id, None)

    if user:
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0

    username = user.username if user else ''

    context = {
        'name': name,
        'username': username,
        'form': form,
        'posts': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
        'keyword': keyword,
        'unread_count': unread_count,
    }
    return render(request, 'community/search.html', context)


# ==================== 分类帖子列表 ====================
def category_posts(request, category_id):
    user, name, _ = get_user_context(request)
    category = get_object_or_404(Category, id=category_id, is_active=True)

    posts = Post.objects.filter(category=category, status='published').select_related('author', 'category').prefetch_related('tags').order_by('-created_at')

    paginator = Paginator(posts, 15)

    # 安全分页
    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
        if page_number < 1:
            page_number = 1
        elif page_number > COMMUNITY_MAX_PAGE_SIZE:
            page_number = COMMUNITY_MAX_PAGE_SIZE
    except (ValueError, TypeError):
        page_number = 1

    page_obj = paginator.get_page(page_number)
    
    if user:
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0
    
    username = user.username if user else ''
    
    context = {
        'name': name,
        'username': username,
        'category': category,
        'posts': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
        'unread_count': unread_count,
    }
    return render(request, 'community/category_posts.html', context)


# ==================== 标签帖子列表 ====================
def tag_posts(request, tag_id):
    user, name, _ = get_user_context(request)
    tag = get_object_or_404(Tag, id=tag_id)

    posts = Post.objects.filter(tags=tag, status='published').select_related('author', 'category').prefetch_related('tags').order_by('-created_at')

    paginator = Paginator(posts, 15)

    # 安全分页
    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
        if page_number < 1:
            page_number = 1
        elif page_number > COMMUNITY_MAX_PAGE_SIZE:
            page_number = COMMUNITY_MAX_PAGE_SIZE
    except (ValueError, TypeError):
        page_number = 1

    page_obj = paginator.get_page(page_number)
    
    if user:
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0
    
    username = user.username if user else ''
    
    context = {
        'name': name,
        'username': username,
        'tag': tag,
        'posts': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
        'unread_count': unread_count,
    }
    return render(request, 'community/tag_posts.html', context)


# ==================== 内容审核管理 ====================

def moderation_queue(request):
    """审核队列页面"""
    user, name, admin = get_user_context(request)

    if not user:
        return redirect(f'/User/login/?next={request.path}')

    if admin != 1 and user.expert != 1:
        return render(request, '403.html', {'message': '无权访问审核队列'}, status=403)

    # 获取待审核内容
    pending_posts = Post.objects.filter(
        needs_moderation=True,
        moderation_queue='pending'
    ).select_related('author', 'category').order_by('created_at')

    pending_replies = Reply.objects.filter(
        post__needs_moderation=True,
        post__moderation_queue='pending',
        is_deleted=False
    ).select_related('author', 'post').order_by('created_at')[:50]

    # 已审核历史
    moderation_history = ContentModeration.objects.filter(
        moderator=user
    ).order_by('-created_at')[:50]

    if user:
        unread_count = _get_unread_count(user)
    else:
        unread_count = 0

    username = user.username if user else ''

    context = {
        'name': name,
        'username': username,
        'pending_posts': pending_posts,
        'pending_replies': pending_replies,
        'moderation_history': moderation_history,
        'unread_count': unread_count,
    }
    return render(request, 'community/moderation_queue.html', context)


@login_required_custom
def moderate_content(request):
    """执行内容审核操作"""
    user, name, admin = get_user_context(request)

    if admin != 1 and user.expert != 1:
        return JsonResponse({'success': False, 'message': '无权审核'}, status=403)

    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '仅支持POST请求'}, status=405)

    content_type = request.POST.get('type')
    content_id = request.POST.get('id')
    action = request.POST.get('action')
    reason = request.POST.get('reason', '')

    if not all([content_type, content_id, action]):
        return JsonResponse({'success': False, 'message': '参数不完整'}, status=400)

    if content_type == 'post':
        try:
            content = Post.objects.get(pk=content_id)
        except Post.DoesNotExist:
            return JsonResponse({'success': False, 'message': '帖子不存在'}, status=404)
    else:
        try:
            content = Reply.objects.get(pk=content_id)
        except Reply.DoesNotExist:
            return JsonResponse({'success': False, 'message': '回复不存在'}, status=404)

    # 记录审核操作
    ContentModeration.objects.create(
        content_type=content_type,
        content_id=content_id,
        moderator=user,
        action=action,
        reason=reason
    )

    # 执行操作
    if action == 'approve':
        if content_type == 'post':
            content.status = 'published'
            content.needs_moderation = False
            content.moderation_queue = 'approved'
        content.save()
    elif action == 'reject':
        if content_type == 'post':
            content.status = 'hidden'
            content.needs_moderation = False
            content.moderation_queue = 'rejected'
        else:
            content.is_deleted = True
        content.save()
    elif action == 'hide':
        if content_type == 'post':
            content.status = 'hidden'
        else:
            content.is_deleted = True
        content.save()
    elif action == 'restore':
        if content_type == 'post':
            content.status = 'published'
            content.needs_moderation = False
        else:
            content.is_deleted = False
        content.save()

    if content_type == 'post':
        refresh_tag_counts(content.tags.all())

    # 发送通知
    if action in ['approve', 'reject', 'hide']:
        action_text = {'approve': '通过', 'reject': '拒绝', 'hide': '隐藏'}[action]
        Message.objects.create(
            recipient=content.author,
            msg_type='system',
            content=f'您发布的内容已被管理员{action_text}。',
            post=content if content_type == 'post' else None
        )

    return JsonResponse({'success': True})


# ==================== 举报处理 ====================
@login_required_redirect
def report_queue(request):
    """管理员/专家处理用户举报。"""
    user, name, admin = get_user_context(request)
    if admin != 1 and user.expert != 1:
        return render(request, '403.html', {'message': '无权访问举报队列'}, status=403)

    pending_reports = Report.objects.filter(status='pending').select_related(
        'reporter', 'post', 'post__author', 'reply', 'reply__author'
    ).order_by('created_at')
    handled_reports = Report.objects.exclude(status='pending').select_related(
        'reporter', 'post', 'reply'
    ).order_by('-created_at')[:30]
    unread_count = _get_unread_count(user)

    return render(request, 'community/report_queue.html', {
        'name': name,
        'username': user.username,
        'pending_reports': pending_reports,
        'handled_reports': handled_reports,
        'unread_count': unread_count,
    })


@require_POST
@login_required_redirect
def report_action(request):
    """处理举报：隐藏内容或驳回举报。"""
    user, name, admin = get_user_context(request)
    if admin != 1 and user.expert != 1:
        return JsonResponse({'success': False, 'message': '无权处理举报'}, status=403)

    report_id = request.POST.get('id')
    action = request.POST.get('action')
    if action not in ('hide', 'dismiss'):
        return JsonResponse({'success': False, 'message': '无效操作'}, status=400)

    report = get_object_or_404(Report.objects.select_related('post', 'reply'), id=report_id)

    with transaction.atomic():
        if action == 'dismiss':
            report.status = 'dismissed'
            report.save(update_fields=['status'])
            flash_messages.success(request, '举报已驳回')
        else:
            report.status = 'processed'
            report.save(update_fields=['status'])
            if report.post:
                report.post.status = 'hidden'
                report.post.needs_moderation = False
                report.post.save(update_fields=['status', 'needs_moderation'])
                ContentModeration.objects.create(
                    content_type='post', content_id=report.post_id,
                    moderator=user, action='hide', reason=report.description,
                )
                refresh_tag_counts(report.post.tags.all())
            elif report.reply:
                report.reply.is_deleted = True
                report.reply.save(update_fields=['is_deleted'])
                report.reply.post.decrement_reply()
                ContentModeration.objects.create(
                    content_type='reply', content_id=report.reply_id,
                    moderator=user, action='hide', reason=report.description,
                )
            flash_messages.success(request, '举报已处理，内容已隐藏')

    return JsonResponse({'success': True})


# ==================== 用户收藏页面 ====================
@login_required_redirect
def user_collections(request):
    """用户收藏列表页面"""
    user, name, _ = get_user_context(request)

    collects = Collect.objects.filter(user=user).select_related(
        'post', 'post__author', 'post__category'
    ).order_by('-created_at')

    paginator = Paginator(collects, 15)

    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
        if page_number < 1:
            page_number = 1
    except (ValueError, TypeError):
        page_number = 1

    page_obj = paginator.get_page(page_number)

    context = {
        'name': name,
        'username': user.username,
        'collects': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
    }

    return render(request, 'community/user_collections.html', context)


# ==================== 用户声望排行 ====================
def user_reputation_rank(request):
    """用户声望排行榜：实时计算，避免 reputation_score 长期为 0。"""
    user, name, _ = get_user_context(request)

    from django.db.models import ExpressionWrapper, F, IntegerField

    top_users = User.objects.filter(is_active=True).annotate(
        post_count=Count('posts', filter=Q(posts__status='published'), distinct=True),
        like_received=Count('posts__like_users', filter=Q(posts__status='published'), distinct=True),
        reply_received=Count(
            'posts__replies',
            filter=Q(posts__status='published', posts__replies__is_deleted=False),
            distinct=True,
        ),
        score=ExpressionWrapper(
            F('post_count') * 10 + F('like_received') * 5 + F('reply_received') * 3
            + F('expert') * 100 + F('admin') * 50,
            output_field=IntegerField(),
        ),
    ).order_by('-score', '-post_count', 'username')[:20]

    for item in top_users:
        item.reputation_score = item.score

    context = {
        'name': name,
        'username': user.username if user else '',
        'top_users': top_users,
    }

    return render(request, 'community/reputation_rank.html', context)


# ==================== 社交图谱 API（Neo4j） ====================

@require_POST
@login_required_custom
def api_follow_network(request, user_id):
    """获取用户关注网络图谱（用于 ECharts 可视化）"""
    try:
        from .services import get_follow_network
        try:
            depth = int(request.POST.get('depth', 2))
        except (TypeError, ValueError):
            depth = 2
        depth = min(max(depth, 1), 3)
        data = get_follow_network(user_id, depth=depth)
        return JsonResponse({'success': True, 'available': True, 'data': data})
    except Exception:
        logger.exception('获取关注网络失败')
        return JsonResponse({
            'success': False,
            'available': False,
            'reason': 'neo4j_unavailable',
            'message': '社交图谱服务当前不可用，已降级显示',
        })


@require_POST
@login_required_custom
def api_mutual_follows(request, user_id):
    """获取共同关注推荐"""
    try:
        from .services import get_mutual_follows
        try:
            limit = int(request.POST.get('limit', 10))
        except (TypeError, ValueError):
            limit = 10
        limit = min(max(limit, 1), 50)
        data = get_mutual_follows(user_id, limit=limit)
        return JsonResponse({'success': True, 'available': True, 'data': data})
    except Exception:
        logger.exception('获取共同关注失败')
        return JsonResponse({
            'success': False,
            'available': False,
            'reason': 'neo4j_unavailable',
            'message': '共同关注服务当前不可用，已降级显示',
        })


# ==================== 社交图谱可视化页面 ====================

@login_required_redirect
def social_graph(request, user_id):
    """社交图谱可视化页面（Neo4j + ECharts）"""
    user, name, _ = get_user_context(request)
    profile_user = get_object_or_404(User, username=user_id, is_active=True)

    unread_count = _get_unread_count(user) if user else 0

    # 获取 Neo4j 里使用的 id（Neo4j 节点 id 存的是 username，不是 name）
    neo4j_id = profile_user.username

    # 服务端预判图谱服务可用性：不可用时页面直接展示降级提示，
    # 前端不再发请求，避免出现没有任何说明的空图。
    graph_available = _neo4j_ok()

    context = {
        'name': name,
        'username': user.username if user else '',
        'profile_user': profile_user,
        'neo4j_id': neo4j_id,
        'graph_available': graph_available,
        'unread_count': unread_count,
    }
    return render(request, 'community/social_graph.html', context)
