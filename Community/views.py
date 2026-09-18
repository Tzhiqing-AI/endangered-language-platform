import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Q, Count
from django.core.paginator import Paginator
from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from language.utils import login_required_custom, login_required_redirect
from django.utils import timezone
from Dashboard.common import get_user_context
from Dashboard.common import ENDANGERMENT_LEVELS
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
        replies = Reply.objects.filter(post=post, parent=None, is_deleted=False).select_related('author').order_by('created_at')
        reply_ids = list(replies.values_list('id', flat=True))
        children_replies = Reply.objects.filter(parent__in=reply_ids, is_deleted=False).order_by('created_at')
        children_by_parent = {}
        for child in children_replies:
            if child.parent_id not in children_by_parent:
                children_by_parent[child.parent_id] = []
            children_by_parent[child.parent_id].append(child)
        all_reply_ids = reply_ids + [c.id for c in children_replies]
        if user and all_reply_ids:
            user_likes = set(Like.objects.filter(user=user, reply_id__in=all_reply_ids).values_list('reply_id', flat=True))
        else:
            user_likes = set()
        replies_with_children = []
        for reply in replies:
            reply_children = children_by_parent.get(reply.id, [])
            for child in reply_children:
                child.is_liked = child.id in user_likes
            reply.is_liked = reply.id in user_likes
            reply.children_list = reply_children
            replies_with_children.append(reply)

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
        related_language_label = ENDANGERMENT_LEVELS.get(
            post.related_language.level_of_endangerment, '未知'
        )

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
                post.language_level = ENDANGERMENT_LEVELS.get(
                    post.related_language.level_of_endangerment, ''
                )

            post.save()
            form.save_m2m()

            # 同步到所有非关系型数据库
            save_post(post)

            # 更新所有关联标签的 post_count
            for tag in post.tags.all():
                tag.post_count = Post.objects.filter(tags=tag, status='published').count()
                tag.save()

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
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            form.save()
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

    # 更新所有关联标签的 post_count
    for tag in post.tags.all():
        tag.post_count = Post.objects.filter(tags=tag, status='published').count()
        tag.save()

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
    else:
        form = ReportForm()

    username = user.username if user else ''
    return render(request, 'community/report_post.html', {'form': form, 'post': post, 'name': name, 'username': username})


# ==================== 回复功能 ====================
@login_required_redirect
def create_reply(request, post_id):
    user, name, _ = get_user_context(request)
    post = get_object_or_404(Post, id=post_id)
    
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
                reply.parent_id = int(parent_id)
            
            reply.save()
            post.increment_reply()

            # 同步评论到 MongoDB（嵌套文档结构）
            try:
                from .nosql_services import MongoService
                MongoService.add_comment(post_id, reply)
            except Exception:
                pass

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

    # 同步删除 MongoDB 中的评论
    try:
        from .nosql_services import MongoService
        MongoService.remove_comment(reply.post_id, reply_id)
    except Exception:
        pass

    return JsonResponse({'success': True, 'message': '删除成功'})


# ==================== 用户主页 ====================

def user_profile(request, user_id):
    user, name, _ = get_user_context(request)
    profile_user = get_object_or_404(User, username=user_id)

    posts = Post.objects.filter(author=profile_user, status='published').order_by('-created_at')
    replies = Reply.objects.filter(author=profile_user, is_deleted=False).order_by('-created_at')
    collects = Collect.objects.filter(user=profile_user).order_by('-created_at')

    # MongoDB 浏览足迹
    footprints = []
    try:
        from .nosql_services import get_mongo_db
        db = get_mongo_db()
        logs = list(db.user_logs.find(
            {'user_id': profile_user.username, 'action': 'view', 'post_id': {'$ne': None}}
        ).sort('created_at', -1).limit(20))
        # 去重（同一个帖子只显示一次，保留最新记录）
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
    target_user = get_object_or_404(User, username=user_id)
    from Community.services import toggle_follow
    result = toggle_follow(user, target_user)
    if result['error']:
        return JsonResponse({'success': False, 'message': result['error']})
    return JsonResponse({
        'success': True,
        'is_following': result['following']
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
@login_required_custom
def mark_read_messages(request):
    user, name, _ = get_user_context(request)
    Message.objects.filter(recipient=user, is_read=False).update(is_read=True)
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
    }

    return render(request, 'community/user_collections.html', context)


# ==================== 用户声望排行 ====================
def user_reputation_rank(request):
    """用户声望排行榜"""
    user, name, _ = get_user_context(request)

    # 计算声望分数：帖子数*10 + 收到的点赞数*5 + 收到的回复数*3 + 专家认证*100
    active_users = User.objects.filter(is_active=True)

    # 如果声望分数都是0，则按帖子数排序（注意：切片后的 QuerySet 不能再 filter）
    if not active_users.filter(reputation_score__gt=0).exists():
        top_users = active_users.annotate(
            post_count=Count('posts'),
            total_likes=Count('posts__like_users'),
        ).order_by('-post_count')[:20]
    else:
        top_users = active_users.order_by('-reputation_score')[:20]

    context = {
        'name': name,
        'username': user.username if user else '',
        'top_users': top_users,
    }

    return render(request, 'community/reputation_rank.html', context)


    def _calculate_user_reputation(user):
        """
        计算用户声望分数

        声望计算规则：
        - 发帖数 * 10
        - 收到的点赞数 * 5
        - 收到的回复数 * 3
        - 专家认证 +100
        - 管理员 +50
        """
        from django.db.models import Count

        post_count = Post.objects.filter(author=user, status='published').count()

        # 收到的点赞数
        like_count = Like.objects.filter(
            post__author=user
        ).count()

        # 收到的回复数
        reply_count = Reply.objects.filter(
            post__author=user,
            is_deleted=False
        ).count()

        # 基础分数
        score = post_count * 10 + like_count * 5 + reply_count * 3

        # 专家加分
        if user.expert == 1:
            score += 100

        # 管理员加分
        if user.admin == 1:
            score += 50

        return score


# ==================== 社交图谱 API（Neo4j） ====================

@require_POST
@login_required_custom
def api_follow_network(request, user_id):
    """获取用户关注网络图谱（用于 ECharts 可视化）"""
    # #region debug-point D:api-entry
    import json, urllib.request, os
    _p = '.dbg/echarts-not-rendering.env'
    _u, _s = 'http://127.0.0.1:7777/event', 'echarts-not-rendering'
    try:
        with open(_p) as f:
            c = f.read()
            _u = next((l.split('=', 1)[1] for l in c.split('\n') if l.startswith('DEBUG_SERVER_URL=')), _u)
            _s = next((l.split('=', 1)[1] for l in c.split('\n') if l.startswith('DEBUG_SESSION_ID=')), _s)
    except Exception:
        pass
    def _dbg(hid, msg, data):
        try:
            urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({"sessionId": _s, "runId": "post-fix", "hypothesisId": hid, "location": "views.py:api_follow_network", "msg": "[DEBUG] " + msg, "data": data}).encode(), headers={"Content-Type": "application/json"}), timeout=1)
        except Exception:
            pass
    # #endregion
    try:
        # Neo4j 里节点 id 存的是 username，不是 name
        from User.models import User
        try:
            user = User.objects.get(username=user_id)
            neo4j_id = user.username
        except User.DoesNotExist:
            neo4j_id = user_id

        # #region debug-point E:url-to-neo4j-id
        _dbg('E', 'url-to-neo4j-id', {"url_user_id": user_id, "resolved_neo4j_id": neo4j_id, "user_name": user.name if 'user' in dir() and user else None, "user_username": user.username if 'user' in dir() and user else None})
        # #endregion

        from .services import get_follow_network
        depth = int(request.POST.get('depth', 2))
        depth = min(depth, 3)
        data = get_follow_network(neo4j_id, depth=depth)

        # #region debug-point D:api-return
        _dbg('D', 'api-return', {"neo4j_id": neo4j_id, "depth": depth, "node_count": len(data.get('nodes', [])), "edge_count": len(data.get('edges', [])), "first_nodes": data.get('nodes', [])[:3], "first_edges": data.get('edges', [])[:3]})
        # #endregion

        return JsonResponse({'success': True, 'data': data})
    except Exception as e:
        # #region debug-point B:api-exception
        _dbg('B', 'api-exception', {"error": str(e)})
        # #endregion
        return JsonResponse({'success': False, 'message': str(e)})


@require_POST
@login_required_custom
def api_mutual_follows(request, user_id):
    """获取共同关注推荐"""
    try:
        from .services import get_mutual_follows
        limit = int(request.POST.get('limit', 10))
        data = get_mutual_follows(user_id, limit=limit)
        return JsonResponse({'success': True, 'data': data})
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})


# ==================== 社交图谱可视化页面 ====================

@login_required_redirect
def social_graph(request, user_id):
    """社交图谱可视化页面（Neo4j + ECharts）"""
    user, name, _ = get_user_context(request)
    profile_user = get_object_or_404(User, username=user_id)

    unread_count = _get_unread_count(user) if user else 0

    # 获取 Neo4j 里使用的 id（Neo4j 节点 id 存的是 username，不是 name）
    neo4j_id = profile_user.username

    context = {
        'name': name,
        'username': user.username if user else '',
        'profile_user': profile_user,
        'neo4j_id': neo4j_id,
        'unread_count': unread_count,
    }
    return render(request, 'community/social_graph.html', context)
