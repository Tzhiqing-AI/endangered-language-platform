"""
Community 服务层

封装社区核心业务逻辑，供模板视图和 REST API 共享使用。
集成 Redis / MongoDB / Neo4j / ChromaDB 四种非关系型数据库。
"""
import logging
from django.db.models import F
from django.conf import settings

from .models import Collect, Follow, Like, Message, Post, Reply

logger = logging.getLogger(__name__)


# =============================================================================
# 便捷访问 NoSQL 服务（带容错）
# =============================================================================

def _redis():
    try:
        from .nosql_services import get_redis
        return get_redis()
    except Exception as e:
        logger.warning(f'Redis 连接失败: {e}')
        return None


def _mongo():
    try:
        from .nosql_services import get_mongo_db
        return get_mongo_db()
    except Exception as e:
        logger.warning(f'MongoDB 连接失败: {e}')
        return None


def _neo4j_ok():
    """检查 Neo4j 是否可用"""
    try:
        from .nosql_services import get_neo4j_driver
        driver = get_neo4j_driver()
        with driver.session() as s:
            s.run('RETURN 1')
        return True
    except Exception:
        return False


def _chroma_ok():
    """检查 ChromaDB 是否可用"""
    try:
        from .nosql_services import get_chroma_collection
        get_chroma_collection()
        return True
    except Exception:
        return False


# =============================================================================
# 帖子浏览（Redis 消息队列异步更新）
# =============================================================================

def _sync_hot_score(post):
    """同步帖子热度分数到 Redis"""
    try:
        from .nosql_services import RedisService
        import time
        # 用帖子创建时间作为基础分，保证新帖出现在排行榜
        created_ts = post.created_at.timestamp() if post.created_at else time.time()
        base_score = created_ts / 1000000
        score = base_score + post.view_count * 0.3 + post.like_count * 0.5 + post.reply_count * 0.2
        RedisService.update_hot_post(post.id, score, post.category_id)
        # 同时更新全站排行
        RedisService.update_hot_post(post.id, score, None)
    except Exception:
        pass


def increment_view(post):
    """浏览帖子：Redis 异步队列 + MySQL 直接更新"""
    # MySQL 直接更新（保底）
    Post.objects.filter(pk=post.pk).update(view_count=F('view_count') + 1)
    post.refresh_from_db(fields=['view_count'])

    # 实时更新 Redis 热度
    _sync_hot_score(post)

    # 浏览行为由调用方按需写入 MongoDB；这里不再向无人消费的 Redis 队列堆积。


# =============================================================================
# 点赞（MySQL + MongoDB 行为日志）
# =============================================================================

def _sync_reply_like_to_mongo(reply):
    """同步回复点赞数到 MongoDB 嵌套评论"""
    try:
        from .nosql_services import get_mongo_db
        db = get_mongo_db()
        # 更新一级评论
        result = db.posts.update_one(
            {'comments.reply_id': reply.pk},
            {'$set': {'comments.$.like_count': reply.like_count}}
        )
        if result.matched_count == 0:
            # 尝试更新子评论
            db.posts.update_one(
                {'comments.children.reply_id': reply.pk},
                {'$set': {'comments.$.children.$.like_count': reply.like_count}}
            )
    except Exception:
        pass

def toggle_like(user, post=None, reply=None):
    """切换点赞状态

    Args:
        user: 点赞用户
        post: 帖子对象（帖子点赞时传入）
        reply: 回复对象（回复点赞时传入）

    Returns:
        dict: {'liked': bool, 'like_count': int}
    """
    if post is None and reply is None:
        raise ValueError('post 和 reply 不能同时为空')

    if post:
        like = Like.objects.filter(user=user, post=post, reply__isnull=True).first()
        if like:
            like.delete()
            post.decrement_like()
            _log_action(user, 'unlike', post_id=post.id)
            _sync_hot_score(post)
            return {'liked': False, 'like_count': post.like_count}
        Like.objects.create(user=user, post=post, reply=None)
        post.increment_like()
        _log_action(user, 'like', post_id=post.id)
        _sync_hot_score(post)
        if user != post.author:
            _notify(post.author, user, 'like', f'{user.name} 赞了你的帖子「{post.title}」', post=post)
        return {'liked': True, 'like_count': post.like_count}

    # reply — 用 filter().first() 替代 get_or_create，避免 NULL 列在 MySQL UNIQUE 约束下的问题
    like = Like.objects.filter(user=user, reply=reply, post__isnull=True).first()
    if like:
        like.delete()
        Reply.objects.filter(pk=reply.pk, like_count__gt=0).update(like_count=F('like_count') - 1)
        reply.refresh_from_db()
        _sync_reply_like_to_mongo(reply)
        return {'liked': False, 'like_count': reply.like_count}
    Like.objects.create(user=user, reply=reply, post=None)
    Reply.objects.filter(pk=reply.pk).update(like_count=F('like_count') + 1)
    reply.refresh_from_db()
    _sync_reply_like_to_mongo(reply)
    if user != reply.author:
        _notify(reply.author, user, 'like', f'{user.name} 赞了你的回复', reply=reply)
    return {'liked': True, 'like_count': reply.like_count}


# =============================================================================
# 收藏
# =============================================================================

def toggle_collect(user, post):
    """切换收藏状态

    Returns:
        dict: {'collected': bool}
    """
    collect, created = Collect.objects.get_or_create(user=user, post=post)
    if not created:
        collect.delete()
        _log_action(user, 'uncollect', post_id=post.id)
        return {'collected': False}
    _log_action(user, 'collect', post_id=post.id)
    if user != post.author:
        _notify(post.author, user, 'system', f'{user.name} 收藏了你的帖子「{post.title}」', post=post)
    return {'collected': True}


# =============================================================================
# 关注（MySQL + Neo4j 图同步）
# =============================================================================

def toggle_follow(follower, following):
    """切换关注状态（同步到 Neo4j 图数据库）

    Returns:
        dict: {'following': bool, 'error': str | None}
    """
    if follower == following:
        return {'following': False, 'error': '不能关注自己'}

    follow, created = Follow.objects.get_or_create(follower=follower, following=following)
    if not created:
        follow.delete()
        # 同步删除 Neo4j 关系
        try:
            from .nosql_services import Neo4jService
            Neo4jService.sync_follow(follower, following, created=False)
        except Exception:
            pass
        return {'following': False, 'error': None}

    # 同步创建 Neo4j 关系
    try:
        from .nosql_services import Neo4jService
        Neo4jService.sync_user(follower)
        Neo4jService.sync_user(following)
        Neo4jService.sync_follow(follower, following, created=True)
    except Exception:
        pass

    _notify(following, follower, 'follow', f'{follower.name} 关注了你')
    return {'following': True, 'error': None}


# =============================================================================
# 帖子发布/更新（MySQL + MongoDB + ChromaDB + Neo4j）
# =============================================================================

def save_post(post, tags=None):
    """保存帖子（同步到所有数据库）

    调用方式：在 views.py 中 post.save() 后调用
    """
    # MongoDB 帖子快照
    try:
        from .nosql_services import MongoService
        MongoService.save_post_snapshot(post)
    except Exception as e:
        logger.debug(f'MongoDB 同步帖子失败: {e}')

    # ChromaDB 向量入库
    try:
        from .nosql_services import ChromaService
        ChromaService.add_post(post)
    except Exception as e:
        logger.debug(f'ChromaDB 同步帖子失败: {e}')

    # Redis 更新排行榜（新帖给基础分，保证出现在热门列表）
    try:
        from .nosql_services import RedisService
        import time
        # 新帖基础分：发布时间戳/1000000，保证新帖排在前面
        base_score = time.time() / 1000000
        score = base_score + post.view_count * 0.3 + post.like_count * 0.5 + post.reply_count * 0.2
        RedisService.update_hot_post(post.id, score, post.category_id)
        RedisService.update_hot_post(post.id, score, None)
    except Exception:
        pass

    # 新帖发布后清除所有推荐缓存，让旧帖的推荐包含新帖
    clear_recommendation_cache()


def refresh_tag_counts(tags):
    """重算指定标签的已发布帖子数，避免审核/编辑后计数过期。"""
    for tag in tags:
        tag.post_count = Post.objects.filter(tags=tag, status='published').count()
        tag.save(update_fields=['post_count'])


def delete_post_data(post_id):
    """删除帖子时清理所有数据库"""
    try:
        from .nosql_services import MongoService
        from .nosql_services import get_mongo_db
        db = get_mongo_db()
        db.posts.delete_one({'post_id': int(post_id)})
    except Exception:
        pass

    try:
        from .nosql_services import ChromaService
        ChromaService.remove_post(post_id)
    except Exception:
        pass


# =============================================================================
# 搜索（MySQL 关键词 + ChromaDB 语义）
# =============================================================================

def search_posts_hybrid(keyword, category=None, tag=None, limit=20):
    """混合搜索：ChromaDB 语义搜索 + MySQL 精确过滤"""
    results = []

    # 1. ChromaDB 语义搜索
    try:
        from .nosql_services import ChromaService
        chroma_results = ChromaService.search_posts(keyword, n_results=limit)
        if chroma_results and chroma_results['ids'][0]:
            for doc_id, meta, dist in zip(
                chroma_results['ids'][0],
                chroma_results['metadatas'][0],
                chroma_results['distances'][0],
            ):
                results.append({
                    'post_id': meta['post_id'],
                    'score': 1 - dist,  # 余弦距离转相似度
                    'source': 'semantic',
                })
    except Exception as e:
        logger.debug(f'ChromaDB 语义搜索失败: {e}')

    # 2. MySQL 精确匹配（作为补充）
    qs = Post.objects.filter(status='published')
    if keyword:
        from django.db.models import Q
        qs = qs.filter(Q(title__icontains=keyword) | Q(content__icontains=keyword))
    if category:
        qs = qs.filter(category_id=category)
    if tag:
        qs = qs.filter(tags__id=tag).distinct()

    mysql_ids = set(qs.values_list('id', flat=True)[:limit])

    # 合并结果（语义搜索优先，过滤低于 30% 的语义结果）
    seen = set()
    merged = []
    for r in results:
        pid = r.get('post_id')
        if pid and pid not in seen and r.get('score', 0) >= 0.3:
            seen.add(pid)
            merged.append(r)
    # 补充 MySQL 独有的（关键词能匹配但语义搜索没覆盖到的）
    for pid in mysql_ids:
        if pid not in seen:
            merged.append({'post_id': pid, 'score': 0, 'source': 'keyword'})
            seen.add(pid)

    return merged[:limit]


# =============================================================================
# 推荐系统（Redis 缓存 + ChromaDB 语义推荐）
# =============================================================================

import json

def get_recommendations(post_id=None, user_id=None, limit=10):
    """获取推荐帖子，优先从 Redis 缓存读取"""
    if post_id:
        cache_key = 'rec:%d' % post_id
        # 1. 先尝试从 Redis 缓存读取
        try:
            from .nosql_services import get_redis
            r = get_redis()
            cached = r.get(cache_key)
            if cached:
                logger.debug('推荐缓存命中: %s' % cache_key)
                return json.loads(cached)
        except Exception as e:
            logger.debug('Redis 读取失败: %s' % e)

        # 2. 缓存未命中，从 ChromaDB 计算
        results = []
        try:
            from .nosql_services import ChromaService
            results = ChromaService.get_recommendations(post_id, n_results=limit)
            logger.debug('ChromaDB 推荐: %d 条' % len(results))
        except Exception as e:
            logger.debug('ChromaDB 失败: %s' % e)

        # 3. 将结果写入 Redis 缓存（24小时过期）
        if results:
            try:
                r = get_redis()
                data = json.dumps(results)
                r.setex(cache_key, 86400, data)
                logger.debug('缓存已写入: %s, TTL=%d' % (cache_key, r.ttl(cache_key)))
            except Exception as e:
                logger.debug('Redis 写入失败: %s' % e)

        return results

    # 基于用户行为推荐（取热门帖子）
    if user_id:
        try:
            from .nosql_services import MongoService
            return MongoService.get_user_feed(user_id, limit=limit)
        except Exception:
            pass

    return []


def clear_recommendation_cache(post_id=None):
    """清除推荐缓存。post_id=None 时清除所有推荐缓存。"""
    try:
        from .nosql_services import get_redis
        r = get_redis()
        if post_id is not None:
            r.delete('rec:%d' % post_id)
        else:
            keys = r.keys('rec:*')
            if keys:
                r.delete(*keys)
    except Exception:
        pass


# =============================================================================
# 社交图谱
# =============================================================================

def get_mutual_follows(user_id, limit=10):
    """获取共同关注"""
    try:
        from .nosql_services import Neo4jService
        return Neo4jService.get_mutual_follows(user_id, limit=limit)
    except Exception:
        return []


def get_follow_network(user_id, depth=2):
    """获取关注网络图谱（用于前端可视化）"""
    try:
        from .nosql_services import Neo4jService
        return Neo4jService.get_follow_network(user_id, depth=depth)
    except Exception:
        return {'nodes': [], 'edges': []}


# =============================================================================
# 内部工具函数
# =============================================================================

def _log_action(user, action, post_id=None, extra=None):
    """记录用户行为到 MongoDB"""
    try:
        from .nosql_services import MongoService
        MongoService.log_user_action(user.username, action, post_id=post_id, extra=extra)
    except Exception:
        pass


def _notify(recipient, sender, msg_type, content, post=None, reply=None):
    """创建消息通知（MySQL + Redis 推送）"""
    if recipient == sender:
        return
    Message.objects.create(
        recipient=recipient,
        sender=sender,
        msg_type=msg_type,
        content=content,
        post=post,
        reply=reply,
    )
    # Redis 推送实时通知
    try:
        from .nosql_services import RedisService
        import json
        RedisService.push_notification(recipient.username, json.dumps({
            'type': msg_type,
            'content': content,
            'sender': sender.username,
        }, ensure_ascii=False))
    except Exception:
        pass
