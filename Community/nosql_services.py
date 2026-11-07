"""
非关系型数据库服务层

统一管理 Redis / MongoDB / Neo4j / ChromaDB 的连接和操作。
按需懒加载，避免启动时连接所有数据库。
"""
import logging
from functools import lru_cache

from django.conf import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Redis 服务
# =============================================================================

def get_redis():
    """获取 Redis 连接（通过 django-redis）"""
    from django_redis import get_redis_connection
    return get_redis_connection("default")


class RedisService:
    """Redis 服务：缓存、排行榜、消息队列、分布式锁"""

    @staticmethod
    def get_hot_posts(category_id=None, limit=20):
        """获取热门帖子 ID 列表（ZSET 排行榜）"""
        r = get_redis()
        key = f'hot_posts:{category_id}' if category_id else 'hot_posts:all'
        return r.zrevrange(key, 0, limit - 1, withscores=True)

    @staticmethod
    def update_hot_post(post_id, score, category_id=None):
        """更新帖子热度分数到排行榜"""
        r = get_redis()
        key = f'hot_posts:{category_id}' if category_id else 'hot_posts:all'
        r.zadd(key, {str(post_id): score})

    @staticmethod
    def distributed_lock(lock_name, timeout=10):
        """分布式锁（上下文管理器）"""
        import time
        r = get_redis()
        lock_key = f'lock:{lock_name}'
        acquired = r.set(lock_key, '1', nx=True, ex=timeout)
        return acquired

    @staticmethod
    def release_lock(lock_name):
        """释放分布式锁"""
        r = get_redis()
        r.delete(f'lock:{lock_name}')

    @staticmethod
    def rate_limit(key, limit=100, window=1):
        """滑动窗口限流：每 window 秒内最多 limit 次请求"""
        import time
        r = get_redis()
        now = time.time()
        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window)
        results = pipe.execute()
        count = results[2]
        return count <= limit

    @staticmethod
    def push_notification(user_id, message):
        """推送用户通知到 Redis List"""
        r = get_redis()
        r.lpush(f'notifications:{user_id}', message)
        r.ltrim(f'notifications:{user_id}', 0, 99)  # 最多保留 100 条


# =============================================================================
# MongoDB 服务
# =============================================================================

_mongo_client = None
_mongo_db = None


def get_mongo_db():
    """获取 MongoDB 数据库连接（懒加载单例）"""
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        from pymongo import MongoClient
        _mongo_client = MongoClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=3000,
            tz_aware=True,
        )
        _mongo_db = _mongo_client[settings.MONGODB_NAME]
        # 创建索引
        _mongo_db.posts.create_index('post_id', unique=True)
        _mongo_db.posts.create_index('author_id')
        _mongo_db.posts.create_index('created_at')
        _mongo_db.user_logs.create_index([('user_id', 1), ('created_at', -1)])
    return _mongo_db


class MongoService:
    """MongoDB 服务：帖子快照、用户行为日志、评论嵌套"""

    @staticmethod
    def save_post_snapshot(post):
        """帖子发布/更新时，同步快照到 MongoDB"""
        db = get_mongo_db()
        from Community.models import Category
        tags = list(post.tags.values_list('name', flat=True))
        category_name = post.category.name if post.category else ''

        doc = {
            'post_id': post.id,
            'title': post.title,
            'content': post.content,
            'author_id': post.author.username,
            'author_name': post.author.name or post.author.username,
            'category': {'id': post.category_id, 'name': category_name} if post.category_id else None,
            'tags': tags,
            'stats': {
                'views': post.view_count,
                'likes': post.like_count,
                'comments': post.reply_count,
            },
            'is_pinned': post.is_pinned,
            'is_essence': post.is_essence,
            'created_at': post.created_at,
            'updated_at': post.updated_at,
        }
        db.posts.update_one({'post_id': post.id}, {'$set': doc}, upsert=True)
        return doc

    @staticmethod
    def get_post_snapshot(post_id):
        """从 MongoDB 获取帖子快照"""
        db = get_mongo_db()
        return db.posts.find_one({'post_id': int(post_id)})

    @staticmethod
    def get_user_feed(user_id, limit=20, skip=0):
        """获取用户 Feed 流（从 MongoDB 聚合查询）"""
        db = get_mongo_db()
        pipeline = [
            {'$match': {'is_pinned': False}},
            {'$sort': {'created_at': -1}},
            {'$skip': skip},
            {'$limit': limit},
            {'$project': {
                'post_id': 1, 'title': 1, 'author_name': 1,
                'tags': 1, 'stats': 1, 'created_at': 1,
            }},
        ]
        return list(db.posts.aggregate(pipeline))

    @staticmethod
    def log_user_action(user_id, action, post_id=None, extra=None):
        """记录用户行为日志（浏览/点赞/收藏/搜索）"""
        db = get_mongo_db()
        from django.utils import timezone
        doc = {
            'user_id': user_id,
            'action': action,
            'post_id': post_id,
            'extra': extra or {},
            'created_at': timezone.now(),
        }
        db.user_logs.insert_one(doc)

    @staticmethod
    def get_similar_posts(post_id, limit=10):
        """获取相似帖子（基于标签和分类）"""
        db = get_mongo_db()
        post = db.posts.find_one({'post_id': int(post_id)})
        if not post:
            return []
        pipeline = [
            {'$match': {
                'post_id': {'$ne': int(post_id)},
                '$or': [
                    {'tags': {'$in': post.get('tags', [])}},
                    {'category': post.get('category', {})},
                ]
            }},
            {'$addFields': {
                'relevance': {
                    '$size': {'$setIntersection': ['$tags', post.get('tags', [])]}
                }
            }},
            {'$sort': {'relevance': -1, 'created_at': -1}},
            {'$limit': limit},
            {'$project': {'post_id': 1, 'title': 1, 'author_name': 1, 'stats': 1}},
        ]
        return list(db.posts.aggregate(pipeline))

    @staticmethod
    def add_comment(post_id, reply):
        """把评论作为子文档追加到 MongoDB 帖子的 comments 数组"""
        db = get_mongo_db()
        comment_doc = {
            'reply_id': reply.id,
            'user_id': reply.author.username,
            'user_name': reply.author.name or reply.author.username,
            'content': reply.content,
            'created_at': reply.created_at,
            'like_count': reply.like_count,
            'children': [],
        }
        if reply.parent_id:
            # 子评论：追加到父评论的 children 数组
            db.posts.update_one(
                {'post_id': int(post_id), 'comments.reply_id': reply.parent_id},
                {'$push': {'comments.$.children': comment_doc}}
            )
        else:
            # 一级评论：追加到帖子的 comments 数组
            db.posts.update_one(
                {'post_id': int(post_id)},
                {'$push': {'comments': comment_doc}}
            )

    @staticmethod
    def remove_comment(post_id, reply_id):
        """从 MongoDB 删除评论"""
        db = get_mongo_db()
        # 先尝试从一级评论删除
        db.posts.update_one(
            {'post_id': int(post_id)},
            {'$pull': {'comments': {'reply_id': reply_id}}}
        )
        # 再尝试从子评论删除
        db.posts.update_one(
            {'post_id': int(post_id), 'comments.children.reply_id': reply_id},
            {'$pull': {'comments.$.children': {'reply_id': reply_id}}}
        )

    @staticmethod
    def rebuild_comments(post_id):
        """从 MySQL 重建帖子的完整嵌套评论树到 MongoDB。"""
        from Community.models import Reply
        db = get_mongo_db()
        replies = list(Reply.objects.filter(
            post_id=post_id, is_deleted=False
        ).select_related('author').order_by('created_at'))

        docs = {
            r.id: {
                'reply_id': r.id,
                'user_id': r.author.username,
                'user_name': r.author.name or r.author.username,
                'content': r.content,
                'created_at': r.created_at,
                'like_count': r.like_count,
                'children': [],
            }
            for r in replies
        }
        top_level = []
        for r in replies:
            doc = docs[r.id]
            if r.parent_id and r.parent_id in docs:
                docs[r.parent_id]['children'].append(doc)
            else:
                top_level.append(doc)

        db.posts.update_one(
            {'post_id': int(post_id)},
            {'$set': {'comments': top_level}}
        )


# =============================================================================
# Neo4j 服务
# =============================================================================

_neo4j_driver = None


def get_neo4j_driver():
    """获取 Neo4j 驱动（懒加载单例）"""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=5,
        )
    return _neo4j_driver


class Neo4jService:
    """Neo4j 服务：社交关系图谱、共同关注、二度人脉推荐"""

    @staticmethod
    def sync_user(user):
        """同步用户节点到 Neo4j"""
        driver = get_neo4j_driver()
        with driver.session() as session:
            session.run(
                "MERGE (u:User {id: $id}) SET u.name = $name",
                id=user.username, name=user.name or user.username,
            )

    @staticmethod
    def sync_follow(follower, following, created=True):
        """同步关注关系到 Neo4j"""
        driver = get_neo4j_driver()
        with driver.session() as session:
            if created:
                session.run(
                    """MATCH (a:User {id: $fid}), (b:User {id: $tid})
                       MERGE (a)-[:FOLLOWS]->(b)""",
                    fid=follower.username, tid=following.username,
                )
            else:
                session.run(
                    """MATCH (a:User {id: $fid})-[r:FOLLOWS]->(b:User {id: $tid})
                       DELETE r""",
                    fid=follower.username, tid=following.username,
                )

    @staticmethod
    def get_mutual_follows(user_id, limit=10):
        """获取共同关注：我关注的人也关注了谁"""
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(
                """MATCH (me:User {id: $uid})-[:FOLLOWS]->(f:User)-[:FOLLOWS]->(fo:User)
                   WHERE fo.id <> $uid AND NOT (me)-[:FOLLOWS]->(fo)
                   RETURN fo.id AS user_id, fo.name AS name, COUNT(f) AS common
                   ORDER BY common DESC LIMIT $limit""",
                uid=user_id, limit=limit,
            )
            return [dict(r) for r in result]

    @staticmethod
    def get_two_degree_friends(user_id, limit=10):
        """获取二度人脉推荐：朋友的朋友"""
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(
                """MATCH (me:User {id: $uid})-[:FOLLOWS]->(:User)-[:FOLLOWS]->(fo:User)
                   WHERE fo.id <> $uid AND NOT (me)-[:FOLLOWS]->(fo)
                   RETURN fo.id AS user_id, fo.name AS name, COUNT(*) AS strength
                   ORDER BY strength DESC LIMIT $limit""",
                uid=user_id, limit=limit,
            )
            return [dict(r) for r in result]

    @staticmethod
    def get_follow_network(user_id, depth=2, limit=50):
        """获取用户关注网络（用于前端图谱可视化）"""
        driver = get_neo4j_driver()
        with driver.session() as session:
            # 手动提取中心用户字段，避免 dict(Record) 失败
            # 兼容 id=username 和 id=name 两种情况
            center_rec = session.run(
                """MATCH (u:User)
                   WHERE u.id = $uid OR u.name = $uid
                   RETURN u.id AS id, u.name AS name LIMIT 1""",
                uid=user_id,
            ).single()
            if center_rec:
                center_node = {'id': center_rec['id'], 'name': center_rec['name']}
            else:
                center_node = {'id': user_id, 'name': user_id}

            # 获取中心用户的真实 id（可能与 URL 传入的不同）
            actual_id = center_node['id']

            # 按跳数逐层查询，避免边塌缩
            all_nodes = {center_node['id']: center_node}
            all_edges = []
            edge_keys = set()
            current_ids = [actual_id]

            for hop in range(1, depth + 1):
                if not current_ids:
                    break
                # 查询当前层的下一层节点
                result = session.run(
                    """MATCH (a:User)-[:FOLLOWS]->(b:User)
                       WHERE a.id IN $ids
                       RETURN DISTINCT b.id AS id, b.name AS name, a.id AS from_id
                       LIMIT $limit""",
                    ids=current_ids, limit=limit,
                )
                next_ids = []
                for r in result:
                    nid = r['id']
                    from_id = r['from_id']
                    # 添加节点（去重）
                    if nid not in all_nodes:
                        all_nodes[nid] = {'id': nid, 'name': r['name']}
                        next_ids.append(nid)
                    # 添加边（去重 + 过滤自环）
                    if from_id != nid:
                        edge_key = '%s->%s' % (from_id, nid)
                        if edge_key not in edge_keys:
                            edge_keys.add(edge_key)
                            all_edges.append({'source': from_id, 'target': nid})
                current_ids = next_ids

            nodes = list(all_nodes.values())
            return {'nodes': nodes, 'edges': all_edges}


# =============================================================================
# ChromaDB 服务
# =============================================================================

_chroma_client = None
_chroma_embedding_fn = None


def get_chroma_collection():
    """获取 ChromaDB 集合（每次显式传入中文嵌入函数）"""
    global _chroma_client, _chroma_embedding_fn
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=settings.CHROMADB_HOST,
            port=settings.CHROMADB_PORT,
        )

    if _chroma_embedding_fn is None:
        _chroma_embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="shibing624/text2vec-base-chinese"
        )

    return _chroma_client.get_or_create_collection(
        name="posts_zh",
        embedding_function=_chroma_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


class ChromaService:
    """ChromaDB 服务：语义搜索、内容推荐"""

    @staticmethod
    def add_post(post):
        """帖子发布时，将标题+摘要入库"""
        collection = get_chroma_collection()
        doc = f"{post.title} {post.content[:200]}"
        collection.upsert(
            ids=[f"post_{post.id}"],
            documents=[doc],
            metadatas=[{
                "post_id": post.id,
                "author_id": post.author.username,
                "category_id": post.category_id or 0,
            }],
        )

    @staticmethod
    def search_posts(query, n_results=10):
        """语义搜索帖子"""
        collection = get_chroma_collection()
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return results

    @staticmethod
    def get_recommendations(post_id, n_results=10):
        """基于帖子内容推荐相似帖子"""
        collection = get_chroma_collection()
        # 先获取目标帖子的文档
        try:
            target = collection.get(ids=[f"post_{post_id}"], include=["documents"])
            if not target["documents"]:
                return []
            query_text = target["documents"][0]
        except Exception:
            return []

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results + 1,  # 多取一个，排除自身
            include=["metadatas", "distances"],
        )
        # 排除自身
        posts = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            if meta.get("post_id") != post_id:
                posts.append({"post_id": meta["post_id"], "score": 1 - dist})
        return posts[:n_results]

    @staticmethod
    def remove_post(post_id):
        """帖子删除时，从向量库移除"""
        collection = get_chroma_collection()
        collection.delete(ids=[f"post_{post_id}"])
