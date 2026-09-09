"""
同步 MySQL 数据到非关系型数据库

用法：
    python manage.py sync_nosql                    # 同步所有
    python manage.py sync_nosql --target redis     # 只同步 Redis
    python manage.py sync_nosql --target mongo     # 只同步 MongoDB
    python manage.py sync_nosql --target neo4j     # 只同步 Neo4j
    python manage.py sync_nosql --target chroma    # 只同步 ChromaDB
"""
from django.core.management.base import BaseCommand
from django.db.models import F
from Community.models import Post, Follow, Category, Tag
from User.models import User


class Command(BaseCommand):
    help = '同步 MySQL 数据到非关系型数据库（Redis/MongoDB/Neo4j/ChromaDB）'

    def add_arguments(self, parser):
        parser.add_argument(
            '--target',
            choices=['redis', 'mongo', 'neo4j', 'chroma', 'all'],
            default='all',
            help='指定同步目标数据库',
        )

    def handle(self, *args, **options):
        target = options['target']
        self.stdout.write(f'开始同步数据到 {target}...')

        if target in ('redis', 'all'):
            self.sync_redis()

        if target in ('mongo', 'all'):
            self.sync_mongo()

        if target in ('neo4j', 'all'):
            self.sync_neo4j()

        if target in ('chroma', 'all'):
            self.sync_chroma()

        self.stdout.write(self.style.SUCCESS('同步完成！'))

    def sync_redis(self):
        """同步热门帖子排行榜到 Redis"""
        self.stdout.write('  同步 Redis 排行榜...')
        try:
            from Community.nosql_services import RedisService
            posts = Post.objects.filter(status='published').annotate(
                hot_score=F('view_count') * 0.3 + F('like_count') * 0.5 + F('reply_count') * 0.2
            ).order_by('-hot_score')[:200]

            for post in posts:
                score = float(post.hot_score)
                RedisService.update_hot_post(post.id, score, post.category_id)
                # 全站排行榜
                RedisService.update_hot_post(post.id, score, None)

            self.stdout.write(f'    同步了 {posts.count()} 个帖子到排行榜')
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'    Redis 同步失败: {e}'))

    def sync_mongo(self):
        """同步所有已发布帖子到 MongoDB"""
        self.stdout.write('  同步 MongoDB 帖子快照...')
        try:
            from Community.nosql_services import MongoService
            posts = Post.objects.filter(status='published').select_related('author', 'category').prefetch_related('tags')

            count = 0
            for post in posts:
                MongoService.save_post_snapshot(post)
                count += 1

            self.stdout.write(f'    同步了 {count} 个帖子快照')
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'    MongoDB 同步失败: {e}'))

    def sync_neo4j(self):
        """同步用户和关注关系到 Neo4j"""
        self.stdout.write('  同步 Neo4j 社交关系图谱...')
        try:
            from Community.nosql_services import Neo4jService

            # 同步所有活跃用户
            users = User.objects.filter(is_active=True)
            for user in users:
                Neo4jService.sync_user(user)
            self.stdout.write(f'    同步了 {users.count()} 个用户节点')

            # 同步所有关注关系
            follows = Follow.objects.select_related('follower', 'following')
            for follow in follows:
                Neo4jService.sync_follow(follow.follower, follow.following, created=True)
            self.stdout.write(f'    同步了 {follows.count()} 条关注关系')
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'    Neo4j 同步失败: {e}'))

    def sync_chroma(self):
        """同步帖子到 ChromaDB 向量库"""
        self.stdout.write('  同步 ChromaDB 向量索引...')
        try:
            from Community.nosql_services import ChromaService
            posts = Post.objects.filter(status='published')

            count = 0
            for post in posts:
                ChromaService.add_post(post)
                count += 1

            self.stdout.write(f'    同步了 {count} 个帖子到向量库')
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'    ChromaDB 同步失败: {e}'))
