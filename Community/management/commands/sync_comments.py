"""
从 MySQL 重建帖子评论到 MongoDB（一次性迁移）

用法：
    python manage.py sync_comments
    python manage.py sync_comments --post-id 1
"""
from django.core.management.base import BaseCommand
from Community.models import Post
from Community.nosql_services import MongoService


class Command(BaseCommand):
    help = '从 MySQL 重建帖子评论到 MongoDB'

    def add_arguments(self, parser):
        parser.add_argument('--post-id', type=int, help='指定帖子ID，不指定则同步所有')

    def handle(self, *args, **options):
        post_id = options.get('post_id')
        if post_id:
            posts = Post.objects.filter(id=post_id)
        else:
            posts = Post.objects.filter(status='published')

        for post in posts:
            MongoService.rebuild_comments(post.id)
            # 同时更新帖子快照
            MongoService.save_post_snapshot(post)
            self.stdout.write('  帖子 %d: %s' % (post.id, post.title[:30]))

        self.stdout.write(self.style.SUCCESS('同步完成'))
