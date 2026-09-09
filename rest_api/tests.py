"""REST API 测试"""
import pytest
from Community.models import Reply


@pytest.mark.django_db
class TestLanguageAPI:
    """濒危语言 API 测试"""

    def test_list_languages_anonymous(self, client):
        """匿名用户可以查看语言列表"""
        resp = client.get('/api/languages')
        assert resp.status_code == 200

    def test_list_languages_with_filter(self, client):
        """支持 level 筛选"""
        resp = client.get('/api/languages?level=endangered')
        assert resp.status_code == 200


@pytest.mark.django_db
class TestCategoryTagAPI:
    """分类和标签 API 测试"""

    def test_list_categories(self, client, test_category):
        resp = client.get('/api/categories')
        assert resp.status_code == 200

    def test_list_tags(self, client, test_tag):
        resp = client.get('/api/tags')
        assert resp.status_code == 200


@pytest.mark.django_db
class TestPostAPI:
    """帖子 API 测试"""

    def test_list_posts_anonymous(self, client, test_post):
        """匿名用户可以查看帖子列表"""
        resp = client.get('/api/posts')
        assert resp.status_code == 200

    def test_create_post_requires_auth(self, client):
        """未登录不能创建帖子"""
        resp = client.post('/api/posts', {'title': 't', 'content': 'c'})
        assert resp.status_code == 403

    def test_create_post_authenticated(self, auth_client, test_user):
        """已登录可以创建帖子"""
        resp = auth_client.post('/api/posts', {
            'title': '新帖子',
            'content': '内容',
        })
        assert resp.status_code == 201

    def test_post_detail(self, client, test_post):
        """查看帖子详情"""
        resp = client.get(f'/api/posts/{test_post.id}')
        assert resp.status_code == 200

    def test_post_like_toggle(self, auth_client, test_post, test_user):
        """点赞/取消点赞切换"""
        resp = auth_client.post(f'/api/posts/{test_post.id}/like')
        assert resp.status_code == 200
        assert resp.data['liked'] is True

        resp = auth_client.post(f'/api/posts/{test_post.id}/like')
        assert resp.status_code == 200
        assert resp.data['liked'] is False

    def test_post_collect_toggle(self, auth_client, test_post, test_user):
        """收藏/取消收藏切换"""
        resp = auth_client.post(f'/api/posts/{test_post.id}/collect')
        assert resp.status_code == 200
        assert resp.data['collected'] is True

        resp = auth_client.post(f'/api/posts/{test_post.id}/collect')
        assert resp.status_code == 200
        assert resp.data['collected'] is False

    def test_post_replies(self, client, test_post):
        """获取帖子回复"""
        resp = client.get(f'/api/posts/{test_post.id}/replies')
        assert resp.status_code == 200


@pytest.mark.django_db
class TestReplyAPI:
    """回复 API 测试"""

    def test_create_reply_requires_auth(self, client, test_post):
        """未登录不能创建回复"""
        resp = client.post('/api/replies', {
            'post': test_post.id,
            'content': '回复内容',
        })
        assert resp.status_code == 403

    def test_reply_like_toggle(self, auth_client, test_post, test_user):
        """回复点赞切换"""
        reply = Reply.objects.create(
            post=test_post,
            author=test_user,
            content='测试回复',
        )
        resp = auth_client.post(f'/api/replies/{reply.id}/like')
        assert resp.status_code == 200
        assert resp.data['liked'] is True


@pytest.mark.django_db
class TestFollowAPI:
    """关注 API 测试"""

    def test_follow_toggle(self, auth_client, test_user, test_admin):
        """关注/取消关注切换"""
        resp = auth_client.post('/api/follow/toggle', {
            'username': test_admin.username,
        }, format='json')
        assert resp.status_code == 200
        assert resp.data['following'] is True

        resp = auth_client.post('/api/follow/toggle', {
            'username': test_admin.username,
        }, format='json')
        assert resp.status_code == 200
        assert resp.data['following'] is False

    def test_cannot_follow_self(self, auth_client, test_user):
        """不能关注自己"""
        resp = auth_client.post('/api/follow/toggle', {
            'username': test_user.username,
        }, format='json')
        assert resp.status_code == 400


@pytest.mark.django_db
class TestMessageAPI:
    """消息 API 测试"""

    def test_unread_count_anonymous(self, client):
        """匿名用户无法获取未读消息数（DRF 默认 IsAuthenticated）"""
        resp = client.get('/api/messages/unread_count')
        assert resp.status_code == 403

    def test_unread_count_authenticated(self, auth_client, test_user):
        """已登录用户获取未读消息数"""
        resp = auth_client.get('/api/messages/unread_count')
        assert resp.status_code == 200


@pytest.mark.django_db
class TestPermissionClasses:
    """权限类测试"""

    def test_expert_verify_requires_expert(self, auth_client, test_user, test_post):
        """普通用户不能进行专家验证"""
        resp = auth_client.post('/api/expert/verify-post/', {
            'post_id': test_post.id,
        }, format='json')
        assert resp.status_code == 403

    def test_expert_verify_by_expert(self, test_expert, test_post):
        """专家可以进行验证"""
        from rest_framework.test import APIClient
        expert_client = APIClient()
        expert_client.force_authenticate(user=test_expert)
        session = expert_client.session
        session['username'] = test_expert.username
        session.save()

        resp = expert_client.post('/api/expert/verify-post/', {
            'post_id': test_post.id,
        }, format='json')
        assert resp.status_code == 200

    def test_moderation_queue_requires_auth(self, client):
        """未登录不能访问审核队列"""
        resp = client.get('/api/moderation/queue/')
        assert resp.status_code == 403

    def test_moderation_queue_requires_moderator(self, auth_client, test_user):
        """普通用户不能访问审核队列"""
        resp = auth_client.get('/api/moderation/queue/')
        assert resp.status_code == 403

    def test_moderation_queue_admin(self, admin_client, test_admin):
        """管理员可以访问审核队列"""
        resp = admin_client.get('/api/moderation/queue/')
        assert resp.status_code == 200
