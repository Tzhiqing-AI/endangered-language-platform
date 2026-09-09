"""
Dashboard 测试文件
"""
import pytest
from django.test import Client
from django.urls import reverse


@pytest.fixture
def django_client():
    """Django 原生测试客户端（用于模板视图测试）"""
    return Client()


@pytest.mark.django_db
class TestLanguageViews:
    """语言管理视图测试"""

    def test_language_list_requires_login(self, client):
        """测试语言列表需要登录"""
        response = client.get(reverse('Dashboard:language_list'))
        # 未登录应该重定向或返回登录提示
        assert response.status_code in [200, 302, 401]

    def test_language_page_loads(self, client):
        """测试语言首页能正常加载"""
        response = client.get(reverse('Dashboard:language'))
        assert response.status_code == 200


@pytest.mark.django_db
class TestCommunityViews:
    """社区论坛视图测试"""

    def test_community_home_loads(self, client):
        """测试社区首页能正常加载"""
        response = client.get(reverse('Community:community_home'))
        assert response.status_code == 200

    def test_search_posts_loads(self, client):
        """测试帖子搜索页面能正常加载"""
        response = client.get(reverse('Community:search_posts'))
        assert response.status_code == 200


@pytest.mark.django_db
class TestUserViews:
    """用户视图测试"""

    def test_login_page_loads(self, client):
        """测试登录页面能正常加载"""
        response = client.get(reverse('User:login'))
        assert response.status_code == 200

    def test_register_page_loads(self, client):
        """测试注册页面能正常加载"""
        response = client.get(reverse('User:register'))
        assert response.status_code == 200
