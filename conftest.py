"""pytest 共享 fixture。"""

import pytest
from django.test import Client
from rest_framework.test import APIClient

from User.models import User, UserProfile
from Community.models import Category, Post, Tag


def _make_user(username, **kwargs):
    user = User.objects.create(
        username=username,
        password='md5$test',
        name=kwargs.pop('name', username),
        **kwargs,
    )
    UserProfile.objects.create(user=user)
    return user


@pytest.fixture
def test_user(db):
    return _make_user('test_user')


@pytest.fixture
def test_admin(db):
    return _make_user('test_admin', admin=1)


@pytest.fixture
def test_expert(db):
    return _make_user('test_expert', expert=1)


@pytest.fixture
def test_category(db):
    return Category.objects.create(name='测试分类')


@pytest.fixture
def test_tag(db):
    return Tag.objects.create(name='测试标签')


@pytest.fixture
def test_post(db, test_user, test_category):
    post = Post.objects.create(
        author=test_user,
        category=test_category,
        title='测试帖子',
        content='测试内容',
        status='published',
        needs_moderation=False,
        moderation_queue='approved',
    )
    return post


def _session_client(user):
    client = APIClient()
    session = client.session
    session['username'] = user.username
    session['name'] = user.name or user.username
    session.save()
    return client


@pytest.fixture
def auth_client(test_user):
    return _session_client(test_user)


@pytest.fixture
def admin_client(test_admin):
    return _session_client(test_admin)


@pytest.fixture
def django_client():
    return Client()
