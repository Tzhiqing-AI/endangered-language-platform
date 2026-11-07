# -*- coding: utf-8 -*-
"""社区回归测试。"""
from django.test import Client, TestCase

from Community.models import Post
from User.models import User


class AnonymousPostDetailTest(TestCase):
    """未登录用户必须能打开帖子详情。

    这条用例防止 post_detail 视图内再出现函数级 import 遮蔽模块级名字
    （曾导致匿名访问时 Q 被判定为局部变量并抛 UnboundLocalError → 500）。
    """

    def setUp(self):
        self.author = User.objects.create(
            username='regression_author',
            password='!unusable',
            name='回归测试作者',
        )
        self.post = Post.objects.create(
            author=self.author,
            title='匿名访问回归测试帖',
            content='这是一条用于回归测试的正文。',
            status='published',
        )

    def test_anonymous_can_open_post_detail(self):
        response = Client().get('/Community/post/{}/'.format(self.post.id))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '匿名访问回归测试帖')
