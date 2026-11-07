from django.db import models
from django.db.models import F
from User.models import User
from Dashboard.models import Language


class Category(models.Model):
    name = models.CharField('分类名称', max_length=50, unique=True)
    description = models.TextField('分类描述', blank=True)
    icon = models.CharField('图标类名', max_length=50, default='bi-chat')
    sort_order = models.IntegerField('排序', default=0)
    is_active = models.BooleanField('是否启用', default=True)

    class Meta:
        verbose_name = '论坛分类'
        verbose_name_plural = '论坛分类'
        ordering = ['sort_order', 'id']

    def __str__(self):
        return self.name


class Tag(models.Model):
    name = models.CharField('标签名称', max_length=30, unique=True)
    color = models.CharField('颜色', max_length=20, default='#0f766e')
    post_count = models.IntegerField('帖子数', default=0)

    class Meta:
        verbose_name = '标签'
        verbose_name_plural = '标签'

    def __str__(self):
        return self.name


class Post(models.Model):
    STATUS_CHOICES = [
        ('draft', '草稿'),
        ('published', '已发布'),
        ('hidden', '隐藏'),
        ('banned', '已封禁'),
    ]

    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    category = models.ForeignKey('Category', on_delete=models.SET_NULL, null=True, blank=True, related_name='posts')
    title = models.CharField('标题', max_length=200)
    content = models.TextField('内容')
    tags = models.ManyToManyField('Tag', blank=True, related_name='posts')
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default='draft')
    is_pinned = models.BooleanField('是否置顶', default=False)
    is_essence = models.BooleanField('是否精华', default=False)
    is_closed = models.BooleanField('是否关闭评论', default=False)
    view_count = models.IntegerField('浏览数', default=0)
    like_count = models.IntegerField('点赞数', default=0)
    reply_count = models.IntegerField('回复数', default=0)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    # 特色字段：关联濒危语言
    related_language = models.ForeignKey(
        Language, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='related_posts', verbose_name='关联语言'
    )
    language_level = models.CharField('濒危等级', max_length=20, blank=True)
    is_expert_verified = models.BooleanField('专家已验证', default=False)

    # 审核相关字段
    needs_moderation = models.BooleanField('需要审核', default=True)
    moderation_queue = models.CharField('审核队列', max_length=20, default='pending')
    verified_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='verified_posts', verbose_name='验证专家'
    )
    verified_at = models.DateTimeField('验证时间', null=True, blank=True)

    class Meta:
        verbose_name = '帖子'
        verbose_name_plural = '帖子'
        ordering = ['-is_pinned', '-created_at']
        indexes = [
            models.Index(fields=['status', '-created_at']),
            models.Index(fields=['category', '-created_at']),
        ]

    def __str__(self):
        return self.title

    def increment_view(self):
        Post.objects.filter(pk=self.pk).update(view_count=F('view_count') + 1)
        self.refresh_from_db(fields=['view_count'])

    def increment_like(self):
        Post.objects.filter(pk=self.pk).update(like_count=F('like_count') + 1)
        self.refresh_from_db(fields=['like_count'])

    def decrement_like(self):
        Post.objects.filter(pk=self.pk, like_count__gt=0).update(like_count=F('like_count') - 1)
        self.refresh_from_db(fields=['like_count'])

    def increment_reply(self):
        Post.objects.filter(pk=self.pk).update(reply_count=F('reply_count') + 1)
        self.refresh_from_db(fields=['reply_count'])

    def decrement_reply(self):
        Post.objects.filter(pk=self.pk, reply_count__gt=0).update(reply_count=F('reply_count') - 1)
        self.refresh_from_db(fields=['reply_count'])


class Reply(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='replies')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='replies')
    parent = models.ForeignKey(
        'self', on_delete=models.CASCADE, null=True, blank=True,
        related_name='children', verbose_name='父回复'
    )
    content = models.TextField('内容')
    like_count = models.IntegerField('点赞数', default=0)
    is_deleted = models.BooleanField('是否删除', default=False)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        verbose_name = '回复'
        verbose_name_plural = '回复'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['post', 'parent', 'is_deleted']),
            models.Index(fields=['author', '-created_at']),
        ]

    def __str__(self):
        return f'{self.author.name}回复了{self.post.title}'


class Like(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='likes')
    post = models.ForeignKey(Post, on_delete=models.CASCADE, null=True, blank=True, related_name='like_users')
    reply = models.ForeignKey('Reply', on_delete=models.CASCADE, null=True, blank=True, related_name='like_users')
    created_at = models.DateTimeField('创建时间', auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = '点赞'
        verbose_name_plural = '点赞'
        unique_together = ['user', 'post', 'reply']

    def __str__(self):
        return f'{self.user.name}点赞'


class Collect(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='collects')
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='collect_users')
    created_at = models.DateTimeField('创建时间', auto_now_add=True)

    class Meta:
        verbose_name = '收藏'
        verbose_name_plural = '收藏'
        unique_together = ['user', 'post']

    def __str__(self):
        return f'{self.user.name}收藏了{self.post.title}'


class Follow(models.Model):
    follower = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='following', verbose_name='关注者'
    )
    following = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='followers', verbose_name='被关注者'
    )
    created_at = models.DateTimeField('创建时间', auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = '关注'
        verbose_name_plural = '关注'
        unique_together = ['follower', 'following']

    def __str__(self):
        return f'{self.follower.name}关注了{self.following.name}'


class Message(models.Model):
    TYPE_CHOICES = [
        ('reply', '回复'),
        ('like', '点赞'),
        ('follow', '关注'),
        ('system', '系统通知'),
        ('mention', '@提及'),
    ]

    recipient = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='messages', verbose_name='接收者'
    )
    sender = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='sent_messages',
        null=True, blank=True, verbose_name='发送者'
    )
    msg_type = models.CharField('类型', max_length=20, choices=TYPE_CHOICES)
    content = models.TextField('内容')
    is_read = models.BooleanField('已读', default=False)
    post = models.ForeignKey(Post, on_delete=models.CASCADE, null=True, blank=True, verbose_name='相关帖子')
    reply = models.ForeignKey(Reply, on_delete=models.CASCADE, null=True, blank=True, verbose_name='相关回复')
    created_at = models.DateTimeField('创建时间', auto_now_add=True)

    class Meta:
        verbose_name = '消息'
        verbose_name_plural = '消息'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recipient', 'is_read', '-created_at']),
        ]

    def __str__(self):
        return f'{self.recipient.name}的{self.get_msg_type_display()}消息'


class Report(models.Model):
    REASON_CHOICES = [
        ('spam', '垃圾广告'),
        ('violence', '暴力色情'),
        ('harassment', '骚扰辱骂'),
        ('fake', '虚假信息'),
        ('copyright', '侵权'),
        ('other', '其他'),
    ]

    reporter = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports')
    post = models.ForeignKey(Post, on_delete=models.CASCADE, null=True, blank=True, related_name='reports')
    reply = models.ForeignKey(Reply, on_delete=models.CASCADE, null=True, blank=True, related_name='reports')
    reason = models.CharField('举报原因', max_length=20, choices=REASON_CHOICES)
    description = models.TextField('详细描述', blank=True)
    status = models.CharField('处理状态', max_length=20, default='pending', choices=[
        ('pending', '待处理'),
        ('processed', '已处理'),
        ('dismissed', '已驳回'),
    ])
    created_at = models.DateTimeField('创建时间', auto_now_add=True)

    class Meta:
        verbose_name = '举报'
        verbose_name_plural = '举报'

    def __str__(self):
        return f'{self.reporter.name}举报了{self.post or self.reply}'


class ContentModeration(models.Model):
    """内容审核记录表"""
    ACTION_CHOICES = [
        ('approve', '通过'),
        ('reject', '拒绝'),
        ('hide', '隐藏'),
        ('restore', '恢复'),
    ]

    content_type = models.CharField('内容类型', max_length=20)  # 'post' 或 'reply'
    content_id = models.IntegerField('内容ID')
    moderator = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='moderation_actions'
    )
    action = models.CharField('操作', max_length=20, choices=ACTION_CHOICES)
    reason = models.TextField('操作原因', blank=True)
    created_at = models.DateTimeField('操作时间', auto_now_add=True)

    class Meta:
        db_table = 'content_moderation'
        verbose_name = '内容审核'
        verbose_name_plural = '内容审核'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['content_type', 'content_id']),
            models.Index(fields=['moderator', '-created_at']),
        ]

    def __str__(self):
        return f'{self.moderator.name}{self.get_action_display()}了{self.content_type}:{self.content_id}'


class ExpertEndorsement(models.Model):
    """专家内容背书表"""
    ENDORSEMENT_TYPE_CHOICES = [
        ('verified', '权威认证'),
        ('corrected', '已纠错'),
        ('annotated', '补充注释'),
    ]

    post = models.ForeignKey(
        Post, on_delete=models.CASCADE, null=True, blank=True,
        related_name='expert_endorsements', verbose_name='相关帖子'
    )
    reply = models.ForeignKey(
        Reply, on_delete=models.CASCADE, null=True, blank=True,
        related_name='expert_endorsements', verbose_name='相关回复'
    )
    expert = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='endorsements_given',
        limit_choices_to=models.Q(expert=1)
    )
    endorsement_type = models.CharField('背书类型', max_length=20, choices=ENDORSEMENT_TYPE_CHOICES)
    comment = models.TextField('专家意见', blank=True)
    created_at = models.DateTimeField('背书时间', auto_now_add=True)

    class Meta:
        db_table = 'expert_endorsement'
        verbose_name = '专家背书'
        verbose_name_plural = '专家背书'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['expert', '-created_at']),
        ]

    def __str__(self):
        target = self.post or self.reply
        return f'{self.expert.name}对"{target}"进行了{self.get_endorsement_type_display()}'
