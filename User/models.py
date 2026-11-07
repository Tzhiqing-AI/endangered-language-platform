from django.db import models


# =============================================================================
# User 应用自己的 User 模型（共享模型，由 Dashboard 管理但在此定义）
# =============================================================================
class User(models.Model):
    """用户模型"""
    username = models.CharField('用户名', primary_key=True, max_length=20)
    password = models.CharField('密码', max_length=255, blank=True, null=True)
    name = models.CharField('昵称', max_length=20, blank=True, null=True)
    email = models.EmailField('邮箱', max_length=254, blank=True, null=True, unique=True)
    regtime = models.DateTimeField('注册时间', auto_now_add=True)
    last_login = models.DateTimeField('最后登录', null=True, blank=True)
    is_active = models.BooleanField('是否激活', default=True)
    admin = models.IntegerField('管理员标识', default=0)  # 0=普通用户, 1=管理员
    expert = models.IntegerField('专家标识', default=0)  # 0=普通用户, 1=专家
    expert_verified_at = models.DateTimeField('专家认证时间', null=True, blank=True)
    login_count = models.IntegerField('登录次数', default=0)
    reputation_score = models.IntegerField('声望分数', default=0, db_index=True)

    class Meta:
        db_table = 'user'
        verbose_name = '用户'
        verbose_name_plural = '用户'

    def set_password(self, raw_password):
        """使用 Django 密码哈希器设置密码，避免在代码中写哈希常量。"""
        from django.contrib.auth.hashers import make_password
        self.password = make_password(raw_password)

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    @property
    def is_expert(self):
        return self.expert == 1

    @property
    def is_admin(self):
        return self.admin == 1

    @property
    def role_display(self):
        if self.admin == 1:
            return '管理员'
        elif self.expert == 1:
            return '专家'
        return '普通用户'

    def update_last_login(self):
        """更新最后登录时间"""
        from django.utils import timezone
        self.last_login = timezone.now()
        self.login_count = models.F('login_count') + 1
        self.save(update_fields=['last_login', 'login_count'])

    def update_reputation(self, delta):
        """更新声望分数"""
        self.reputation_score = models.F('reputation_score') + delta
        self.save(update_fields=['reputation_score'])

    def __str__(self):
        return self.name or self.username


# =============================================================================
# ChatHistory 模型（聊天历史记录）
# =============================================================================
class ChatHistory(models.Model):
    """聊天历史记录"""
    chat_id = models.AutoField(primary_key=True)
    username = models.ForeignKey(User, on_delete=models.CASCADE, db_column='username', related_name='chat_histories')
    question = models.TextField('问题')
    answer = models.TextField('回答')
    timestamp = models.DateTimeField('时间', auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'chat_history'
        verbose_name = '聊天记录'
        verbose_name_plural = '聊天记录'
        ordering = ['-timestamp']

    def __str__(self):
        return f'{self.username}: {self.question[:20]}'


# =============================================================================
# 用户扩展信息模型
# =============================================================================
class UserProfile(models.Model):
    """用户扩展信息模型"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar = models.ImageField('头像', upload_to='avatars/', default='avatars/default.png', blank=True, null=True)
    bio = models.TextField('个人简介', max_length=500, blank=True, default='')
    phone = models.CharField('手机号', max_length=20, blank=True, default='')
    location = models.CharField('所在地', max_length=100, blank=True, default='')
    website = models.URLField('个人网站', max_length=200, blank=True, default='')
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        db_table = 'user_profile'
        verbose_name = '用户资料'
        verbose_name_plural = '用户资料'

    def __str__(self):
        return f'{self.user.username}的资料'


# =============================================================================
# 专家认证申请模型
# =============================================================================
class ExpertApplication(models.Model):
    """专家认证申请表"""
    STATUS_CHOICES = [
        ('pending', '待审核'),
        ('approved', '已通过'),
        ('rejected', '已拒绝'),
    ]

    FIELD_CHOICES = [
        ('linguistics', '语言学'),
        ('anthropology', '人类学'),
        ('cultural_studies', '文化研究'),
        ('language_education', '语言教育'),
        ('documentation', '语言记录'),
        ('translation', '翻译'),
        ('other', '其他'),
    ]

    applicant = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='expert_applications'
    )
    field = models.CharField('专家领域', max_length=50, choices=FIELD_CHOICES)
    credentials = models.TextField('资质证明', help_text='请详细描述您的专业背景、学历、工作经验等')
    languages = models.CharField(
        '相关语言', max_length=255,
        help_text='您专长的濒危语言，多个用逗号分隔'
    )
    reason = models.TextField('申请理由', help_text='为什么您想成为专家认证用户')
    supporting_docs = models.CharField(
        '证明材料链接', max_length=500, blank=True,
        help_text='可上传到网盘等，附上链接作为补充证明'
    )
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default='pending')
    reviewer = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='reviewed_applications', verbose_name='审核人'
    )
    review_comment = models.TextField('审核备注', blank=True)
    reviewed_at = models.DateTimeField('审核时间', null=True, blank=True)
    created_at = models.DateTimeField('申请时间', auto_now_add=True)

    class Meta:
        db_table = 'expert_application'
        verbose_name = '专家申请'
        verbose_name_plural = '专家申请'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['applicant', 'status']),
            models.Index(fields=['status', '-created_at']),
        ]

    def __str__(self):
        return f'{self.applicant.name}的专家申请({self.get_status_display()})'
