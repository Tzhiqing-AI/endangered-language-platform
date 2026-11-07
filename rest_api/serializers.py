"""
REST API 序列化器

将 Django 模型转换为 JSON API 格式
"""
from rest_framework import serializers
from User.models import User, UserProfile, ChatHistory, ExpertApplication
from Dashboard.models import Language
from Community.models import Post, Reply, Category, Tag, Message, Follow, ExpertEndorsement


class UserSerializer(serializers.ModelSerializer):
    """用户序列化器"""
    class Meta:
        model = User
        fields = ['username', 'name', 'regtime']
        read_only_fields = ['username', 'regtime']


class UserProfileSerializer(serializers.ModelSerializer):
    """用户资料序列化器"""
    username = serializers.CharField(source='user.username', read_only=True)
    name = serializers.CharField(source='user.name', read_only=True)

    class Meta:
        model = UserProfile
        fields = ['username', 'name', 'avatar', 'bio', 'phone', 'location', 'website', 'updated_at']
        read_only_fields = ['updated_at']


class LanguageSerializer(serializers.ModelSerializer):
    """濒危语言序列化器"""
    level_name = serializers.CharField(source='get_level_of_endangerment_display', read_only=True)

    class Meta:
        model = Language
        fields = [
            'language_id', 'language_name', 'level_of_endangerment', 'level_name',
            'number_of_speakers', 'speaker_trends', 'intergenerational_transmission',
            'language_family', 'country', 'additional_notes', 'media'
        ]
        read_only_fields = ['language_id']


class CategorySerializer(serializers.ModelSerializer):
    """分类序列化器"""
    post_count = serializers.SerializerMethodField()

    class Meta:
        model = Category
        fields = ['id', 'name', 'description', 'icon', 'sort_order', 'is_active', 'post_count']

    def get_post_count(self, obj):
        # 使用 annotation 预计算的 post_count，如果不存在则回退到 count()
        if hasattr(obj, 'post_count'):
            return obj.post_count
        return obj.posts.count()


class TagSerializer(serializers.ModelSerializer):
    """标签序列化器"""
    post_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Tag
        fields = ['id', 'name', 'color', 'post_count']


class PostListSerializer(serializers.ModelSerializer):
    """帖子列表序列化器（简短信息）"""
    author_name = serializers.CharField(source='author.name', read_only=True)
    category_name = serializers.CharField(source='category.name', read_only=True)
    tags = TagSerializer(many=True, read_only=True)

    class Meta:
        model = Post
        fields = [
            'id', 'title', 'author_name', 'category_name', 'tags',
            'view_count', 'like_count', 'reply_count', 'is_pinned', 'is_essence',
            'created_at', 'updated_at'
        ]


class PostDetailSerializer(serializers.ModelSerializer):
    """帖子详情序列化器"""
    author = UserSerializer(read_only=True)
    category = CategorySerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    related_language = LanguageSerializer(read_only=True)

    class Meta:
        model = Post
        fields = [
            'id', 'title', 'content', 'author', 'category', 'tags',
            'related_language', 'language_level', 'is_expert_verified',
            'view_count', 'like_count', 'reply_count', 'is_pinned', 'is_essence', 'is_closed',
            'created_at', 'updated_at'
        ]


class ReplySerializer(serializers.ModelSerializer):
    """回复序列化器"""
    author_name = serializers.CharField(source='author.name', read_only=True)
    children = serializers.SerializerMethodField()

    class Meta:
        model = Reply
        fields = ['id', 'post', 'content', 'author_name', 'parent', 'like_count', 'children', 'created_at']
        read_only_fields = ['id', 'created_at', 'author_name', 'like_count']

    def get_children(self, obj):
        max_depth = self.context.get('max_depth', 3)
        current_depth = self.context.get('depth', 0)
        if current_depth >= max_depth:
            return []
        # 使用 prefetched 的 children 如果可用，否则查询
        if hasattr(obj, '_prefetched_objects_cache') and 'children' in obj._prefetched_objects_cache:
            children = obj._prefetched_objects_cache['children']
        else:
            children = Reply.objects.filter(parent=obj, is_deleted=False).select_related('author')
        return ReplySerializer(children, many=True, context={
            **self.context,
            'depth': current_depth + 1
        }).data


class MessageSerializer(serializers.ModelSerializer):
    """消息序列化器"""
    sender_name = serializers.CharField(source='sender.name', read_only=True)

    class Meta:
        model = Message
        fields = ['id', 'msg_type', 'content', 'sender_name', 'is_read', 'created_at']


class ChatHistorySerializer(serializers.ModelSerializer):
    """聊天历史序列化器"""
    class Meta:
        model = ChatHistory
        fields = ['chat_id', 'question', 'answer', 'timestamp']
        read_only_fields = ['chat_id', 'timestamp']


class FollowSerializer(serializers.ModelSerializer):
    """关注序列化器"""
    follower_name = serializers.CharField(source='follower.name', read_only=True)
    following_name = serializers.CharField(source='following.name', read_only=True)

    class Meta:
        model = Follow
        fields = ['id', 'follower_name', 'following_name', 'created_at']


class ExpertApplicationSerializer(serializers.ModelSerializer):
    """专家申请序列化器"""
    applicant_name = serializers.CharField(source='applicant.name', read_only=True)
    reviewer_name = serializers.SerializerMethodField()
    field_display = serializers.CharField(source='get_field_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = ExpertApplication
        fields = [
            'id', 'applicant', 'applicant_name', 'field', 'field_display',
            'credentials', 'languages', 'reason', 'supporting_docs',
            'status', 'status_display', 'reviewer', 'reviewer_name',
            'review_comment', 'reviewed_at', 'created_at'
        ]
        read_only_fields = [
            'id', 'created_at', 'applicant', 'applicant_name', 'field_display',
            'status', 'status_display', 'reviewer', 'reviewer_name',
            'review_comment', 'reviewed_at',
        ]

    def get_reviewer_name(self, obj):
        if obj.reviewer:
            return obj.reviewer.name or obj.reviewer.username
        return None


class ExpertEndorsementSerializer(serializers.ModelSerializer):
    """专家背书序列化器"""
    expert_name = serializers.CharField(source='expert.name', read_only=True)
    endorsement_type_display = serializers.CharField(source='get_endorsement_type_display', read_only=True)

    class Meta:
        model = ExpertEndorsement
        fields = [
            'id', 'post', 'reply', 'expert', 'expert_name',
            'endorsement_type', 'endorsement_type_display', 'comment', 'created_at'
        ]
        read_only_fields = [
            'id', 'created_at', 'expert', 'expert_name',
            'endorsement_type_display',
        ]
