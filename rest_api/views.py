"""
REST API 视图

提供社区论坛的 RESTful API 接口
"""
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db import transaction
from django.db.models import Q, Count, Prefetch
from django.shortcuts import get_object_or_404

from User.models import User
from Dashboard.models import Language
from Community.models import Post, Reply, Category, Tag, Message, Like, Collect, Follow, ContentModeration, ExpertEndorsement
from User.models import ExpertApplication, ChatHistory
from .serializers import (
    UserSerializer, UserProfileSerializer, LanguageSerializer,
    PostListSerializer, PostDetailSerializer, ReplySerializer,
    CategorySerializer, TagSerializer, MessageSerializer,
    ChatHistorySerializer, FollowSerializer,
    ExpertApplicationSerializer, ExpertEndorsementSerializer
)


# =============================================================================
# 权限类：统一基于 request.user（由 SessionUserMiddleware 注入）
# =============================================================================

class IsAdminUser(permissions.BasePermission):
    """仅管理员。"""

    def has_permission(self, request, view):
        return bool(request.user and getattr(request.user, 'is_authenticated', False)
                    and getattr(request.user, 'admin', 0) == 1)


class IsExpertUser(permissions.BasePermission):
    """仅专家或管理员。"""

    def has_permission(self, request, view):
        return bool(request.user and getattr(request.user, 'is_authenticated', False)
                    and (getattr(request.user, 'expert', 0) == 1
                         or getattr(request.user, 'admin', 0) == 1))


class IsAdminOrExpertUser(permissions.BasePermission):
    """管理员或专家。"""

    def has_permission(self, request, view):
        return bool(request.user and getattr(request.user, 'is_authenticated', False)
                    and (getattr(request.user, 'admin', 0) == 1
                         or getattr(request.user, 'expert', 0) == 1))


class IsAdminOrReadOnly(permissions.BasePermission):
    """公开只读，写操作仅管理员。"""

    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return IsAdminUser().has_permission(request, view)


class IsAuthorOrReadOnly(permissions.BasePermission):
    """对象级权限：作者本人或管理员可写。"""

    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        author = getattr(obj, 'author', None)
        if author == request.user:
            return True
        return getattr(request.user, 'admin', 0) == 1


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """用户 API：列表/详情仅管理员可见，个人资料接口仅本人可用。"""
    queryset = User.objects.filter(is_active=True).order_by('username')
    serializer_class = UserSerializer
    lookup_field = 'username'
    permission_classes = [IsAdminUser]

    @action(detail=False, methods=['get'], permission_classes=[permissions.IsAuthenticated])
    def profile(self, request):
        """获取当前用户资料"""
        serializer = UserSerializer(request.user)
        return Response(serializer.data)


class LanguageViewSet(viewsets.ModelViewSet):
    """濒危语言 API：公开只读，写操作仅管理员。"""
    queryset = Language.objects.all().order_by('language_id')
    serializer_class = LanguageSerializer
    permission_classes = [IsAdminOrReadOnly]

    def get_queryset(self):
        queryset = Language.objects.all()
        level = self.request.query_params.get('level')
        country = self.request.query_params.get('country')
        search = self.request.query_params.get('search')

        if level:
            queryset = queryset.filter(level_of_endangerment=level)
        if country:
            queryset = queryset.filter(country__icontains=country)
        if search:
            queryset = queryset.filter(
                Q(language_name__icontains=search) |
                Q(language_family__icontains=search)
            )
        return queryset.order_by('language_id')


class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """分类 API"""
    queryset = Category.objects.filter(is_active=True).annotate(post_count=Count('posts')).order_by('sort_order', 'id')
    serializer_class = CategorySerializer
    permission_classes = [permissions.AllowAny]


class TagViewSet(viewsets.ReadOnlyModelViewSet):
    """标签 API"""
    queryset = Tag.objects.all().order_by('name')
    serializer_class = TagSerializer
    permission_classes = [permissions.AllowAny]


class PostViewSet(viewsets.ModelViewSet):
    """帖子 API：公开只读，作者本人或管理员可改删。"""
    queryset = Post.objects.filter(status='published')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsAuthorOrReadOnly]

    def get_serializer_class(self):
        if self.action == 'list':
            return PostListSerializer
        return PostDetailSerializer

    def get_queryset(self):
        queryset = Post.objects.filter(status='published')
        category = self.request.query_params.get('category')
        tag = self.request.query_params.get('tag')
        search = self.request.query_params.get('search')

        if category:
            queryset = queryset.filter(category_id=category)
        if tag:
            queryset = queryset.filter(tags__id=tag)
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) |
                Q(content__icontains=search)
            )
        return queryset.order_by('-is_pinned', '-created_at')

    def perform_create(self, serializer):
        user = self.request.user
        if user.expert == 1:
            serializer.save(
                author=user,
                status='published',
                needs_moderation=False,
                is_expert_verified=True,
                verified_by=user,
            )
        else:
            serializer.save(
                author=user,
                status='draft',
                needs_moderation=True,
                moderation_queue='pending',
            )

    @action(detail=True, methods=['post'])
    def like(self, request, pk=None):
        """点赞帖子"""
        post = self.get_object()
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        user = User.objects.get(username=username)
        from Community.services import toggle_like
        result = toggle_like(user, post=post)
        return Response(result)

    @action(detail=True, methods=['post'])
    def collect(self, request, pk=None):
        """收藏帖子"""
        post = self.get_object()
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        user = User.objects.get(username=username)
        from Community.services import toggle_collect
        result = toggle_collect(user, post)
        return Response(result)

    @action(detail=True, methods=['get'])
    def replies(self, request, pk=None):
        """获取帖子的回复"""
        post = self.get_object()
        # Prefetch children replies to avoid N+1 query
        replies = Reply.objects.filter(post=post, parent=None, is_deleted=False).prefetch_related(
            Prefetch('children', queryset=Reply.objects.filter(is_deleted=False).order_by('created_at'))
        )
        serializer = ReplySerializer(replies, many=True)
        return Response(serializer.data)


class ReplyViewSet(viewsets.ModelViewSet):
    """回复 API：公开只读，作者本人或管理员可改删。"""
    queryset = Reply.objects.filter(is_deleted=False)
    serializer_class = ReplySerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsAuthorOrReadOnly]

    def get_queryset(self):
        post_id = self.request.query_params.get('post')
        if post_id:
            return Reply.objects.filter(post_id=post_id, is_deleted=False).order_by('created_at')
        return Reply.objects.filter(is_deleted=False).order_by('created_at')

    def perform_create(self, serializer):
        post = serializer.validated_data.get('post')
        if post.status != 'published':
            from rest_framework.exceptions import ValidationError
            raise ValidationError({'post': '只能回复已发布的帖子'})
        parent = serializer.validated_data.get('parent')
        if parent and parent.post_id != post.id:
            from rest_framework.exceptions import ValidationError
            raise ValidationError({'parent': '父回复不属于该帖子'})
        serializer.save(author=self.request.user)

    @action(detail=True, methods=['post'])
    def like(self, request, pk=None):
        """点赞回复"""
        reply = self.get_object()
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        user = User.objects.get(username=username)
        from Community.services import toggle_like
        result = toggle_like(user, reply=reply)
        return Response(result)


class MessageViewSet(viewsets.ReadOnlyModelViewSet):
    """消息 API：只能查看自己的消息。"""
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Message.objects.filter(recipient=self.request.user).order_by('-created_at')

    @action(detail=False, methods=['post'])
    def mark_read(self, request):
        """标记所有消息为已读"""
        Message.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
        return Response({'success': True})

    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        """获取未读消息数"""
        count = Message.objects.filter(recipient=request.user, is_read=False).count()
        return Response({'count': count})


class ChatHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    """聊天历史 API：只能查看自己的记录。"""
    serializer_class = ChatHistorySerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ChatHistory.objects.filter(username=self.request.user).order_by('-timestamp')


class FollowViewSet(viewsets.ReadOnlyModelViewSet):
    """关注 API：列表仅返回自己的关注；关注/取关走 toggle。"""
    serializer_class = FollowSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Follow.objects.filter(follower=self.request.user).order_by('-created_at')

    @action(detail=False, methods=['post'])
    def toggle(self, request):
        """切换关注状态"""
        target_username = request.data.get('username')
        if not target_username:
            return Response({'error': '缺少用户名'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            target_user = User.objects.get(username=target_username, is_active=True)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

        from Community.services import toggle_follow
        result = toggle_follow(request.user, target_user)
        if result['error']:
            return Response({'error': result['error']}, status=status.HTTP_400_BAD_REQUEST)
        return Response({'following': result['following']})


# =============================================================================
# 专家认证与内容审核 API
# =============================================================================

class ExpertApplicationViewSet(viewsets.ModelViewSet):
    """专家申请 API：普通用户只能看自己的申请，管理员可看全部。"""
    queryset = ExpertApplication.objects.all()
    serializer_class = ExpertApplicationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = ExpertApplication.objects.all()
        if getattr(self.request.user, 'admin', 0) != 1:
            queryset = queryset.filter(applicant=self.request.user)
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        return queryset.order_by('-created_at')

    def create(self, request):
        """用户提交专家申请"""
        user = request.user
        if user.expert == 1:
            return Response({'error': '您已是专家用户'}, status=status.HTTP_400_BAD_REQUEST)

        pending = ExpertApplication.objects.filter(
            applicant=user, status='pending'
        ).exists()
        if pending:
            return Response({'error': '您有待处理的申请，请等待审核结果'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(applicant=user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ExpertApplicationReviewView(APIView):
    """专家申请审核：仅管理员。"""
    permission_classes = [IsAdminUser]

    def post(self, request, pk):
        moderator = request.user
        if moderator.admin != 1:
            return Response({'error': '仅管理员可以审核申请'}, status=status.HTTP_403_FORBIDDEN)

        application = get_object_or_404(ExpertApplication, pk=pk)
        if application.status != 'pending':
            return Response({'error': '该申请已处理'}, status=status.HTTP_400_BAD_REQUEST)

        action = request.data.get('action')
        comment = request.data.get('comment', '')

        if action not in ['approve', 'reject']:
            return Response({'error': '无效的操作'}, status=status.HTTP_400_BAD_REQUEST)

        from django.utils import timezone
        with transaction.atomic():
            application.status = 'approved' if action == 'approve' else 'rejected'
            application.reviewer = moderator
            application.review_comment = comment
            application.reviewed_at = timezone.now()
            application.save()

            if action == 'approve':
                user = application.applicant
                user.expert = 1
                user.expert_verified_at = timezone.now()
                user.save(update_fields=['expert', 'expert_verified_at'])
                Message.objects.create(
                    recipient=user,
                    msg_type='system',
                    content='您的专家申请已通过！您现在可以在社区中发布权威认证的内容了。'
                )
            else:
                Message.objects.create(
                    recipient=application.applicant,
                    msg_type='system',
                    content=f'您的专家申请未通过。审核备注：{comment or "无"}'
                )

        return Response({'success': True, 'status': application.status})


class MyExpertApplicationView(APIView):
    """获取当前用户的专家申请记录"""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        applications = ExpertApplication.objects.filter(applicant=request.user).order_by('-created_at')
        serializer = ExpertApplicationSerializer(applications, many=True)
        return Response(serializer.data)


class ExpertVerifyPostView(APIView):
    """专家验证内容"""
    permission_classes = [IsExpertUser]

    def post(self, request):
        expert = request.user
        post = get_object_or_404(Post, pk=request.data.get('post_id'))

        from django.utils import timezone
        post.is_expert_verified = True
        post.verified_by = expert
        post.verified_at = timezone.now()
        post.save(update_fields=['is_expert_verified', 'verified_by', 'verified_at'])

        return Response({'success': True, 'message': '内容已标记为已验证'})


class ExpertEndorseView(APIView):
    """专家背书内容"""
    permission_classes = [IsExpertUser]

    def post(self, request):
        expert = request.user
        post_id = request.data.get('post_id')
        reply_id = request.data.get('reply_id')
        endorsement_type = request.data.get('type')
        comment = request.data.get('comment', '')

        if not post_id and not reply_id:
            return Response({'error': '必须指定帖子或回复'}, status=status.HTTP_400_BAD_REQUEST)
        if post_id and reply_id:
            return Response({'error': '只能指定一个背书对象'}, status=status.HTTP_400_BAD_REQUEST)
        if endorsement_type not in ['verified', 'corrected', 'annotated']:
            return Response({'error': '无效的背书类型'}, status=status.HTTP_400_BAD_REQUEST)

        post = get_object_or_404(Post, pk=post_id) if post_id else None
        reply = get_object_or_404(Reply, pk=reply_id, is_deleted=False) if reply_id else None

        with transaction.atomic():
            ExpertEndorsement.objects.create(
                expert=expert,
                endorsement_type=endorsement_type,
                comment=comment,
                post=post,
                reply=reply,
            )
            if post:
                post.is_expert_verified = True
                post.verified_by = expert
                from django.utils import timezone
                post.verified_at = timezone.now()
                post.save(update_fields=['is_expert_verified', 'verified_by', 'verified_at'])

        return Response({'success': True, 'message': '背书成功'})


class ContentModerationQueueView(APIView):
    """内容审核队列"""
    permission_classes = [IsAdminOrExpertUser]

    def get(self, request):
        pending_posts = Post.objects.filter(
            needs_moderation=True,
            moderation_queue='pending'
        ).select_related('author').order_by('created_at')[:50]

        result = []
        for post in pending_posts:
            result.append({
                'type': 'post',
                'id': post.id,
                'title': post.title,
                'author': post.author.name or post.author.username,
                'created_at': post.created_at.isoformat(),
                'preview': post.content[:200],
            })

        return Response(result)


class ContentModerationActionView(APIView):
    """内容审核操作"""
    permission_classes = [IsAdminOrExpertUser]

    def post(self, request):
        moderator = request.user
        content_type = request.data.get('type')
        content_id = request.data.get('id')
        action = request.data.get('action')
        reason = request.data.get('reason', '')

        if content_type not in ['post', 'reply'] or action not in ['approve', 'reject', 'hide', 'restore']:
            return Response({'error': '无效参数'}, status=status.HTTP_400_BAD_REQUEST)

        model = Post if content_type == 'post' else Reply
        content = get_object_or_404(model, pk=content_id)

        with transaction.atomic():
            ContentModeration.objects.create(
                content_type=content_type,
                content_id=content_id,
                moderator=moderator,
                action=action,
                reason=reason
            )

            if action == 'approve':
                if content_type == 'post':
                    content.status = 'published'
                    content.needs_moderation = False
                    content.moderation_queue = 'approved'
                else:
                    content.is_deleted = False
            elif action == 'reject':
                if content_type == 'post':
                    content.status = 'hidden'
                    content.needs_moderation = False
                    content.moderation_queue = 'rejected'
                else:
                    content.is_deleted = True
            elif action == 'hide':
                if content_type == 'post':
                    content.status = 'hidden'
                else:
                    content.is_deleted = True
            elif action == 'restore':
                if content_type == 'post':
                    content.status = 'published'
                    content.needs_moderation = False
                else:
                    content.is_deleted = False
            content.save()
            if content_type == 'post':
                from Community.services import refresh_tag_counts
                refresh_tag_counts(content.tags.all())

            action_text = {'approve': '通过', 'reject': '拒绝', 'hide': '隐藏', 'restore': '恢复'}[action]
            Message.objects.create(
                recipient=content.author,
                msg_type='system',
                content=f'您发布的内容已被管理员{action_text}。',
                post=content if content_type == 'post' else None
            )

        return Response({'success': True})
