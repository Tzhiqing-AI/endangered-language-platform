"""
REST API 视图

提供社区论坛的 RESTful API 接口
"""
from rest_framework import viewsets, status, permissions, serializers
from django.db.models import F
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Q, Count, Prefetch

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


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """用户 API"""
    queryset = User.objects.all()
    serializer_class = UserSerializer
    lookup_field = 'username'
    permission_classes = [permissions.AllowAny]

    @action(detail=False, methods=['get'])
    def profile(self, request):
        """获取当前用户资料"""
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        try:
            user = User.objects.get(username=username)
            serializer = UserSerializer(user)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)


class LanguageViewSet(viewsets.ModelViewSet):
    """濒危语言 API"""
    queryset = Language.objects.all()
    serializer_class = LanguageSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

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
        return queryset


class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """分类 API"""
    queryset = Category.objects.filter(is_active=True).annotate(post_count=Count('posts'))
    serializer_class = CategorySerializer
    permission_classes = [permissions.AllowAny]


class TagViewSet(viewsets.ReadOnlyModelViewSet):
    """标签 API"""
    queryset = Tag.objects.all()
    serializer_class = TagSerializer
    permission_classes = [permissions.AllowAny]


class PostViewSet(viewsets.ModelViewSet):
    """帖子 API"""
    queryset = Post.objects.filter(status='published')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

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
        username = self.request.session.get('username')
        if username:
            user = User.objects.get(username=username)
            serializer.save(author=user)
        else:
            serializer.save()

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
    """回复 API"""
    queryset = Reply.objects.filter(is_deleted=False)
    serializer_class = ReplySerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        post_id = self.request.query_params.get('post')
        if post_id:
            return Reply.objects.filter(post_id=post_id, is_deleted=False)
        return Reply.objects.filter(is_deleted=False)

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
    """消息 API"""
    serializer_class = MessageSerializer

    def get_queryset(self):
        username = self.request.session.get('username')
        if not username:
            return Message.objects.none()
        return Message.objects.filter(recipient__username=username).order_by('-created_at')

    @action(detail=False, methods=['post'])
    def mark_read(self, request):
        """标记所有消息为已读"""
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        Message.objects.filter(recipient__username=username, is_read=False).update(is_read=True)
        return Response({'success': True})

    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        """获取未读消息数"""
        username = request.session.get('username')
        if not username:
            return Response({'count': 0})
        count = Message.objects.filter(recipient__username=username, is_read=False).count()
        return Response({'count': count})


class ChatHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    """聊天历史 API"""
    serializer_class = ChatHistorySerializer

    def get_queryset(self):
        username = self.request.session.get('username')
        if not username:
            return ChatHistory.objects.none()
        return ChatHistory.objects.filter(username=username).order_by('-timestamp')


class FollowViewSet(viewsets.ModelViewSet):
    """关注 API"""
    queryset = Follow.objects.all()
    serializer_class = FollowSerializer

    @action(detail=False, methods=['post'])
    def toggle(self, request):
        """切换关注状态"""
        username = request.session.get('username')
        target_username = request.data.get('username')

        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            user = User.objects.get(username=username)
            target_user = User.objects.get(username=target_username)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

        from Community.services import toggle_follow
        result = toggle_follow(user, target_user)
        if result['error']:
            return Response({'error': result['error']}, status=status.HTTP_400_BAD_REQUEST)
        return Response({'following': result['following']})


# =============================================================================
# 专家认证与内容审核 API
# =============================================================================

class IsAdminUser(permissions.BasePermission):
    """自定义权限：仅管理员"""
    def has_permission(self, request, view):
        username = request.session.get('username')
        if not username:
            return False
        try:
            user = User.objects.get(username=username)
            return user.admin == 1
        except User.DoesNotExist:
            return False


class IsExpertUser(permissions.BasePermission):
    """自定义权限：仅专家"""
    def has_permission(self, request, view):
        username = request.session.get('username')
        if not username:
            return False
        try:
            user = User.objects.get(username=username)
            return user.expert == 1
        except User.DoesNotExist:
            return False


class IsAdminOrExpertUser(permissions.BasePermission):
    """自定义权限：管理员或专家"""
    def has_permission(self, request, view):
        username = request.session.get('username')
        if not username:
            return False
        try:
            user = User.objects.get(username=username)
            return user.admin == 1 or user.expert == 1
        except User.DoesNotExist:
            return False


class ExpertApplicationViewSet(viewsets.ModelViewSet):
    """专家申请 API"""
    queryset = ExpertApplication.objects.all()
    serializer_class = ExpertApplicationSerializer

    def get_queryset(self):
        queryset = ExpertApplication.objects.all()
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        return queryset.order_by('-created_at')

    def create(self, request):
        """用户提交专家申请"""
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

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
    """专家申请审核"""

    def post(self, request, pk):
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            moderator = User.objects.get(username=username)
            if moderator.admin != 1:
                return Response({'error': '仅管理员可以审核申请'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

        try:
            application = ExpertApplication.objects.get(pk=pk)
        except ExpertApplication.DoesNotExist:
            return Response({'error': '申请不存在'}, status=status.HTTP_404_NOT_FOUND)

        if application.status != 'pending':
            return Response({'error': '该申请已处理'}, status=status.HTTP_400_BAD_REQUEST)

        action = request.data.get('action')
        comment = request.data.get('comment', '')

        if action not in ['approve', 'reject']:
            return Response({'error': '无效的操作'}, status=status.HTTP_400_BAD_REQUEST)

        from django.utils import timezone
        application.status = 'approved' if action == 'approve' else 'rejected'
        application.reviewer = moderator
        application.review_comment = comment
        application.reviewed_at = timezone.now()
        application.save()

        if action == 'approve':
            user = application.applicant
            user.expert = 1
            user.expert_verified_at = timezone.now()
            user.save()

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

    def get(self, request):
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

        applications = ExpertApplication.objects.filter(applicant=user).order_by('-created_at')
        serializer = ExpertApplicationSerializer(applications, many=True)
        return Response(serializer.data)


class ExpertVerifyPostView(APIView):
    """专家验证内容"""

    def post(self, request):
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            expert = User.objects.get(username=username)
            if expert.expert != 1:
                return Response({'error': '仅专家用户可以验证内容'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

        post_id = request.data.get('post_id')
        try:
            post = Post.objects.get(pk=post_id)
        except Post.DoesNotExist:
            return Response({'error': '帖子不存在'}, status=status.HTTP_404_NOT_FOUND)

        from django.utils import timezone
        post.is_expert_verified = True
        post.verified_by = expert
        post.verified_at = timezone.now()
        post.save()

        return Response({'success': True, 'message': '内容已标记为已验证'})


class ExpertEndorseView(APIView):
    """专家背书内容"""

    def post(self, request):
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            expert = User.objects.get(username=username)
            if expert.expert != 1:
                return Response({'error': '仅专家用户可以进行背书'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

        post_id = request.data.get('post_id')
        reply_id = request.data.get('reply_id')
        endorsement_type = request.data.get('type')
        comment = request.data.get('comment', '')

        if not endorsement_type:
            return Response({'error': '请选择背书类型'}, status=status.HTTP_400_BAD_REQUEST)

        if endorsement_type not in ['verified', 'corrected', 'annotated']:
            return Response({'error': '无效的背书类型'}, status=status.HTTP_400_BAD_REQUEST)

        endorsement = ExpertEndorsement.objects.create(
            expert=expert,
            endorsement_type=endorsement_type,
            comment=comment,
            **({'post_id': post_id} if post_id else {'reply_id': reply_id})
        )

        if post_id:
            post = Post.objects.get(pk=post_id)
            post.is_expert_verified = True
            post.verified_by = expert
            post.save()

        return Response({'success': True, 'message': '背书成功'})


class ContentModerationQueueView(APIView):
    """内容审核队列"""

    def get(self, request):
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            user = User.objects.get(username=username)
            if user.admin != 1 and user.expert != 1:
                return Response({'error': '无权访问'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

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

    def post(self, request):
        username = request.session.get('username')
        if not username:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            moderator = User.objects.get(username=username)
            if moderator.admin != 1 and moderator.expert != 1:
                return Response({'error': '无权审核'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)

        content_type = request.data.get('type')
        content_id = request.data.get('id')
        action = request.data.get('action')
        reason = request.data.get('reason', '')

        if content_type not in ['post', 'reply'] or action not in ['approve', 'reject', 'hide', 'restore']:
            return Response({'error': '无效参数'}, status=status.HTTP_400_BAD_REQUEST)

        if content_type == 'post':
            try:
                content = Post.objects.get(pk=content_id)
            except Post.DoesNotExist:
                return Response({'error': '帖子不存在'}, status=status.HTTP_404_NOT_FOUND)
        else:
            try:
                content = Reply.objects.get(pk=content_id)
            except Reply.DoesNotExist:
                return Response({'error': '回复不存在'}, status=status.HTTP_404_NOT_FOUND)

        ContentModeration.objects.create(
            content_type=content_type,
            content_id=content_id,
            moderator=moderator,
            action=action,
            reason=reason
        )

        from django.utils import timezone

        if action == 'approve':
            if content_type == 'post':
                content.status = 'published'
                content.needs_moderation = False
                content.moderation_queue = 'approved'
            else:
                content.is_deleted = False
            content.save()
        elif action == 'reject':
            if content_type == 'post':
                content.status = 'hidden'
                content.needs_moderation = False
                content.moderation_queue = 'rejected'
            else:
                content.is_deleted = True
            content.save()
        elif action == 'hide':
            if content_type == 'post':
                content.status = 'hidden'
            else:
                content.is_deleted = True
            content.save()
        elif action == 'restore':
            if content_type == 'post':
                content.status = 'published'
                content.needs_moderation = False
            else:
                content.is_deleted = False
            content.save()

        if action in ['approve', 'reject', 'hide', 'restore']:
            action_text = {'approve': '通过', 'reject': '拒绝', 'hide': '隐藏', 'restore': '恢复'}[action]
            Message.objects.create(
                recipient=content.author,
                msg_type='system',
                content=f'您发布的内容已被管理员{action_text}。',
                post=content if content_type == 'post' else None
            )

        return Response({'success': True})
