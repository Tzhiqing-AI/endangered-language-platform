"""
REST API URL 配置
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    UserViewSet, LanguageViewSet, CategoryViewSet, TagViewSet,
    PostViewSet, ReplyViewSet, MessageViewSet, ChatHistoryViewSet, FollowViewSet,
    ExpertApplicationViewSet, ExpertApplicationReviewView, MyExpertApplicationView,
    ExpertVerifyPostView, ExpertEndorseView,
    ContentModerationQueueView, ContentModerationActionView
)

app_name = 'rest_api'

router = DefaultRouter(trailing_slash=False)
router.register(r'users', UserViewSet, basename='user')
router.register(r'languages', LanguageViewSet, basename='language')
router.register(r'categories', CategoryViewSet, basename='category')
router.register(r'tags', TagViewSet, basename='tag')
router.register(r'posts', PostViewSet, basename='post')
router.register(r'replies', ReplyViewSet, basename='reply')
router.register(r'messages', MessageViewSet, basename='message')
router.register(r'chat', ChatHistoryViewSet, basename='chat')
router.register(r'follow', FollowViewSet, basename='follow')

urlpatterns = [
    path('', include(router.urls)),
    # 专家认证API
    path('expert/applications/', ExpertApplicationViewSet.as_view({'get': 'list', 'post': 'create'})),
    path('expert/applications/<int:pk>/', ExpertApplicationViewSet.as_view({'get': 'retrieve'})),
    path('expert/applications/<int:pk>/review/', ExpertApplicationReviewView.as_view()),
    path('expert/applications/my/', MyExpertApplicationView.as_view()),
    path('expert/verify-post/', ExpertVerifyPostView.as_view()),
    path('expert/endorse/', ExpertEndorseView.as_view()),
    # 内容审核API
    path('moderation/queue/', ContentModerationQueueView.as_view()),
    path('moderation/action/', ContentModerationActionView.as_view()),
]
