from django.urls import path
from . import views

app_name = 'Community'

urlpatterns = [
    path('', views.community_home, name='community_home'),
    path('post/<int:post_id>/', views.post_detail, name='post_detail'),
    path('post/create/', views.create_post, name='create_post'),
    path('post/<int:post_id>/edit/', views.edit_post, name='edit_post'),
    path('post/<int:post_id>/delete/', views.delete_post, name='delete_post'),
    path('post/<int:post_id>/reply/', views.create_reply, name='create_reply'),
    path('post/<int:post_id>/like/', views.like_post, name='like_post'),
    path('post/<int:post_id>/collect/', views.collect_post, name='collect_post'),
    path('post/<int:post_id>/report/', views.report_post, name='report_post'),
    path('reply/<int:reply_id>/like/', views.like_reply, name='like_reply'),
    path('reply/<int:reply_id>/delete/', views.delete_reply, name='delete_reply'),
    path('user/<str:user_id>/', views.user_profile, name='user_profile'),
    path('user/<str:user_id>/follow/', views.follow_user, name='follow_user'),
    path('collections/', views.user_collections, name='user_collections'),
    path('reputation/', views.user_reputation_rank, name='reputation_rank'),
    path('messages/', views.messages, name='messages'),
    path('messages/mark-read/', views.mark_read_messages, name='mark_read_messages'),
    path('search/', views.search_posts, name='search_posts'),
    path('category/<int:category_id>/', views.category_posts, name='category_posts'),
    path('tag/<int:tag_id>/', views.tag_posts, name='tag_posts'),
    # 内容审核管理
    path('moderation/queue/', views.moderation_queue, name='moderation_queue'),
    path('moderation/action/', views.moderate_content, name='moderate_content'),
    # 社交图谱 API（Neo4j）
    path('api/follow-network/<str:user_id>/', views.api_follow_network, name='api_follow_network'),
    path('api/mutual-follows/<str:user_id>/', views.api_mutual_follows, name='api_mutual_follows'),
    # 社交图谱可视化页面
    path('social-graph/<str:user_id>/', views.social_graph, name='social_graph'),
]
