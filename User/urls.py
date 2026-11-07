from django.urls import path
from User.views import *
from User.expert_views import expert_management, expert_application_review

app_name = 'User'

urlpatterns = [
    # 认证相关
    path('register/', language_register, name='register'),
    path('login/', language_login, name='login'),
    path('password-help/', password_help, name='password_help'),
    path('logout/', logout_view, name='logout'),

    # 用户中心
    path('profile/', profile_view, name='profile'),
    path('profile/edit/', profile_edit, name='profile_edit'),
    path('profile/avatar/', avatar_upload, name='profile_avatar'),
    path('profile/password/', change_password, name='profile_password'),

    # 专家认证
    path('expert/application/', expert_application, name='expert_application'),
    path('expert/application/status/', expert_application_status, name='expert_application_status'),
    path('expert/management/', expert_management, name='expert_management'),
    path('expert/management/<int:pk>/review/', expert_application_review, name='expert_application_review'),
]