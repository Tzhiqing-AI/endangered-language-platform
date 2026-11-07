"""
User 用户中心视图

包含用户资料查看、编辑、头像上传等功能
"""

import os
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.hashers import check_password
from django.conf import settings

from User.models import User
from .models import UserProfile
from .forms import UserProfileForm, AvatarUploadForm

logger = logging.getLogger(__name__)


def get_user_from_session(request):
    """从 Session 获取用户"""
    username = request.session.get('username', '')
    if not username:
        return None
    try:
        return User.objects.get(username=username)
    except User.DoesNotExist:
        return None


def get_or_create_profile(user):
    """获取或创建用户资料"""
    profile, created = UserProfile.objects.get_or_create(user=user)
    return profile


def get_user_stats(user):
    """获取用户中心侧边栏需要的统计数据"""
    from Community.models import Post, Reply, Follow
    return {
        'post_count': Post.objects.filter(author=user, status='published').count(),
        'reply_count': Reply.objects.filter(author=user, is_deleted=False).count(),
        'followers_count': Follow.objects.filter(following=user).count(),
        'following_count': Follow.objects.filter(follower=user).count(),
        'is_admin': user.admin == 1,
        'is_expert': user.expert == 1,
    }


def profile_view(request):
    """用户中心 - 查看资料"""
    user = get_user_from_session(request)
    if not user:
        return redirect('/User/login/?next=/User/profile/')

    profile = get_or_create_profile(user)
    stats = get_user_stats(user)

    # 获取用户显示名称
    name = user.name if user.name else user.username

    context = {
        'user': user,
        'profile': profile,
        'name': name,
        'username': user.username,
        **stats,
    }
    return render(request, 'user/user_profile.html', context)


@require_http_methods(["GET", "POST"])
@csrf_protect
def profile_edit(request):
    """用户中心 - 编辑资料"""
    user = get_user_from_session(request)
    if not user:
        return redirect('/User/login/?next=/User/profile/edit/')

    profile = get_or_create_profile(user)
    stats = get_user_stats(user)

    if request.method == 'GET':
        form = UserProfileForm(instance=profile, user=user)
        context = {
            'form': form,
            'user': user,
            'profile': profile,
            'name': user.name if user.name else user.username,
            'username': user.username,
            **stats,
        }
        return render(request, 'user/user_profile_edit.html', context)

    # POST - 处理表单提交
    form = UserProfileForm(request.POST, instance=profile, user=user)
    if form.is_valid():
        form.save()
        # 更新 session 中的 name
        user.refresh_from_db()
        if user.name:
            request.session['name'] = user.name
        messages.success(request, '资料更新成功！')
        return redirect('/User/profile/')
    else:
        messages.error(request, '资料更新失败，请检查输入。')
        context = {
            'form': form,
            'user': user,
            'profile': profile,
            'name': user.name if user.name else user.username,
            'username': user.username,
            **stats,
        }
        return render(request, 'user/user_profile_edit.html', context)


@require_http_methods(["GET", "POST"])
@csrf_protect
def avatar_upload(request):
    """用户中心 - 上传头像"""
    user = get_user_from_session(request)
    if not user:
        return redirect('/User/login/?next=/User/profile/avatar/')

    profile = get_or_create_profile(user)
    stats = get_user_stats(user)

    if request.method == 'GET':
        form = AvatarUploadForm(instance=profile)
        context = {
            'form': form,
            'user': user,
            'profile': profile,
            'name': user.name if user.name else user.username,
            'username': user.username,
            **stats,
        }
        return render(request, 'user/user_avatar.html', context)

    # POST - 处理头像上传
    form = AvatarUploadForm(request.POST, request.FILES, instance=profile)
    if form.is_valid():
        # 删除旧头像（如果不是默认头像）
        if profile.avatar and profile.avatar.name != 'avatars/default.png':
            old_avatar_path = os.path.join(settings.MEDIA_ROOT, profile.avatar.name)
            if os.path.exists(old_avatar_path):
                try:
                    os.remove(old_avatar_path)
                except Exception as e:
                    logger.warning(f"删除旧头像失败: {e}")

        form.save()
        messages.success(request, '头像上传成功！')
        return redirect('/User/profile/')
    else:
        messages.error(request, '头像上传失败，请检查文件格式和大小。')
        context = {
            'form': form,
            'user': user,
            'profile': profile,
            'name': user.name if user.name else user.username,
            'username': user.username,
            **stats,
        }
        return render(request, 'user/user_avatar.html', context)


@require_http_methods(["GET", "POST"])
@csrf_protect
def change_password(request):
    """用户中心 - 修改密码"""
    user = get_user_from_session(request)
    if not user:
        return redirect('/User/login/?next=/User/profile/password/')

    profile = get_or_create_profile(user)
    stats = get_user_stats(user)

    if request.method == 'GET':
        return render(request, 'user/user_password.html', {
            'user': user,
            'profile': profile,
            'name': user.name if user.name else user.username,
            'username': user.username,
            **stats,
        })

    # POST - 处理密码修改
    old_password = request.POST.get('old_password', '')
    new_password = request.POST.get('new_password', '')
    confirm_password = request.POST.get('confirm_password', '')

    base_ctx = {
        'user': user,
        'profile': profile,
        'name': user.name if user.name else user.username,
        'username': user.username,
        **stats,
    }

    # 验证旧密码
    if not check_password(old_password, user.password):
        messages.error(request, '原密码错误！')
        return render(request, 'user/user_password.html', base_ctx)

    # 验证新密码
    if len(new_password) < 6:
        messages.error(request, '新密码至少6位！')
        return render(request, 'user/user_password.html', base_ctx)

    if new_password != confirm_password:
        messages.error(request, '两次输入的新密码不一致！')
        return render(request, 'user/user_password.html', base_ctx)

    from django.contrib.auth.password_validation import validate_password
    from django.core.exceptions import ValidationError

    try:
        validate_password(new_password, user=user)
    except ValidationError as exc:
        messages.error(request, ' '.join(exc.messages))
        return render(request, 'user/user_password.html', base_ctx)

    # 更新密码
    from django.contrib.auth.hashers import make_password
    user.password = make_password(new_password)
    user.save()

    messages.success(request, '密码修改成功！')
    return redirect('/User/profile/')
