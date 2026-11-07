from django.shortcuts import render, HttpResponseRedirect, redirect
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.contrib.auth import logout
from django.utils.http import url_has_allowed_host_and_scheme
from User.models import User, ExpertApplication
from Dashboard.common import get_user_context
from language.utils import rate_limit
import re

# 用户中心视图
from .profile_views import profile_view, profile_edit, avatar_upload, change_password


def _safe_redirect_url(url, request, default_url='/Dashboard/language/'):
    """验证重定向 URL 是否安全（防止开放重定向攻击）"""
    if url and url_has_allowed_host_and_scheme(url, allowed_hosts={request.get_host()}):
        return url
    return default_url


def get_redirect_url(request, default_url='/Dashboard/language/'):
    """获取重定向URL，优先使用next参数，否则使用默认URL"""
    next_url = request.GET.get('next', '')
    if next_url:
        return _safe_redirect_url(next_url, request, default_url)
    referer = request.META.get('HTTP_REFERER', '')
    if referer and 'login' not in referer and 'register' not in referer:
        return _safe_redirect_url(referer, request, default_url)
    return default_url


@rate_limit(max_requests=5, window_seconds=60)
@require_http_methods(["GET", "POST"])
def language_register(request):
    if request.method == 'GET':
        next_url = request.GET.get('next', '')
        return render(request, 'language_register.html', {'next': next_url})

    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '')
    confirm_password = request.POST.get('confirm_password', '')
    name = request.POST.get('name', '').strip()
    next_url = request.POST.get('next', '')

    if not username:
        return render(request, 'language_register.html', {
            'message': '用户名不能为空',
            'next': next_url
        })

    if len(username) < 3 or len(username) > 20:
        return render(request, 'language_register.html', {
            'message': '用户名必须为3-20位',
            'next': next_url
        })

    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return render(request, 'language_register.html', {
            'message': '用户名只能包含字母、数字和下划线',
            'next': next_url
        })

    if User.objects.filter(username=username).exists():
        return render(request, 'language_register.html', {
            'message': '该用户名已存在',
            'next': next_url
        })

    if len(password) < 6:
        return render(request, 'language_register.html', {
            'message': '密码至少6位',
            'next': next_url
        })

    if password != confirm_password:
        return render(request, 'language_register.html', {
            'message': '两次输入的密码不一致',
            'next': next_url
        })

    if len(name) > 20:
        return render(request, 'language_register.html', {
            'message': '昵称不能超过20个字符',
            'next': next_url
        })

    try:
        validate_password(password)
    except ValidationError as exc:
        return render(request, 'language_register.html', {
            'message': ' '.join(exc.messages),
            'next': next_url
        })

    user = User.objects.create(
        username=username,
        password=make_password(password),
        name=name if name else username,
        admin=0,  # 默认普通用户
    )
    user.update_last_login()

    # 防御会话固定攻击：认证成功后更换 session id（保留 session 数据）
    request.session.cycle_key()
    request.session["username"] = user.username
    request.session["name"] = user.name

    # 创建用户资料
    from .models import UserProfile
    UserProfile.objects.create(user=user)

    if next_url:
        return HttpResponseRedirect(_safe_redirect_url(next_url, request))
    return HttpResponseRedirect('/Dashboard/language/')

@rate_limit(max_requests=5, window_seconds=60)
@require_http_methods(["GET", "POST"])
def language_login(request):
    if request.method == 'GET':
        next_url = request.GET.get('next', '')
        return render(request, 'language_login.html', {'next': next_url})

    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '')
    next_url = request.POST.get('next', '')

    if not username or not password:
        return render(request, 'language_login.html', {
            'message': '用户名和密码不能为空',
            'next': next_url
        })

    try:
        user = User.objects.get(username=username)
        if not user.is_active:
            return render(request, 'language_login.html', {
                'message': '该账号已被禁用，请联系管理员',
                'next': next_url
            })
        if check_password(password, user.password):
            # 防御会话固定攻击：登录成功后更换 session id（保留 session 数据）
            request.session.cycle_key()
            request.session["username"] = user.username
            request.session["name"] = user.name
            # 勾选“记住登录状态”时使用默认有效期；否则关闭浏览器即失效
            if request.POST.get('remember_me') == 'yes':
                request.session.set_expiry(None)
            else:
                request.session.set_expiry(0)
            user.update_last_login()

            if next_url:
                return HttpResponseRedirect(_safe_redirect_url(next_url, request))
            return HttpResponseRedirect('/Dashboard/language/')
        else:
            return render(request, 'language_login.html', {
                'message': '用户名或密码错误',
                'next': next_url
            })
    except User.DoesNotExist:
        return render(request, 'language_login.html', {
            'message': '用户名或密码错误',
            'next': next_url
        })

@require_http_methods(["GET"])
def password_help(request):
    """密码找回说明页（当前系统不发送邮件，由管理员协助重置）。"""
    return render(request, 'user/password_help.html')


@require_POST
def logout_view(request):
    request.session.flush()
    logout(request)
    return HttpResponseRedirect('/Dashboard/language/')


# ==================== 专家认证 ====================

@require_http_methods(["GET", "POST"])
def expert_application(request):
    """专家申请页面"""
    user, name, _ = get_user_context(request)

    if not user:
        return redirect(f'/User/login/?next={request.path}')

    if request.method == 'GET':
        # 检查是否已是专家
        if user.expert == 1:
            return render(request, 'user/expert_already.html', {
                'name': name,
                'username': user.username,
            })

        # 检查是否有待处理申请
        pending = ExpertApplication.objects.filter(
            applicant=user, status='pending'
        ).exists()

        # 获取申请历史
        applications = ExpertApplication.objects.filter(
            applicant=user
        ).order_by('-created_at')[:5]

        return render(request, 'user/expert_application.html', {
            'name': name,
            'username': user.username,
            'pending': pending,
            'applications': applications,
        })

    # POST - 提交申请
    field = request.POST.get('field', '').strip()
    credentials = request.POST.get('credentials', '').strip()
    languages = request.POST.get('languages', '').strip()
    reason = request.POST.get('reason', '').strip()
    supporting_docs = request.POST.get('supporting_docs', '').strip()

    # 验证
    if not all([field, credentials, languages, reason]):
        return render(request, 'user/expert_application.html', {
            'name': name,
            'username': user.username,
            'error': '请填写所有必填项',
            'field': field,
            'credentials': credentials,
            'languages': languages,
            'reason': reason,
            'supporting_docs': supporting_docs,
        }, status=400)

    # 创建申请
    ExpertApplication.objects.create(
        applicant=user,
        field=field,
        credentials=credentials,
        languages=languages,
        reason=reason,
        supporting_docs=supporting_docs,
    )

    return render(request, 'user/expert_application_success.html', {
        'name': name,
        'username': user.username,
    })


@require_http_methods(["GET"])
def expert_application_status(request):
    """查看申请状态"""
    user, name, _ = get_user_context(request)

    if not user:
        return redirect('/User/login/?next=/User/expert/application/')

    # 如果已经是专家，直接跳转到"已为专家"页面
    if user.expert == 1:
        return render(request, 'user/expert_already.html', {
            'name': name,
            'username': user.username,
        })

    applications = ExpertApplication.objects.filter(
        applicant=user
    ).order_by('-created_at')

    return render(request, 'user/expert_application_status.html', {
        'name': name,
        'username': user.username,
        'applications': applications,
    })
