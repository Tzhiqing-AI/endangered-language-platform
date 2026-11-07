from django.http import JsonResponse
from functools import wraps
from django.shortcuts import redirect
from django.contrib.auth import logout
from django.core.cache import cache
from django.utils.html import escape
import bleach
import time

def api_response(status='success', message='', data=None):
    """Standardized API response format."""
    return JsonResponse({
        'status': 'success' if status == 'success' else 'error',
        'message': message,
        'data': data if data is not None else {}
    })

def _clear_stale_session(request):
    """会话里有用户名但用户已不存在/被禁用时，清掉脏会话。"""
    if request.session.get('username'):
        request.session.flush()


def login_required_custom(view_func):
    """装饰器：确保用户已通过 session 登录（API 端点用，返回 JSON 401）"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        user = get_authenticated_user(request)
        if not user:
            _clear_stale_session(request)
            return JsonResponse({
                'status': 'error',
                'message': '请先登录'
            }, status=401)
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def login_required_redirect(view_func):
    """装饰器：确保用户已通过 session 登录（模板视图用，未登录重定向到登录页）"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        user = get_authenticated_user(request)
        if not user:
            _clear_stale_session(request)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': '请先登录'}, status=401)
            from django.shortcuts import redirect
            return redirect(f'/User/login/?next={request.path}')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def expert_required(view_func):
    """装饰器：仅专家用户可访问"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        user = get_authenticated_user(request)
        if not user:
            _clear_stale_session(request)
            return JsonResponse({'status': 'error', 'message': '请先登录'}, status=401)
        if user.expert != 1:
            return JsonResponse({'status': 'error', 'message': '仅专家用户可访问'}, status=403)
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def moderator_required(view_func):
    """装饰器：仅管理员或专家可访问"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        user = get_authenticated_user(request)
        if not user:
            _clear_stale_session(request)
            return JsonResponse({'status': 'error', 'message': '请先登录'}, status=401)
        if user.admin != 1 and user.expert != 1:
            return JsonResponse({'status': 'error', 'message': '无权访问'}, status=403)
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def _client_ip(request):
    """仅当明确配置了可信反向代理时才读取 X-Forwarded-For。"""
    from django.conf import settings

    if getattr(settings, 'TRUST_PROXY_HEADERS', False):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def rate_limit(max_requests=10, window_seconds=60, methods=('POST',)):
    """
    速率限制装饰器

    Args:
        max_requests: 时间窗口内允许的最大请求数
        window_seconds: 时间窗口大小（秒）
        methods: 需要计数的请求方法，默认只限制写操作，避免正常翻页被拦
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if request.method not in methods:
                return view_func(request, *args, **kwargs)

            ip = _client_ip(request)
            identity = f"{ip}:{request.session.get('username', 'anon')}"

            current_time = time.time()
            count_key = f'rate_limit:{identity}:{view_func.__name__}:count'
            window_key = f'rate_limit:{identity}:{view_func.__name__}:window'

            window_start = cache.get(window_key)
            if window_start is None or (current_time - window_start) > window_seconds:
                cache.set(window_key, current_time, timeout=window_seconds + 10)
                cache.set(count_key, 1, timeout=window_seconds + 10)
                count = 1
            else:
                try:
                    count = cache.incr(count_key)
                except (ValueError, TypeError):
                    cache.set(count_key, 1, timeout=window_seconds + 10)
                    count = 1

            if count > max_requests:
                remaining_time = int(window_seconds - (current_time - window_start))
                return JsonResponse({
                    'status': 'error',
                    'message': f'请求过于频繁，请 {max(remaining_time, 1)} 秒后重试',
                    'retry_after': max(remaining_time, 1)
                }, status=429)

            response = view_func(request, *args, **kwargs)
            if isinstance(response, JsonResponse):
                response['X-RateLimit-Limit'] = str(max_requests)
                response['X-RateLimit-Remaining'] = str(max(max_requests - count, 0))
            return response
        return _wrapped_view
    return decorator

def get_authenticated_user(request):
    """获取当前登录用户；优先使用中间件注入的 request.user，避免重复查库。"""
    from User.models import User

    current = getattr(request, 'user', None)
    if isinstance(current, User):
        return current if current.is_active else None

    username = request.session.get('username', '')
    if not username:
        return None
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return None
    return user if user.is_active else None


# 允许的 HTML 标签（用于 AI 输出的安全渲染）
_ALLOWED_TAGS = ['br', 'b', 'i', 'u', 'em', 'strong', 'p', 'ul', 'ol', 'li', 'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote']
_ALLOWED_ATTRS = {}


def sanitize_ai_output(text):
    """
    净化 AI 生成的 HTML 内容，防止 XSS。

    先用 bleach 清除危险标签，再将换行符转为 <br>。
    返回的字符串可安全用于模板的 |safe 渲染。
    """
    if not text:
        return ''
    cleaned = bleach.clean(text, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS, strip=True)
    return cleaned.replace('\n', '<br>')
