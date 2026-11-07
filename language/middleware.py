"""把自定义 Session 用户注入 request.user，让 Django/DRF 权限体系可用。"""

from django.contrib.auth.models import AnonymousUser


class SessionUserMiddleware:
    """在 AuthenticationMiddleware 之后解析应用自身的 Session 用户。

    - 会话中的用户不存在或被禁用时，直接清理会话；
    - Django Admin 使用独立的 auth_user 表，必须跳过，避免覆盖后台登录用户。
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.path.startswith('/admin/'):
            username = request.session.get('username', '')
            if username:
                from User.models import User

                try:
                    user = User.objects.get(username=username)
                except User.DoesNotExist:
                    request.session.flush()
                    request.user = AnonymousUser()
                else:
                    if not user.is_active:
                        request.session.flush()
                        request.user = AnonymousUser()
                    else:
                        request.user = user
        return self.get_response(request)
