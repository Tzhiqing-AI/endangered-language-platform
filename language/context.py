"""
自定义模板上下文处理器
"""
def user_context(request):
    """
    将用户信息添加到所有模板上下文

    在settings.py的TEMPLATES中添加:
    'OPTIONS': {
        'context_processors': [
            ...
            'language.context.user_context',
        ],
    }
    """
    username = request.session.get('username', '')
    is_admin = False

    if username:
        try:
            from User.models import User
            user = User.objects.get(username=username)
            is_admin = user.admin == 1
        except User.DoesNotExist:
            pass

    return {
        'is_admin': is_admin,
    }
