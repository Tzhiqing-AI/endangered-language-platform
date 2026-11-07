"""
自定义模板上下文处理器
"""
def user_context(request):
    """将当前登录用户的管理员标识注入所有模板。"""
    from language.utils import get_authenticated_user

    user = get_authenticated_user(request)
    return {
        'is_admin': bool(user and user.admin == 1),
    }


def site_context(request):
    """站点级配置（地图 Key 等），避免把密钥写进模板。"""
    from django.conf import settings

    return {
        'amap_js_key': getattr(settings, 'AMAP_JS_KEY', ''),
    }
