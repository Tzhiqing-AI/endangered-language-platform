from django.template.defaultfilters import register


@register.filter(name='language_display_name')
def language_display_name(obj):
    """统一语言名称展示，空名称回退为「未命名语言 #<id>」。
    Usage: {{ language|language_display_name }}"""
    from Dashboard.common import display_language_name
    return display_language_name(obj)


@register.filter(name='name_source_display')
def name_source_display(value):
    """名称来源对外表述，隐藏 manual:unresolved 等内部标记。"""
    from Dashboard.common import display_name_source
    return display_name_source(value)


@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Allows accessing dictionary values using dot notation in templates.
    Usage: {{ my_dict|get_item:"my_key" }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None


@register.filter(name='file_exists')
def file_exists(field_file):
    """检查 FileField/ImageField 对应的物理文件是否存在。
    Usage: {% if profile.avatar|file_exists %}"""
    try:
        return bool(field_file) and bool(field_file.storage.exists(field_file.name))
    except Exception:
        return False


@register.simple_tag
def get_user_avatar(username):
    """
    获取用户头像URL
    Usage: {% get_user_avatar username as avatar_url %}
    """
    if not username:
        return None
    try:
        import os
        from User.models import User, UserProfile
        user = User.objects.get(username=username)
        profile = UserProfile.objects.get(user=user)
        # 数据库有记录但物理文件已被删除时，回退到默认头像，避免页面裂图
        if profile.avatar and os.path.exists(profile.avatar.path):
            return profile.avatar.url
    except Exception:
        pass
    return None


@register.filter(name='filter_is_admin')
def filter_is_admin(users_list):
    """统计管理员数量"""
    try:
        return sum(1 for item in users_list if (item.get('user') if isinstance(item, dict) else item.user).admin == 1)
    except Exception:
        return 0


@register.filter(name='subtract')
def subtract(value, arg):
    """减法过滤器"""
    try:
        return int(value) - int(arg)
    except (ValueError, TypeError):
        return 0
