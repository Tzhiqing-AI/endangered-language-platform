from django.template.defaultfilters import register

@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Allows accessing dictionary values using dot notation in templates.
    Usage: {{ my_dict|get_item:"my_key" }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None


@register.simple_tag
def get_user_avatar(username):
    """
    获取用户头像URL
    Usage: {% get_user_avatar username as avatar_url %}
    """
    if not username:
        return None
    try:
        from User.models import User, UserProfile
        user = User.objects.get(username=username)
        profile = UserProfile.objects.get(user=user)
        if profile.avatar:
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
