"""
创建社区论坛的默认标签和分类
运行：python manage.py shell < init_community_data.py
"""
from Community.models import Category, Tag

# 创建默认分类
categories_data = [
    {'name': '语言讨论', 'description': '濒危语言相关讨论', 'icon': 'bi-chat-quote', 'sort_order': 1},
    {'name': '语言保护', 'description': '语言保护经验分享', 'icon': 'bi-shield-check', 'sort_order': 2},
    {'name': '学术研究', 'description': '语言学研究与论文', 'icon': 'bi-book', 'sort_order': 3},
    {'name': '文化交流', 'description': '语言与文化活动', 'icon': 'bi-globe-asia-australia', 'sort_order': 4},
    {'name': '资源分享', 'description': '学习资料与工具', 'icon': 'bi-share', 'sort_order': 5},
    {'name': '提问求助', 'description': '语言学习提问', 'icon': 'bi-question-circle', 'sort_order': 6},
]

print('=== 创建分类 ===')
for cat_data in categories_data:
    cat, created = Category.objects.get_or_create(
        name=cat_data['name'],
        defaults=cat_data
    )
    if created:
        print(f'  + 新建分类: {cat.name}')
    else:
        print(f'  = 已存在: {cat.name}')

# 创建默认标签
tags_data = [
    {'name': '濒危语言', 'color': '#ef4444'},
    {'name': '方言保护', 'color': '#f97316'},
    {'name': '少数民族语言', 'color': '#eab308'},
    {'name': '吴语', 'color': '#22c55e'},
    {'name': '粤语', 'color': '#10b981'},
    {'name': '客家话', 'color': '#14b8a6'},
    {'name': '藏语', 'color': '#06b6d4'},
    {'name': '维吾尔语', 'color': '#0ea5e9'},
    {'name': '蒙古语', 'color': '#3b82f6'},
    {'name': '古汉语', 'color': '#6366f1'},
    {'name': '语言学', 'color': '#8b5cf6'},
    {'name': '语音学', 'color': '#a855f7'},
    {'name': '语法研究', 'color': '#d946ef'},
    {'name': '词汇学', 'color': '#ec4899'},
    {'name': '语言记录', 'color': '#f43f5e'},
    {'name': '口述历史', 'color': '#fb7185'},
    {'name': '文化传承', 'color': '#84cc16'},
    {'name': '教育传承', 'color': '#a3e635'},
    {'name': 'AI与语言', 'color': '#64748b'},
    {'name': '语言复兴', 'color': '#dc2626'},
    {'name': '新文字', 'color': '#0891b2'},
    {'name': '国际母语日', 'color': '#7c3aed'},
    {'name': '联合国教科文', 'color': '#0284c7'},
    {'name': '学术会议', 'color': '#475569'},
    {'name': '田野调查', 'color': '#059669'},
    {'name': '语言地图', 'color': '#be185d'},
    {'name': '濒危等级', 'color': '#b91c1c'},
    {'name': '代际传承', 'color': '#c2410c'},
    {'name': '母语教育', 'color': '#65a30d'},
    {'name': '数字典藏', 'color': '#0d9488'},
]

print('\n=== 创建标签 ===')
for tag_data in tags_data:
    tag, created = Tag.objects.get_or_create(
        name=tag_data['name'],
        defaults=tag_data
    )
    if created:
        print(f'  + 新建标签: {tag.name}')
    else:
        print(f'  = 已存在: {tag.name}')

print(f'\n=== 统计 ===')
print(f'分类总数: {Category.objects.count()}')
print(f'标签总数: {Tag.objects.count()}')
print('完成！')
