"""语言样本库视图"""
import re
from collections import Counter

from django.core.paginator import Paginator
from django.db.models import Count
from django.shortcuts import render

from Dashboard.common import get_user_context, display_language_name, pagination_window
from Dashboard.models import Language, LanguageSample

SAMPLE_TYPES = {
    'V': '视频',
    'A': '音频',
    'L': '词汇',
    'G': '语法',
    'D': '词典',
    'I': '图片',
}


def sample_media(sample):
    """从缩略图 HTML / YouTube embed 中提取可展示的图片和播放链接"""
    thumbnail = ''
    match = re.search(r'src="([^"]+)"', sample.thumbnail_img or '')
    if match:
        thumbnail = match.group(1)
    video_id = ''
    match = re.search(r'youtube\.com/embed/([A-Za-z0-9_-]+)',
                      sample.youtube_video_embed or '')
    if match:
        video_id = match.group(1)
        if not thumbnail:
            thumbnail = f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
    play_url = f'https://www.youtube.com/watch?v={video_id}' if video_id else (sample.url or '')
    return thumbnail, play_url


def language_samples(request):
    """语言样本库：声音、视频、词汇、语法、词典和图片"""
    user, name, _ = get_user_context(request)

    sample_type = request.GET.get('type', '').strip().upper()
    query = request.GET.get('q', '').strip()

    base = LanguageSample.objects.filter(removed=False).exclude(title='')
    type_counts = Counter(dict(
        base.values('sample_type').annotate(count=Count('id'))
        .values_list('sample_type', 'count')
    ))

    qs = base
    if sample_type in SAMPLE_TYPES:
        qs = qs.filter(sample_type=sample_type)
    if query:
        qs = qs.filter(title__icontains=query)
    qs = qs.order_by('-featured', '-view_count', '-id')

    all_total = base.count()
    total = qs.count()
    sample_type_stats = [
        {'key': key, 'label': label, 'count': type_counts.get(key, 0)}
        for key, label in SAMPLE_TYPES.items()
    ]
    paginator = Paginator(qs, 24)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    language_names = dict(Language.objects.values_list('language_id', 'language_name'))
    for sample in page_obj.object_list:
        sample.language_name = display_language_name({
            'language_id': sample.code_id,
            'language_name': language_names.get(sample.code_id),
        })
        sample.thumbnail, sample.play_url = sample_media(sample)
        sample.type_label = SAMPLE_TYPES.get(sample.sample_type, '其他')

    context = {
        'page_obj': page_obj,
        'page_numbers': pagination_window(page_obj.number, paginator.num_pages),
        'total': total,
        'all_total': all_total,
        'sample_type': sample_type,
        'query': query,
        'type_counts': type_counts,
        'sample_type_stats': sample_type_stats,
        'sample_types': SAMPLE_TYPES,
        'name': name,
        'username': user.username if user else '',
        'is_admin': False,
    }
    return render(request, 'language_samples.html', context)
