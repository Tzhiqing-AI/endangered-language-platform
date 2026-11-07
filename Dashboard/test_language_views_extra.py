"""语言管理视图补充测试：列表/详情/检索/分页、权限、名称兜底与 CSV 导入。"""
from __future__ import annotations

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from Dashboard import language_views
from Dashboard.common import display_language_name, display_name_source
from Dashboard.models import (
    Country, Language, LanguageCountry, LanguageLocation,
)
from Community.models import Post

FORM_DATA = {
    'language_name': '新语言',
    'level_of_endangerment': 'unsafe',
    'legacy_endangerment_level': 'unsafe',
    'number_of_speakers': '12',
    'speaker_trends': 'Stable',
    'intergenerational_transmission': 'some',
    'language_family': '测试语系',
    'country': '测试国',
    'additional_notes': '',
}


def _make_language(name='测试语言', **kwargs):
    data = {
        'language_name': name,
        'level_of_endangerment': 'unsafe',
        'legacy_endangerment_level': 'unsafe',
        'number_of_speakers': 100,
        'language_family': '语系A',
        'country': '国家A',
        **kwargs,
    }
    return Language.objects.create(**data)


@pytest.fixture
def quiet_side_effects(monkeypatch):
    """列表页使用真实风险产物会很慢，测试只关心视图分支。"""
    monkeypatch.setattr(language_views, 'get_alerts', lambda **kwargs: [])
    monkeypatch.setattr(language_views, 'get_risk_stats', lambda: {
        'total': 0, 'average_score': 0, 'high_count': 0,
        'calibration_status': 'missing', 'calibration_applied': False,
    })
    monkeypatch.setattr(language_views, 'get_chart_data', lambda: {})
    monkeypatch.setattr(language_views, 'get_risk_profile', lambda code_id: None)
    monkeypatch.setattr(language_views, 'get_service_meta', lambda: {})


def test_list_pagination_sort_and_invalid_params(auth_client, quiet_side_effects):
    for index in range(3):
        _make_language(f'批量语言{index}', number_of_speakers=index)

    response = auth_client.get(reverse('Dashboard:language_list'), {
        'sort': 'not_allowed', 'order': 'desc', 'page': 'bad',
    })
    assert response.status_code == 200
    assert response.context['page_obj'].number == 1
    assert response.context['sort_field'] == 'language_id'

    response = auth_client.get(reverse('Dashboard:language_list'), {
        'sort': 'number_of_speakers', 'order': 'desc', 'page': 999,
    })
    assert response.status_code == 200
    assert response.context['page_obj'].number == 1
    assert response.context['query_prefix'] == 'sort=number_of_speakers&order=desc&'


@pytest.mark.parametrize('search_type,expected', [
    ('name', 0), ('family', 1), ('country', 1), ('all', 1), ('bad', 1),
])
def test_search_types(auth_client, quiet_side_effects, search_type, expected):
    _make_language('检索目标', language_family='唯一语系', country='唯一国家')

    response = auth_client.get(reverse('Dashboard:language_search'), {
        'q': '唯一', 'search_type': search_type,
    })
    assert response.status_code == 200
    assert response.context['search_type'] == (
        search_type if search_type in ('name', 'family', 'country', 'all') else 'all'
    )
    assert response.context['result_count'] == expected


def test_search_truncates_long_query(auth_client):
    long_query = 'x' * 500
    response = auth_client.get(reverse('Dashboard:language_search'), {'q': long_query})
    assert response.status_code == 200
    assert len(response.context['search_query']) <= 100
    assert response.context['search_query']


def test_list_and_search_require_login(django_client):
    assert django_client.get(reverse('Dashboard:language_list')).status_code == 302
    assert django_client.get(reverse('Dashboard:language_search')).status_code == 302


def test_detail_renders_with_related_data(admin_client, quiet_side_effects,
                                          test_user, test_category):
    lang = _make_language('详情语言', language_family='同族')
    LanguageLocation.objects.create(
        language_name=lang.language_name, longitude='120.1', latitude='30.2',
    )
    country = Country.objects.create(name='详情国')
    LanguageCountry.objects.create(id=1, language_id=lang.language_id,
                                   country_id=country.id)
    Post.objects.create(
        author=test_user, category=test_category, title='相关帖子',
        content='详情语言', status='published', related_language=lang,
    )

    response = admin_client.get(reverse('Dashboard:language_detail',
                                        args=[lang.language_id]))
    assert response.status_code == 200
    assert response.context['location_valid'] is True
    assert list(response.context['countries']) == ['详情国']
    assert response.context['post_count'] == 1


def test_detail_404(admin_client, quiet_side_effects):
    response = admin_client.get(reverse('Dashboard:language_detail', args=[999999]))
    assert response.status_code == 404


def test_non_admin_cannot_write(auth_client):
    response = auth_client.post(reverse('Dashboard:language_list'),
                                {'addBtn': '1', **FORM_DATA})
    assert response.status_code == 302
    assert not Language.objects.filter(language_name='新语言').exists()


def test_anonymous_cannot_write(django_client, db):
    response = django_client.post(reverse('Dashboard:language_list'),
                                  {'addBtn': '1', **FORM_DATA})
    assert response.status_code == 302
    assert not Language.objects.filter(language_name='新语言').exists()


def test_admin_add_invalid_duplicate_and_success(admin_client):
    response = admin_client.post(reverse('Dashboard:language_list'),
                                 {'addBtn': '1',
                                  **{**FORM_DATA, 'number_of_speakers': 'not-a-number'}})
    assert response.status_code == 302
    assert Language.objects.count() == 0

    _make_language('新语言')
    response = admin_client.post(reverse('Dashboard:language_list'),
                                 {'addBtn': '1', **FORM_DATA})
    assert response.status_code == 302
    assert Language.objects.count() == 1

    response = admin_client.post(reverse('Dashboard:language_list'),
                                 {'addBtn': '1', **FORM_DATA, 'language_name': '全新语言'})
    assert response.status_code == 302
    assert Language.objects.filter(language_name='全新语言').exists()


def test_admin_update_branches(admin_client):
    lang = _make_language('待更新')
    other = _make_language('已占用')

    for payload in ({}, {'language_id': 'abc'}, {'language_id': '999999'}):
        response = admin_client.post(reverse('Dashboard:language_list'),
                                     {'updateBtn': '1', **payload})
        assert response.status_code == 302

    response = admin_client.post(reverse('Dashboard:language_list'), {
        'updateBtn': '1', 'language_id': lang.language_id,
        **FORM_DATA, 'language_name': '已占用',
    })
    assert Language.objects.get(language_id=lang.language_id).language_name == '待更新'

    response = admin_client.post(reverse('Dashboard:language_list'), {
        'updateBtn': '1', 'language_id': lang.language_id,
        **FORM_DATA, 'language_name': '更新成功',
    })
    assert response.status_code == 302
    assert Language.objects.get(language_id=lang.language_id).language_name == '更新成功'
    assert other.language_name == '已占用'


def test_admin_delete_branches(admin_client, monkeypatch):
    monkeypatch.setattr(language_views, '_archive_deleted_language', lambda obj: None)
    lang = _make_language('待删除')

    for payload in ({}, {'language_id': 'abc'}, {'language_id': '999999'}):
        response = admin_client.post(reverse('Dashboard:language_list'),
                                     {'deleteBtn': '1', **payload})
        assert response.status_code == 302

    response = admin_client.post(reverse('Dashboard:language_list'), {
        'deleteBtn': '1', 'language_id': lang.language_id,
    })
    assert response.status_code == 302
    assert not Language.objects.filter(language_id=lang.language_id).exists()


def test_unknown_action_redirects(admin_client):
    response = admin_client.post(reverse('Dashboard:language_list'), {'other': '1'})
    assert response.status_code == 302


def test_csv_import_validation_and_rows(admin_client):
    _make_language('已存在语言')

    response = admin_client.post(reverse('Dashboard:language_list'), {'fileBtn': '1'})
    assert response.status_code == 302

    csv_content = (
        'name,level,speakers,trend,transmission,family,country,notes\n'
        'CSV语言1,unsafe,10,Stable,some,语系,国家,备注\n'
        'CSV语言1,unsafe,20,Stable,some,语系,国家,重复\n'
        '已存在语言,unsafe,30,Stable,some,语系,国家,已存在\n'
        ',unsafe,40,Stable,some,语系,国家,空名\n'
        'CSV坏行,unsafe,abc\n'
        '\n'
    )
    upload = SimpleUploadedFile('languages.csv', csv_content.encode('utf-8'),
                                content_type='text/csv')
    response = admin_client.post(reverse('Dashboard:language_list'),
                                 {'fileBtn': '1', 'myFile': upload})
    assert response.status_code == 302

    result = admin_client.session['csv_import_result']
    assert result['success'] == 1
    assert result['failed'] >= 3
    assert Language.objects.filter(language_name='CSV语言1').exists()


def test_csv_import_rejects_bad_extension(admin_client):
    upload = SimpleUploadedFile('languages.txt', b'a,b,c',
                                content_type='text/plain')
    response = admin_client.post(reverse('Dashboard:language_list'),
                                 {'fileBtn': '1', 'myFile': upload})
    assert response.status_code == 302
    assert not Language.objects.filter(language_name='CSV语言1').exists()


def test_admin_user_management_page(admin_client):
    response = admin_client.get(reverse('Dashboard:user_management'),
                                {'role': 'expert', 'page': 2})
    assert response.status_code == 200
    assert response.context['current_role'] == 'expert'
    assert response.context['expert_count'] >= 0

    response = admin_client.get(reverse('Dashboard:user_management'),
                                {'role': 'bad'})
    assert response.context['current_role'] == 'all'


def test_normal_user_cannot_view_user_management(auth_client):
    response = auth_client.get(reverse('Dashboard:user_management'))
    assert response.status_code == 302


def test_admin_dashboard_permissions(admin_client, auth_client):
    assert admin_client.get(reverse('Dashboard:admin_dashboard')).status_code == 200
    assert auth_client.get(reverse('Dashboard:admin_dashboard')).status_code == 302


def test_user_management_post_branches(admin_client, test_user, test_admin):
    from User.models import User

    def post(**data):
        return admin_client.post(reverse('Dashboard:user_management_post'), data)

    assert post(action='set_expert')['Content-Type'].startswith('application/json')
    assert post(action='set_expert', username='missing').json()['success'] is False
    assert post(action='delete', username=test_admin.username).json()['success'] is False

    other_admin = User.objects.create(username='other_admin', password='md5$test',
                                      name='other_admin', admin=1)
    response = post(action='set_expert', username=other_admin.username)
    assert response.status_code == 403

    response = post(action='set_admin', username=test_user.username)
    assert response.status_code == 403

    assert post(action='set_expert', username=test_user.username).json()['success'] is True
    test_user.refresh_from_db()
    assert test_user.expert == 1

    assert post(action='unset_expert', username=test_user.username).json()['success'] is True
    test_user.refresh_from_db()
    assert test_user.expert == 0

    assert post(action='unknown', username=test_user.username).json()['success'] is False

    assert post(action='delete', username=test_user.username).json()['success'] is True
    assert not User.objects.filter(username=test_user.username).exists()


def test_non_admin_user_management_post_returns_403(auth_client):
    response = auth_client.post(reverse('Dashboard:user_management_post'),
                                {'action': 'set_expert', 'username': 'someone'})
    assert response.status_code == 403


def test_display_name_fallbacks():
    assert display_language_name({'language_id': 3, 'language_name': ''}) == '未命名语言 #3'
    assert display_language_name({'code_id': 5, 'language_name': None}) == '未命名语言 #5'
    assert display_language_name({'language_name': '正常名称'}) == '正常名称'
    assert display_name_source('manual:unresolved:abc') == '待核实'
    assert display_name_source('manual:verified') == 'manual:verified'
    assert display_name_source(None) == ''
