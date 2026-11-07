"""三个界面的联网开关集成测试：全部 mock，不发起真实搜索。"""

from unittest import mock

import pytest
from django.test import override_settings
from django.urls import reverse

from Dashboard.ai_views import ChatAnswer
from Dashboard.search_service import SearchOutcome, SearchResult, SearchStatus


@pytest.mark.django_db
@override_settings(SEARCH_ENABLED=False, SEARCH_API_KEY='', TAVILY_API_KEY='')
def test_toggles_disabled_and_explained_without_key(auth_client):
    """未配置 Key 时，三个页面的开关都应置灰并说明未配置搜索服务。"""
    pages = [
        (reverse('Dashboard:language_chat'), 'webSearchToggle'),
        (reverse('Dashboard:language_visualization'), 'analysis-web-search'),
        (reverse('Dashboard:language_translation'), 'translationWebSearch'),
    ]
    for url, checkbox_id in pages:
        response = auth_client.get(url)
        html = response.content.decode()
        assert response.status_code == 200
        assert checkbox_id in html
        assert '未配置搜索服务' in html
        assert 'disabled' in html


@pytest.mark.django_db
def test_chat_passes_switch_state_to_backend(auth_client):
    """开关状态必须显式传给 chat()；关闭时 allow_search=False。"""
    with mock.patch('Dashboard.ai_views.chat') as chat_mock, \
            mock.patch('Dashboard.ai_views.search_is_available', return_value=False):
        chat_mock.return_value = ChatAnswer(text='回答', search_status='off')
        response = auth_client.post(
            reverse('Dashboard:language_chat'),
            {'language_text': '濒危语言是什么', 'web_search': '0'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    assert response.status_code == 200
    assert response.json()['search_status'] == 'off'
    assert chat_mock.call_args.kwargs['allow_search'] is False

    with mock.patch('Dashboard.ai_views.chat') as chat_mock, \
            mock.patch('Dashboard.ai_views.search_is_available', return_value=True):
        chat_mock.return_value = ChatAnswer(text='回答', search_status='not_used')
        response = auth_client.post(
            reverse('Dashboard:language_chat'),
            {'language_text': '最新进展', 'web_search': '1'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )
    assert chat_mock.call_args.kwargs['allow_search'] is True


@pytest.mark.django_db
def test_visualization_uses_switch_and_returns_sources(auth_client):
    result = SearchResult(title='来源', url='https://example.com/a', snippet='摘要')
    answer = ChatAnswer(
        text='解读结果',
        sources=[{'title': '来源', 'url': result.url, 'site': '', 'date': ''}],
        search_status='ok',
        search_notice='已联网检索，找到 1 条来源。',
        search_provider='tavily',
        search_used=True,
    )
    with mock.patch('Dashboard.ai_views.chat_with_web_search', return_value=answer) as chat_mock, \
            mock.patch('Dashboard.ai_views.search_is_available', return_value=True), \
            mock.patch('Dashboard.ai_views.search_web') as search_mock:
        response = auth_client.post(
            reverse('Dashboard:language_visualization'),
            {'language_text': '一段分析文本', 'web_search': '1'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload['search_status'] == 'ok'
    assert payload['sources'][0]['url'] == 'https://example.com/a'
    assert chat_mock.call_args.kwargs['allow_search'] is True
    search_mock.assert_not_called()  # 搜索由 chat 内部触发，视图不重复调用


@pytest.mark.django_db
@override_settings(SEARCH_ENABLED=False, SEARCH_API_KEY='', TAVILY_API_KEY='')
def test_translation_switch_off_never_searches_when_uncertain(auth_client):
    """模型标记不确定、开关关闭时：不搜索，如实告知未找到可靠译法。"""
    with mock.patch('Dashboard.ai_views.translate_once', return_value={
        'translation': '', 'uncertain': True, 'reason': '词义不确定',
    }), mock.patch('Dashboard.ai_views.search_web') as search_mock:
        response = auth_client.post(
            reverse('Dashboard:language_translation'),
            {'language_text': 'unknown text', 'web_search': '0'},
        )

    html = response.content.decode()
    assert response.status_code == 200
    assert '未找到可靠译法' in html
    search_mock.assert_not_called()


@pytest.mark.django_db
def test_translation_switch_on_searches_exactly_once_and_shows_reference(auth_client):
    result = SearchResult(title='参考页面', url='https://example.com/ref', snippet='译法')
    outcome = SearchOutcome(SearchStatus.OK, provider='tavily', results=[result])
    reference = {
        'found': True,
        'reference_translation': '参考译法内容',
        'explanation': '资料中存在对应译法',
        'results': [result],
    }
    with mock.patch('Dashboard.ai_views.translate_once', return_value={
        'translation': '', 'uncertain': True, 'reason': '词义不确定',
    }), mock.patch('Dashboard.ai_views.search_is_available', return_value=True), \
            mock.patch('Dashboard.ai_views.search_web', return_value=outcome) as search_mock, \
            mock.patch('Dashboard.ai_views.reference_translation_from_search', return_value=reference):
        response = auth_client.post(
            reverse('Dashboard:language_translation'),
            {'language_text': 'unknown text', 'web_search': '1'},
        )

    html = response.content.decode()
    assert response.status_code == 200
    assert '参考译法（来自网络，需核对来源）' in html
    assert '参考译法内容' in html
    assert 'https://example.com/ref' in html
    assert '联网结果可能不准确' in html
    assert search_mock.call_count == 1


@pytest.mark.django_db
def test_translation_switch_on_but_certain_does_not_search(auth_client):
    """模型标记有把握时，即使开关打开也不搜索。"""
    with mock.patch('Dashboard.ai_views.translate_once', return_value={
        'translation': '确定译法', 'uncertain': False, 'reason': '',
    }), mock.patch('Dashboard.ai_views.search_is_available', return_value=True), \
            mock.patch('Dashboard.ai_views.search_web') as search_mock:
        response = auth_client.post(
            reverse('Dashboard:language_translation'),
            {'language_text': 'some text', 'web_search': '1'},
        )

    assert response.status_code == 200
    assert '确定译法' in response.content.decode()
    search_mock.assert_not_called()


@pytest.mark.django_db
def test_translation_uncertain_without_key_reports_unavailable(auth_client):
    """开关打开但未配置服务时，不搜索，也不编造译法。"""
    with mock.patch('Dashboard.ai_views.translate_once', return_value={
        'translation': '', 'uncertain': True, 'reason': '',
    }), mock.patch('Dashboard.ai_views.search_is_available', return_value=False), \
            mock.patch('Dashboard.ai_views.search_web') as search_mock:
        response = auth_client.post(
            reverse('Dashboard:language_translation'),
            {'language_text': 'unknown text', 'web_search': '1'},
        )

    html = response.content.decode()
    assert response.status_code == 200
    assert '未找到可靠译法' in html
    assert '未配置搜索服务' in html
    search_mock.assert_not_called()
