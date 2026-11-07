"""离线降级相关测试：失败分类、离线兜底、图表不受影响、不可达 API 模拟。"""

from unittest import mock

import pytest
from django.urls import reverse

from Dashboard.ai_views import AIUnavailable, classify_ai_error, offline_chat_answer
from Dashboard.models import Glossary


class _DummyError(Exception):
    def __init__(self, message, status_code=None):
        self.status_code = status_code
        super().__init__(message)


def test_classify_ai_error_by_message_and_status():
    assert classify_ai_error(_DummyError('Request timed out.')) == 'timeout'
    assert classify_ai_error(_DummyError('Connection error.')) == 'network'
    assert classify_ai_error(_DummyError('The api_key client option must be set')) == 'disabled'
    assert classify_ai_error(_DummyError('unauthorized', status_code=401)) == 'auth'
    assert classify_ai_error(_DummyError('forbidden', status_code=403)) == 'auth'
    assert classify_ai_error(_DummyError('too many requests', status_code=429)) == 'rate_limit'
    assert classify_ai_error(_DummyError('bad gateway', status_code=502)) == 'upstream'
    assert classify_ai_error(_DummyError('boom')) == 'unknown'


def test_offline_chat_answer_only_matches_preset_examples():
    assert 'UNESCO' in offline_chat_answer('什么是濒危语言？')
    assert '满语' in offline_chat_answer('满语的现状如何')
    assert offline_chat_answer('今天天气怎么样') == ''


@pytest.mark.django_db
def test_chat_network_failure_returns_labeled_offline_example(auth_client):
    with mock.patch('Dashboard.ai_views.chat', side_effect=AIUnavailable('network')):
        response = auth_client.post(
            reverse('Dashboard:language_chat'),
            {'language_text': '什么是濒危语言'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'success'
    assert data['offline'] is True
    assert data['failure_code'] == 'network'
    assert '离线示例' in data['answer']
    assert '非实时' in data['answer']
    assert '不是模型实时生成' in data['answer']
    assert data['sources'] == []


@pytest.mark.django_db
def test_chat_network_failure_without_preset_does_not_fake_answer(auth_client):
    with mock.patch('Dashboard.ai_views.chat', side_effect=AIUnavailable('network')):
        response = auth_client.post(
            reverse('Dashboard:language_chat'),
            {'language_text': '今天天气怎么样'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    data = response.json()
    assert data['offline'] is True
    assert '离线提示' in data['answer']
    assert '未生成回答' in data['answer']
    assert '无法连接' in data['answer']


@pytest.mark.django_db
def test_visualization_failure_keeps_chart_usable(auth_client):
    with mock.patch(
        'Dashboard.ai_views.chat_with_web_search',
        side_effect=AIUnavailable('timeout'),
    ):
        response = auth_client.post(
            reverse('Dashboard:language_visualization'),
            {'language_text': '测试文本'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'error'
    assert '超时' in data['message']
    assert '图表数据不受影响' in data['message']
    assert data['failure_code'] == 'timeout'


@pytest.mark.django_db
def test_translation_failure_uses_local_glossary_with_offline_label(auth_client):
    Glossary.objects.create(term='满语', translation='Manchu', is_approved=True)
    with mock.patch('Dashboard.ai_views.translate_once', side_effect=AIUnavailable('network')):
        response = auth_client.post(
            reverse('Dashboard:language_translation'),
            {'language_text': '满语'},
        )

    assert response.status_code == 200
    assert '离线参考' in response.content.decode('utf-8')
    assert '非实时' in response.content.decode('utf-8')
    assert 'Manchu' in response.content.decode('utf-8')


@pytest.mark.django_db
def test_translation_failure_without_local_data_reports_no_translation(auth_client):
    with mock.patch('Dashboard.ai_views.translate_once', side_effect=AIUnavailable('disabled')):
        response = auth_client.post(
            reverse('Dashboard:language_translation'),
            {'language_text': '未知文本'},
        )

    html = response.content.decode('utf-8')
    assert '未配置' in html
    assert '未生成翻译' in html


@pytest.mark.django_db
def test_ai_pages_load_without_calling_external_services(auth_client):
    pages = (
        'Dashboard:language_chat',
        'Dashboard:language_visualization',
        'Dashboard:language_translation',
    )
    with mock.patch(
        'Dashboard.common.get_openai_client',
        side_effect=AssertionError('页面加载不应调用外部 AI 服务'),
    ):
        for name in pages:
            response = auth_client.get(reverse(name))
            assert response.status_code == 200
            assert len(response.content) > 0


@pytest.mark.django_db
def test_unreachable_api_base_returns_visible_offline_notice(auth_client):
    from openai import OpenAI

    unreachable = OpenAI(
        api_key='test-key',
        base_url='http://127.0.0.1:1/v1',
        timeout=1,
        max_retries=0,
    )
    with mock.patch('Dashboard.common.get_openai_client', return_value=unreachable):
        response = auth_client.post(
            reverse('Dashboard:language_chat'),
            {'language_text': '今天天气怎么样'},
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

    assert response.status_code == 200
    data = response.json()
    assert data['offline'] is True
    # 不可达地址在不同系统上可能表现为连接失败或连接超时，两者都必须是可见提示
    assert data['failure_code'] in ('network', 'timeout')
    assert '本次未生成回答' in data['answer']
    assert data['answer'].strip()
