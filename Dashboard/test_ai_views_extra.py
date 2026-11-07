"""AI 视图补充测试：全部使用 mock，不访问任何外部 API 或搜索服务。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from django.core.cache import cache
from django.urls import reverse

from Dashboard import ai_views
from Dashboard.models import (
    Glossary, Language, LanguageLocation, LanguageLocationDetail, TranslationMemory,
)
from Dashboard.search_service import SearchOutcome, SearchResult, SearchStatus
from User.models import ChatHistory


@pytest.fixture(autouse=True)
def _clear_cache():
    cache.clear()
    yield
    cache.clear()


def _completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(call_id='call-1', name='web-search', arguments='{"query":"濒危语言"}'):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(
        name=name, arguments=arguments))


def _outcome(results=None, status=SearchStatus.OK):
    return SearchOutcome(
        status=status,
        provider='tavily',
        results=results or [],
        message='',
    )


def _results():
    return [
        SearchResult(title='语言保护', url='https://example.com/a', snippet='摘要A'),
        SearchResult(title='濒危语言', url='https://example.com/b', snippet='摘要B'),
    ]


def _enable_ai(monkeypatch, module=ai_views):
    import Dashboard.common as common
    monkeypatch.setattr(common, 'OPENAI_API_KEY', 'test-key')


def test_chat_tool_loop_runs_search_once(monkeypatch):
    _enable_ai(monkeypatch)
    monkeypatch.setattr(ai_views, 'search_is_available', lambda: True)
    calls = []

    def fake_search(query, top_k=5):
        calls.append((query, top_k))
        return _outcome(_results())

    monkeypatch.setattr(ai_views, 'execute_web_search', fake_search)
    completions = [
        _completion(tool_calls=[_tool_call(arguments='{"query":"濒危语言","top_k":2}')]),
        _completion(content='根据资料，濒危语言需要保护 https://example.com/a'),
    ]
    monkeypatch.setattr(ai_views, '_create_completion',
                        lambda messages, tools=None: completions.pop(0))

    answer = ai_views.chat_with_web_search('问题', [])

    assert calls == [('濒危语言', 2)]
    assert answer.search_used is True
    assert answer.search_status == 'ok'
    assert len(answer.sources) == 2
    assert 'https://example.com/a' in answer.text


def test_chat_invalid_tool_arguments_and_second_call_has_no_tools(monkeypatch):
    _enable_ai(monkeypatch)
    monkeypatch.setattr(ai_views, 'search_is_available', lambda: True)
    seen = []

    def fake_search(query, top_k=5):
        seen.append((query, top_k))
        return _outcome([], status=SearchStatus.EMPTY)

    monkeypatch.setattr(ai_views, 'execute_web_search', fake_search)
    completions = [_completion(tool_calls=[_tool_call(arguments='not-json')]),
                   _completion(content='普通回答')]
    calls = []

    def fake_completion(messages, tools=None):
        calls.append(tools)
        return completions.pop(0)

    monkeypatch.setattr(ai_views, '_create_completion', fake_completion)

    answer = ai_views.chat_with_web_search('问题', [])

    assert seen == [('', 5)]
    assert calls[0] is not None and calls[1] is None  # 第二轮不暴露工具
    assert answer.search_status == 'empty'


def test_chat_ignores_duplicate_tool_calls(monkeypatch):
    _enable_ai(monkeypatch)
    monkeypatch.setattr(ai_views, 'search_is_available', lambda: True)
    called = []
    monkeypatch.setattr(ai_views, 'execute_web_search',
                        lambda query, top_k=5: called.append(query) or _outcome(_results()))
    completions = [_completion(tool_calls=[_tool_call('1'), _tool_call('2')]),
                   _completion(content='回答')]
    captured = []

    def fake_completion(messages, tools=None):
        captured.append(list(messages))
        return completions.pop(0)

    monkeypatch.setattr(ai_views, '_create_completion', fake_completion)

    ai_views.chat_with_web_search('问题', [])

    assert called == ['濒危语言']
    tool_messages = [m for m in captured[1] if isinstance(m, dict) and m.get('role') == 'tool']
    assert len(tool_messages) == 2
    assert '最多执行一次' in tool_messages[1]['content']


def test_chat_without_tool_calls_and_switch_off(monkeypatch):
    _enable_ai(monkeypatch)
    monkeypatch.setattr(ai_views, 'search_is_available', lambda: True)
    calls = []
    monkeypatch.setattr(ai_views, '_create_completion',
                        lambda messages, tools=None: calls.append(tools)
                        or _completion(content='直接回答'))

    answer = ai_views.chat_with_web_search('问题', [], allow_search=False)

    assert calls == [None]
    assert answer.search_status == 'off'
    assert answer.search_used is False


def test_chat_search_service_disabled(monkeypatch):
    _enable_ai(monkeypatch)
    monkeypatch.setattr(ai_views, 'search_is_available', lambda: False)
    monkeypatch.setattr(ai_views, '_create_completion',
                        lambda messages, tools=None: _completion(content='回答'))

    answer = ai_views.chat_with_web_search('问题', [], allow_search=True)

    assert answer.search_status == 'disabled'
    assert answer.search_used is False


def test_chat_exception_is_wrapped(monkeypatch):
    _enable_ai(monkeypatch)
    monkeypatch.setattr(ai_views, 'search_is_available', lambda: True)

    def boom(messages, tools=None):
        raise RuntimeError('boom')

    monkeypatch.setattr(ai_views, '_create_completion', boom)

    with pytest.raises(ai_views.AIUnavailable) as exc:
        ai_views.chat_with_web_search('问题', [])
    assert exc.value.code == 'unknown'


def test_chat_requires_api_key(monkeypatch):
    import Dashboard.common as common
    monkeypatch.setattr(common, 'OPENAI_API_KEY', '')
    with pytest.raises(ai_views.AIUnavailable) as exc:
        ai_views.chat_with_web_search('问题', [])
    assert exc.value.code == 'disabled'


def test_create_completion_and_retry(monkeypatch):
    import Dashboard.common as common

    monkeypatch.setattr(common, 'OPENAI_API_KEY', 'test-key')
    fake_client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kwargs: _completion('ok')),
    ))
    monkeypatch.setattr(common, 'get_openai_client', lambda: fake_client)

    assert ai_views._create_completion([{'role': 'user', 'content': 'x'}]) is not None

    monkeypatch.setattr(common, 'OPENAI_API_KEY', '')
    with pytest.raises(ai_views.AIUnavailable):
        ai_views._create_completion([])


def test_chat_rules_for_search_switches(monkeypatch):
    captured = {}

    def fake_chat(query, history, system_prompt=None, *, allow_search=True):
        captured['prompt'] = system_prompt
        return ai_views.ChatAnswer(text='ok')

    monkeypatch.setattr(ai_views, 'chat_with_web_search', fake_chat)

    monkeypatch.setattr(ai_views, 'search_is_available', lambda: True)
    ai_views.chat('q', [], allow_search=True)
    assert '已打开联网搜索开关' in captured['prompt']

    monkeypatch.setattr(ai_views, 'search_is_available', lambda: False)
    ai_views.chat('q', [], allow_search=True)
    assert '未配置搜索服务' in captured['prompt']

    ai_views.chat('q', [], allow_search=False)
    assert '未启用联网搜索' in captured['prompt']


@pytest.mark.parametrize('raw,expected', [
    ('```json\n{"translation":"你好","uncertain":false,"reason":""}\n```',
     ('你好', False, '')),
    ('{"translation":"你好"}', ('你好', True, '')),
    ('{"translation":"","uncertain":false}', ('', True, '')),
    ('not-json', ('', True, '模型未返回结构化结果')),
    ('{"translation":"你好","uncertain":"no"}', ('你好', True, '')),
])
def test_translate_once(monkeypatch, raw, expected):
    monkeypatch.setattr(ai_views, '_create_completion',
                        lambda messages, tools=None: _completion(raw))

    result = ai_views.translate_once('原文')

    assert (result['translation'], result['uncertain'], result['reason']) == expected


def test_reference_translation_from_search(monkeypatch):
    results = _results()
    monkeypatch.setattr(ai_views, '_create_completion', lambda messages, tools=None:
                        _completion(json.dumps({
                            'reference_translation': '参考译文',
                            'found': True,
                            'explanation': '说明',
                            'used_sources': [2, 99, 2],
                        })))

    reference = ai_views.reference_translation_from_search('原文', _outcome(results))

    assert reference['found'] is True
    assert reference['reference_translation'] == '参考译文'
    assert reference['results'] == [results[1]]

    monkeypatch.setattr(ai_views, '_create_completion',
                        lambda messages, tools=None: _completion('oops'))
    fallback = ai_views.reference_translation_from_search('原文', _outcome(results))
    assert fallback['found'] is False
    assert fallback['results'] == []

    monkeypatch.setattr(ai_views, '_create_completion', lambda messages, tools=None:
                        _completion(json.dumps({'reference_translation': '',
                                                'found': True})))
    no_sources = ai_views.reference_translation_from_search('原文', _outcome(results))
    assert no_sources['found'] is False
    assert no_sources['results'] == results


def test_format_reference_and_search_query():
    reference = {'reference_translation': '译文', 'explanation': '说明',
                 'results': _results()}
    text = ai_views._format_reference_translation(reference)
    assert '参考译法' in text and '核对来源' in text
    assert ai_views.build_translation_search_query('  a   b  ').endswith('中文翻译 释义 参考译法')
    assert len(ai_views.build_translation_search_query('x' * 300)) < 240


def test_offline_chat_and_translation_fallback(db, test_user):
    assert '濒危语言' in ai_views.offline_chat_answer('什么是濒危语言')
    assert ai_views.offline_chat_answer('完全无关的问题') == ''

    failure = ai_views.AIUnavailable('disabled')
    assert '本次未生成翻译' in ai_views.offline_translation_fallback('abc', failure)

    Glossary.objects.create(term='词', translation='word', username=test_user.username)
    TranslationMemory.objects.create(
        source_text='abc 文本', translated_text='ABC 译文',
        username=test_user.username,
    )
    glossary = Glossary.objects.get(term='词')
    text = ai_views.offline_translation_fallback('abc', failure, [glossary])
    assert '词 → word' in text
    assert 'ABC 译文' in text


def test_chat_context_and_history_building(db, test_user):
    ChatHistory.objects.create(username=test_user, question='问1', answer='答1')
    ChatHistory.objects.create(username=test_user, question='', answer='空问题')
    ChatHistory.objects.create(username=test_user, question='问2', answer='答2')

    questions, answers, times, user_obj = ai_views.get_user_chat_context(test_user.username)
    assert sorted(questions) == ['问1', '问2']
    assert sorted(answers) == ['答1', '答2']
    assert len(times) == 2
    assert user_obj == test_user

    empty = ai_views.get_user_chat_context('no-such-user')
    assert empty[0] == [] and empty[3] is None
    assert ai_views.get_user_chat_context('')[0] == []

    history = ai_views.build_chat_history(['q'], ['a'])
    assert history == [{'role': 'user', 'content': 'q'},
                       {'role': 'assistant', 'content': 'a'}]


def test_get_chat_history_view(auth_client, test_user):
    ChatHistory.objects.create(username=test_user, question='问题', answer='回答')
    url = reverse('Dashboard:language_chat_history')

    assert auth_client.get(url).json()['status'] == 'error'
    assert auth_client.get(url, {'before': 'bad'}).json()['status'] == 'error'
    response = auth_client.get(url, {'before': '2020-01-01T00:00:00'})
    assert response.json()['status'] == 'success'
    assert response.json()['messages'][0]['question'] == '问题'


def test_history_answer_text_appends_sources():
    address = ai_views.ChatAnswer(text='回答', sources=[
        {'title': '来源', 'url': 'https://example.com/x'},
    ])
    text = ai_views._history_answer_text(address)
    assert '参考来源' in text and 'https://example.com/x' in text

    address.text = '回答 https://example.com/x'
    assert '参考来源' not in ai_views._history_answer_text(address)
    assert ai_views._history_answer_text(ai_views.ChatAnswer(text='纯文本')) == '纯文本'


def test_map_parsers_and_filters():
    assert ai_views._parse_coordinates('30.2,120.1;31,121') == (120.1, 30.2)
    assert ai_views._parse_coordinates('bad') == (None, None)
    assert ai_views._parse_coordinates('a,b') == (None, None)
    assert ai_views._parse_coordinates('') == (None, None)

    assert ai_views._parse_bbox('120,30,121,31') == (120.0, 30.0, 121.0, 31.0)
    for bad in ('1,2,3', 'a,b,c,d', '200,0,201,1', '0,100,1,101', '2,0,1,1'):
        with pytest.raises(ValueError):
            ai_views._parse_bbox(bad)

    assert ai_views._resolve_level('') is None
    assert ai_views._resolve_level('endangered') == ai_views.LEGACY_LEVEL_LABELS['endangered']
    assert ai_views._resolve_level('unknown') == '未分类'
    assert ai_views._resolve_level('未分类') == '未分类'
    with pytest.raises(ValueError):
        ai_views._resolve_level('bad-level')

    points = [
        {'longitude': 120.0, 'latitude': 30.0, 'endangerment': '濒危'},
        {'longitude': 130.0, 'latitude': 40.0, 'endangerment': '危险'},
    ]
    assert ai_views._filter_points(points) == points
    assert ai_views._filter_points(points, bbox=(119, 29, 121, 31)) == [points[0]]
    assert ai_views._filter_points(points, level='危险') == [points[1]]


def test_map_snapshot_and_json_view(monkeypatch, db):
    lang = Language.objects.create(
        language_name='地图语言', legacy_endangerment_level='endangered',
        level_of_endangerment='endangered', number_of_speakers=50,
        country='地图国', language_family='地图语系',
    )
    LanguageLocationDetail.objects.create(
        id=1, code_id=lang.language_id, coordinates='120.1,30.2', preferred=True,
    )
    monkeypatch.setattr('Dashboard.risk_service.get_risk_profile',
                        lambda code_id: {'risk_score': 88.0, 'risk_band_label': '极高'})

    snapshot = ai_views._build_map_snapshot()
    assert snapshot['total'] == 1
    assert snapshot['points'][0]['id'] == lang.language_id
    assert snapshot['points'][0]['riskScore'] == 88.0
    assert snapshot['points'][0]['endangerment'] == ai_views.LEGACY_LEVEL_LABELS['endangered']
    assert ai_views._get_map_snapshot()['total'] == 1

    # 缓存结构损坏时重新构建
    cache.set(ai_views.MAP_SNAPSHOT_CACHE_KEY, {'bad': 1})
    assert ai_views._get_map_snapshot()['total'] == 1


def test_language_map_view(admin_client):
    lang = Language.objects.create(
        language_name='地图语言', legacy_endangerment_level='endangered',
        number_of_speakers=10, country='国', language_family='族',
    )
    LanguageLocation.objects.create(
        language_name=lang.language_name, longitude='120.1', latitude='30.2',
    )
    url = reverse('Dashboard:language_map')

    response = admin_client.get(url, {'bbox': '119,29,121,31', 'level': 'endangered',
                                      'mode': 'marker', 'format': 'json'})
    data = response.json()
    assert data['status'] == 'success'
    assert data['count'] == 1
    assert data['mode'] == 'marker'

    assert admin_client.get(url, {'bbox': 'bad'}).status_code == 400
    assert admin_client.get(url, {'level': 'bad'}).status_code == 400

    page = admin_client.get(url)
    assert page.status_code == 200


def test_translation_memory_and_rate(auth_client, test_user):
    memory = TranslationMemory.objects.create(
        source_text='原文', translated_text='译文', username=test_user.username,
    )
    url = reverse('Dashboard:translation_memory_list')
    assert auth_client.get(url).status_code == 200

    rate_url = reverse('Dashboard:translation_rate')
    assert auth_client.get(rate_url).json()['status'] == 'error'
    assert auth_client.post(rate_url, {'id': memory.id, 'score': '9'}).json()['status'] == 'error'
    assert auth_client.post(rate_url, {'id': memory.id, 'score': 'x'}).json()['status'] == 'error'
    assert auth_client.post(rate_url, {'id': 99999, 'score': '4'}).json()['status'] == 'error'
    assert auth_client.post(rate_url, {'id': memory.id, 'score': '4'}).json()['status'] == 'success'
    memory.refresh_from_db()
    assert memory.quality_score == 4


def test_glossary_manage_flows(auth_client, admin_client, test_user):
    url = reverse('Dashboard:glossary_manage')
    assert auth_client.get(url, {'search': '词', 'category': '语言'}).status_code == 200

    assert auth_client.post(url, {'action': 'add'}).json()['status'] == 'error'
    assert auth_client.post(url, {
        'action': 'add', 'term': '词', 'translation': 'word',
        'language': 'zh', 'category': '语言',
    }).json()['status'] == 'success'
    glossary = Glossary.objects.get(term='词')
    assert glossary.is_approved is False

    other = Glossary.objects.create(term='他人词', translation='x', username='other')
    assert auth_client.post(url, {'action': 'delete', 'id': other.id}).json()['message'] == '权限不足'
    assert auth_client.post(url, {'action': 'delete', 'id': glossary.id}).json()['status'] == 'success'
    assert auth_client.post(url, {'action': 'delete', 'id': 99999}).json()['status'] == 'error'

    pending = Glossary.objects.create(term='待审', translation='y', username='other')
    assert auth_client.post(url, {'action': 'approve', 'id': pending.id}).json()['message'] == '权限不足'
    assert admin_client.post(url, {'action': 'approve', 'id': pending.id}).json()['status'] == 'success'
    pending.refresh_from_db()
    assert pending.is_approved is True
    assert admin_client.post(url, {'action': 'approve', 'id': 99999}).json()['status'] == 'error'

    admin_client.post(url, {'action': 'add', 'term': '管理员词', 'translation': 'admin'})
    assert Glossary.objects.get(term='管理员词').is_approved is True


def test_rate_limit_and_classify_error():
    assert ai_views._retry_on_transient_error(lambda: 'ok') == 'ok'
    error = SimpleNamespace(status_code=429)
    assert ai_views.classify_ai_error(error) == 'rate_limit'
