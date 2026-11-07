"""联网搜索服务单测：全部使用 mock，不消耗 Tavily 额度。"""

from datetime import date
from unittest import mock

import httpx
import pytest
from django.core.cache import cache
from django.test import override_settings

from Dashboard.search_service import (
    NullProvider,
    SearchOutcome,
    SearchProvider,
    SearchResult,
    SearchStatus,
    enforce_citations,
    format_results_for_llm,
    get_provider,
    search_is_available,
    search_web,
    serialize_sources,
)
from Dashboard.search_service import TavilyProvider


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append({'url': url, 'json': json, 'headers': headers})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()
    yield
    cache.clear()


def _payload():
    return {
        'results': [
            {
                'title': '濒危语言保护进展',
                'url': 'https://example.com/a',
                'content': '这是一段摘要。',
                'score': 0.9,
            },
            {
                'title': '没有链接的结果',
                'url': 'javascript:alert(1)',
                'content': '应被过滤',
            },
        ]
    }


@override_settings(SEARCH_ENABLED=False, SEARCH_API_KEY='', TAVILY_API_KEY='')
def test_no_key_falls_back_to_null_provider():
    assert isinstance(get_provider(), NullProvider)
    assert search_is_available() is False
    outcome = search_web('濒危语言', use_cache=False)
    assert outcome.status == SearchStatus.DISABLED
    assert outcome.results == []


@override_settings(SEARCH_ENABLED=True, SEARCH_PROVIDER='tavily', SEARCH_API_KEY='test-key', SEARCH_DAILY_LIMIT=0)
def test_tavily_parses_results_and_filters_bad_urls():
    client = FakeClient([FakeResponse(200, _payload())])
    with mock.patch('Dashboard.search_service._http_client_factory', return_value=client):
        outcome = search_web('濒危语言 保护', use_cache=False)

    assert outcome.status == SearchStatus.OK
    assert len(outcome.results) == 1
    assert outcome.results[0].url == 'https://example.com/a'
    assert client.calls[0]['headers']['Authorization'] == 'Bearer test-key'
    assert client.calls[0]['json']['search_depth'] == 'basic'


@override_settings(SEARCH_ENABLED=True, SEARCH_PROVIDER='tavily', SEARCH_API_KEY='test-key', SEARCH_DAILY_LIMIT=0)
def test_401_returns_unauthorized_without_raising():
    client = FakeClient([FakeResponse(401)])
    with mock.patch('Dashboard.search_service._http_client_factory', return_value=client):
        outcome = search_web('test', use_cache=False)
    assert outcome.status == SearchStatus.UNAUTHORIZED
    assert 'Key 无效' in outcome.message


@override_settings(SEARCH_ENABLED=True, SEARCH_PROVIDER='tavily', SEARCH_API_KEY='test-key', SEARCH_DAILY_LIMIT=0)
def test_timeout_returns_timeout_without_raising():
    client = FakeClient([httpx.TimeoutException('boom')])
    with mock.patch('Dashboard.search_service._http_client_factory', return_value=client):
        outcome = search_web('test', use_cache=False)
    assert outcome.status == SearchStatus.TIMEOUT


@override_settings(
    SEARCH_ENABLED=True, SEARCH_PROVIDER='tavily', SEARCH_API_KEY='test-key',
    SEARCH_DAILY_LIMIT=0, SEARCH_RETRY_MAX_WAIT=3,
)
def test_429_retries_once_after_retry_after():
    client = FakeClient([
        FakeResponse(429, headers={'Retry-After': '2'}),
        FakeResponse(200, _payload()),
    ])
    provider = TavilyProvider('test-key', 'https://api.tavily.com')
    with mock.patch('Dashboard.search_service.get_provider', return_value=provider), \
            mock.patch('Dashboard.search_service._http_client_factory', return_value=client), \
            mock.patch('Dashboard.search_service._sleep') as sleep:
        outcome = search_web('test', use_cache=False)

    assert outcome.status == SearchStatus.OK
    assert len(client.calls) == 2
    sleep.assert_called_once_with(2.0)


@override_settings(SEARCH_ENABLED=True, SEARCH_PROVIDER='tavily', SEARCH_API_KEY='test-key', SEARCH_DAILY_LIMIT=0)
def test_429_second_time_returns_rate_limited():
    client = FakeClient([
        FakeResponse(429, headers={'Retry-After': '1'}),
        FakeResponse(429),
    ])
    with mock.patch('Dashboard.search_service._http_client_factory', return_value=client), \
            mock.patch('Dashboard.search_service._sleep'):
        outcome = search_web('test', use_cache=False)
    assert outcome.status == SearchStatus.RATE_LIMITED


class FakeProvider(SearchProvider):
    name = 'fake'

    def __init__(self):
        self.calls = 0

    def search(self, query, top_k=5):
        self.calls += 1
        return [SearchResult(title='示例', url='https://example.com/x', snippet='摘要')]


@override_settings(SEARCH_ENABLED=True, SEARCH_PROVIDER='tavily', SEARCH_API_KEY='test-key', SEARCH_DAILY_LIMIT=0)
def test_cache_hit_does_not_call_provider_twice():
    provider = FakeProvider()
    with mock.patch('Dashboard.search_service.get_provider', return_value=provider):
        first = search_web('同一个问题')
        second = search_web('同一个问题')

    assert first.status == SearchStatus.OK
    assert second.cached is True
    assert provider.calls == 1


@override_settings(SEARCH_ENABLED=True, SEARCH_PROVIDER='tavily', SEARCH_API_KEY='test-key', SEARCH_DAILY_LIMIT=1)
def test_daily_budget_stops_real_calls():
    provider = FakeProvider()
    with mock.patch('Dashboard.search_service.get_provider', return_value=provider):
        first = search_web('问题一', use_cache=False)
        second = search_web('问题二', use_cache=False)

    assert first.status == SearchStatus.OK
    assert second.status == SearchStatus.BUDGET_EXCEEDED
    assert provider.calls == 1


@pytest.mark.parametrize('daily_limit', [0, ''])
def test_daily_limit_zero_or_empty_is_unlimited_and_still_counts(daily_limit):
    """0 或空 = 不拦截，但每日计数仍然保留。"""
    provider = FakeProvider()
    with override_settings(
        SEARCH_ENABLED=True,
        SEARCH_PROVIDER='tavily',
        SEARCH_API_KEY='test-key',
        SEARCH_DAILY_LIMIT=daily_limit,
    ), mock.patch('Dashboard.search_service.get_provider', return_value=provider):
        outcomes = [search_web(f'不限流问题{i}', use_cache=False) for i in range(3)]

    assert [outcome.status for outcome in outcomes] == [SearchStatus.OK] * 3
    assert provider.calls == 3
    day_key = f'websearch:daily:{date.today().isoformat()}:fake'
    assert cache.get(day_key) == 3


def test_format_results_marks_untrusted_and_includes_urls():
    outcome = SearchOutcome(
        SearchStatus.OK,
        provider='tavily',
        results=[SearchResult(title='标题', url='https://example.com/a', snippet='摘要')],
    )
    text = format_results_for_llm(outcome)
    assert '<search_results' in text
    assert 'https://example.com/a' in text
    assert '不可信数据' in text


def test_enforce_citations_removes_unknown_urls_and_appends_footer():
    results = [SearchResult(title='真实来源', url='https://example.com/real')]
    answer = '据最新资料，答案在这里 [1](https://example.com/real)，另见 https://fake.example/bad'
    cleaned, warning = enforce_citations(answer, results)

    assert warning is True
    assert 'https://example.com/real' in cleaned
    assert 'fake.example' not in cleaned
    # 正文已经引用了真实 URL，不再重复追加页脚
    assert '参考来源' not in cleaned


def test_enforce_citations_adds_footer_when_no_citation():
    results = [SearchResult(title='来源', url='https://example.com/a')]
    cleaned, warning = enforce_citations('模型没有引用链接。', results)
    assert warning is False
    assert '参考来源' in cleaned


def test_enforce_citations_corrects_fake_search_claim():
    cleaned, warning = enforce_citations('根据最新搜索结果，答案是 X。', [])
    assert warning is True
    assert '未检索到可核实的网页来源' in cleaned


def test_serialize_sources_rejects_non_http():
    results = [
        SearchResult(title='ok', url='https://example.com/a'),
        SearchResult(title='bad', url='javascript:alert(1)'),
    ]
    sources = serialize_sources(results)
    assert len(sources) == 1
    assert sources[0]['url'] == 'https://example.com/a'


def test_tavily_provider_requires_key():
    provider = TavilyProvider('', 'https://api.tavily.com')
    with pytest.raises(Exception) as exc:
        provider.search('x')
    assert exc.value.status == SearchStatus.DISABLED
