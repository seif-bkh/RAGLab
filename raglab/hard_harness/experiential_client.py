"""Harness adapter for the Experiential Labs OpenAI-compatible gateway.

Why this exists: the semantic grader is a *judging* role, and the answer model must not grade
itself. Google's free tier is the only other approved judge and its per-minute ceiling left 206
of 300 comparisons unscored, so grading needed a provider that is neither the answerer nor
quota-bound in the same way.

What the gateway contract requires of us (platform.experientiallabs.ai/docs):
  * base URL ``https://api.experientiallabs.ai/v1``, one header ``Authorization: Bearer <xpl_..>``;
  * models are named by bare catalog slug - here ``glm-5.3-flash``;
  * the smallest useful body: the docs warn that sampling parameters some routes reject come
    back as ``all_routes_failed`` (502), so temperature, top_p and response_format are never sent.
    ``max_tokens`` is kept because GLM-5.3 thinks before it writes: with a small ceiling the
    reasoning consumes the whole completion and the assistant message arrives empty.
This is not a free tier. Routed tokens are billed per token against the account's credits, so the
plan has to say ``credits_acknowledged: true`` and every published row labels the grader provider.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from nvidia_api import NvidiaAPIError, retry_after_seconds, safe_error
from provider_catalog import NoCredentialRedirects

ALLOWED_GATEWAY_MODELS = {'glm-5.3-flash'}
DEFAULT_BASE_URL = 'https://api.experientiallabs.ai/v1'

_PACE_LOCK = threading.Lock()
_PACE_INTERVAL = {}
_LAST_REQUEST = {}
PACING_KEYS = {'min_interval_seconds', 'max_retry_delay_seconds', 'rate_limit_floor_seconds'}
DEFAULT_PACING = {'min_interval_seconds': 1.0, 'max_retry_delay_seconds': 120.0,
                  'rate_limit_floor_seconds': 2.0}
_RETRY_TEXT = re.compile(r'retry after\s+([0-9]+(?:\.[0-9]+)?)\s*second', re.I)


def gateway_base_url(profile=None):
    """The plan may point at another deployment of the same gateway, never at a different vendor."""
    configured = (profile or {}).get('base_url') or os.environ.get('EXPERIENTIAL_BASE_URL') or DEFAULT_BASE_URL
    return str(configured).rstrip('/')


def advertised_retry_seconds(body):
    match = _RETRY_TEXT.search(body or '')
    return float(match.group(1)) if match else None


def resolve_pacing(pacing):
    values = dict(DEFAULT_PACING)
    if pacing:
        unknown = set(pacing) - PACING_KEYS
        if unknown:
            raise ValueError('Unknown Experiential Labs pacing keys: ' + ', '.join(sorted(unknown)))
        values.update(pacing)
    return float(values['min_interval_seconds']), float(values['max_retry_delay_seconds']), \
        float(values['rate_limit_floor_seconds'])


def chat_request_body(model, messages, max_tokens=None, include_limit=True):
    """Model and messages, plus an output ceiling only when asked to send one."""
    body = {'model': model, 'messages': messages}
    if include_limit and max_tokens:
        body['max_tokens'] = int(max_tokens)
    return body


def assistant_text(data):
    """Content only - a reasoning summary is not an answer and must never be graded as one."""
    choices = data.get('choices') or []
    if not choices:
        raise NvidiaAPIError('Gateway returned no completion choices', 502)
    message = choices[0].get('message') or {}
    content = message.get('content')
    if isinstance(content, str) and content.strip():
        return content, message.get('reasoning_content') or message.get('reasoning') or ''
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    if reasoning:
        raise NvidiaAPIError('GLM returned reasoning but no answer text: the output ceiling was '
                             'consumed by thinking. Raise the role max_tokens instead of accepting '
                             'an empty reply', 502)
    raise NvidiaAPIError('Gateway returned an empty assistant message', 502)


class ExperientialHarnessClient:
    """Chat Completions against the gateway, with the harness's own call budget and pacing."""

    # The adapter owns its pacing, so the checkpoint layer must not stack a retry allowance on top.
    paces_rate_limits = True

    def __init__(self, model, key, *, budget, opener=None, pacing=None, base_url=None):
        if model not in ALLOWED_GATEWAY_MODELS:
            raise ValueError('The harness may only call an approved Experiential Labs catalog slug; '
                             f'got {model!r}')
        if not key:
            raise NvidiaAPIError('The Experiential Labs harness credential is missing', 401)
        self.min_interval, self.max_retry_delay, self.rate_limit_floor = resolve_pacing(pacing)
        self.model, self.api_key = model, key
        self.base_url = gateway_base_url({'base_url': base_url})
        self.budget = budget
        self.opener = opener or urllib.request.build_opener(NoCredentialRedirects())
        self.calls = 0
        self.rate_limit_events = 0
        self.last_call = 0.0
        self.timeout = 180
        self.attempts = 3
        self.model_metadata = {
            'capabilities': {'structured_output': False, 'vision': True},
            'pricing_policy': ('Gateway-routed tokens billed per token against account credits; no '
                              'free-tier assumption is made by this adapter.')}

    def _wait_for_turn(self):
        with _PACE_LOCK:
            interval = _PACE_INTERVAL.get(self.base_url, self.min_interval)
            due = _LAST_REQUEST.get(self.base_url, 0.0) + interval
        now = time.monotonic()
        if due > now:
            time.sleep(due - now)
        with _PACE_LOCK:
            _LAST_REQUEST[self.base_url] = time.monotonic()

    def _post(self, path, payload):
        request = urllib.request.Request(self.base_url + path,
                                        data=json.dumps(payload).encode('utf-8'),
                                        headers={'Authorization': 'Bearer ' + self.api_key,
                                                 'Content-Type': 'application/json',
                                                 'User-Agent': 'raglab-hard-harness/1'},
                                        method='POST')
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    def list_models(self):
        """Catalog check for this key. Costs nothing, and is how a slug typo is caught early."""
        request = urllib.request.Request(self.base_url + '/models',
                                        headers={'Authorization': 'Bearer ' + self.api_key},
                                        method='GET')
        with self.opener.open(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
        return sorted({row.get('id') or row.get('slug') for row in data.get('data') or []
                       if (row.get('id') or row.get('slug'))})

    def chat(self, model, messages, *, max_tokens=6000, **kwargs):
        if model != self.model:
            raise NvidiaAPIError(f'Experiential client is bound to {self.model}, not {model}', 400)
        attempts, throttled = 0, 0
        limit_dropped = False
        while True:
            attempts += 1
            if self.budget is not None and self.budget['used'] >= self.budget['limit']:
                raise NvidiaAPIError('Harness logical-call budget exhausted', 429)
            if self.budget is not None:
                self.budget['used'] += 1
            payload = chat_request_body(model, messages, max_tokens, include_limit=not limit_dropped)
            self._wait_for_turn()
            self.calls += 1
            started = time.monotonic()
            try:
                data = self._post('/chat/completions', payload)
                text, reasoning = assistant_text(data)
                usage = data.get('usage') or {}
                with _PACE_LOCK:
                    current = _PACE_INTERVAL.get(self.base_url, self.min_interval)
                    if current > self.rate_limit_floor:
                        _PACE_INTERVAL[self.base_url] = max(self.rate_limit_floor, current * 0.9)
                return {'text': text, 'requested_model': model, 'served_model': data.get('model'),
                        'seconds': round(time.monotonic() - started, 3),
                        'rate_limit_waits': self.rate_limit_events,
                        'reasoning_chars': len(reasoning or ''),
                        'usage': {'prompt_tokens': usage.get('prompt_tokens'),
                                  'completion_tokens': usage.get('completion_tokens'),
                                  'total_tokens': usage.get('total_tokens')}}
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors='replace').replace(self.api_key, '[REDACTED]')
                status = exc.code
                wait = retry_after_seconds(exc.headers.get('Retry-After'))
                if wait is None:
                    wait = advertised_retry_seconds(body)
                # The gateway says all_routes_failed when a route rejects a field. Dropping the
                # output ceiling once is the documented escape; a second such answer is a real
                # failure and is raised rather than retried into a spend.
                if status == 502 and not limit_dropped and 'all_routes_failed' in body:
                    limit_dropped = True
                    continue
                if status == 429:
                    throttled += 1
                    if throttled > 3 or (wait or 0) > self.max_retry_delay:
                        raise NvidiaAPIError(f'Experiential Labs HTTP {status}: {safe_error(body)}',
                                             status, wait) from None
                    self.rate_limit_events += 1
                    with _PACE_LOCK:
                        _PACE_INTERVAL[self.base_url] = max(_PACE_INTERVAL.get(self.base_url, 0.0),
                                                             wait or self.rate_limit_floor)
                        _LAST_REQUEST[self.base_url] = time.monotonic()
                    time.sleep(min(wait or self.rate_limit_floor, self.max_retry_delay))
                    continue
                if status in {401, 402, 403}:
                    raise NvidiaAPIError(f'Experiential Labs refused the credential (HTTP {status}); '
                                         'no other provider is substituted: ' + safe_error(body),
                                         status) from None
                raise NvidiaAPIError(f'Experiential Labs HTTP {status}: {safe_error(body)}', status) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempts >= 3:
                    raise NvidiaAPIError('Experiential Labs connection error: ' + safe_error(exc)) from exc
                time.sleep(min(self.max_retry_delay, 5 * (2 ** (attempts - 1))))
