"""Explicit harness-only Google fallback; disabled unless the plan enables it.

No billing settings are changed. The selected model has a documented free tier,
but a project configured for paid billing is not made free by this adapter.
"""
import base64
import json
import re
import time
import threading
import urllib.error
import urllib.request

from nvidia_api import NvidiaAPIError, retry_after_seconds, safe_error
from provider_catalog import NoCredentialRedirects

ALLOWED_FREE_TIER_MODELS = {'gemini-3.1-flash-lite', 'gemini-3.5-flash'}
BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
_PACE_LOCK = threading.Lock()
_LAST_REQUEST = {}
# Free-tier ceilings are per project/minute, so every client in this process must
# share one pacing state. A run may not quietly outrun the quota it was given.
_PACE_INTERVAL = {}
PACING_KEYS = {'min_interval_seconds', 'max_retry_delay_seconds', 'quota_retry_attempts',
               'rate_limit_floor_seconds'}
DEFAULT_PACING = {'min_interval_seconds': 8.0, 'max_retry_delay_seconds': 240.0,
                  'quota_retry_attempts': 4, 'rate_limit_floor_seconds': 20.0}
_RETRY_TEXT = re.compile(r'retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s', re.I)


def advertised_retry_seconds(body):
    """Google puts the minute-window wait in the message text, not a header."""
    match = _RETRY_TEXT.search(body or '')
    return float(match.group(1)) if match else None


def resolve_pacing(pacing):
    values = dict(DEFAULT_PACING)
    if pacing:
        unknown = set(pacing) - PACING_KEYS
        if unknown:
            raise ValueError('Unknown Google harness pacing keys: ' + ', '.join(sorted(unknown)))
        values.update(pacing)
    min_interval = float(values['min_interval_seconds'])
    cap = float(values['max_retry_delay_seconds'])
    attempts = int(values['quota_retry_attempts'])
    floor = float(values['rate_limit_floor_seconds'])
    if min_interval < 3:
        raise ValueError('Free-tier Google pacing must stay above 3 seconds per request')
    if not 30 <= cap <= 900:
        raise ValueError('Google retry delay cap must stay between 30 and 900 seconds')
    if not 1 <= attempts <= 8:
        raise ValueError('Google quota retries must stay between 1 and 8')
    if floor < min_interval:
        raise ValueError('The rate-limit floor cannot be shorter than normal pacing')
    return min_interval, cap, attempts, floor


def _shared_interval(base_url, floor):
    with _PACE_LOCK:
        return max(floor, _PACE_INTERVAL.get(base_url, floor))


def _bump_interval(base_url, floor, cap=120.0):
    """After a rate-limit reply, slow every client down, not just the unlucky one."""
    with _PACE_LOCK:
        current = max(floor, _PACE_INTERVAL.get(base_url, floor))
        _PACE_INTERVAL[base_url] = min(cap, max(current * 2.0, floor))
        return _PACE_INTERVAL[base_url]


def _relax_interval(base_url, floor):
    with _PACE_LOCK:
        current = _PACE_INTERVAL.get(base_url)
        if current and current > floor:
            _PACE_INTERVAL[base_url] = max(floor, current * 0.9)


def google_payload(messages, max_tokens):
    systems, contents = [], []
    for message in messages:
        content = message['content']
        if message['role'] == 'system':
            if not isinstance(content,str):
                raise ValueError('Google system instruction must be text')
            systems.append(content); continue
        parts = []
        if isinstance(content,str):
            parts = [{'text':content}]
        elif isinstance(content,list):
            for part in content:
                if part.get('type') == 'text':
                    parts.append({'text':part['text']})
                elif part.get('type') == 'image_url':
                    uri = part['image_url']['url']
                    match = re.fullmatch(r'data:(image/(?:jpeg|png));base64,(.+)',uri,re.S)
                    if not match:
                        raise ValueError('Only explicit local image data is supported; no remote fetch')
                    base64.b64decode(match.group(2),validate=True)
                    parts.append({'inlineData':{'mimeType':match.group(1),'data':match.group(2)}})
                else:
                    raise ValueError('Unsupported Google content part')
        else:
            raise ValueError('Unsupported message content')
        contents.append({'role':'model' if message['role']=='assistant' else 'user','parts':parts})
    return {'systemInstruction':{'parts':[{'text':'\n'.join(systems)}]},'contents':contents,
            'generationConfig':{'temperature':0,'maxOutputTokens':max_tokens,
                                'responseMimeType':'application/json','thinkingConfig':{'thinkingLevel':'minimal'}}}


class GoogleHarnessClient:
    def __init__(self, model, key, *, free_project_confirmed, budget, opener=None, pacing=None):
        if model not in ALLOWED_FREE_TIER_MODELS or free_project_confirmed is not True:
            raise ValueError('Google fallback requires an explicitly approved free-tier model/project in the harness plan')
        if not key:
            raise NvidiaAPIError('The selected Google harness credential is missing',401)
        self.min_interval, self.max_retry_delay, self.quota_attempts, self.rate_limit_floor = resolve_pacing(pacing)
        self.model, self.api_key = model, key
        self.base_url = BASE_URL
        self.timeout, self.attempts = 120, 3
        self.calls = 0
        self.budget = budget
        self.opener = opener or urllib.request.build_opener(NoCredentialRedirects())
        self.last_call = 0
        self.rate_limit_events = 0
        self.model_metadata = {'capabilities':{'vision':True},
                               'pricing_policy':'Documented free-tier model; user-confirmed free-tier project, not independent billing verification.'}

    def _pace(self):
        interval = _shared_interval(self.base_url, self.min_interval)
        with _PACE_LOCK:
            delay = interval-(time.monotonic()-_LAST_REQUEST.get(self.base_url,0))
            if delay>0: time.sleep(delay)
            self.last_call = time.monotonic()
            _LAST_REQUEST[self.base_url] = self.last_call

    def chat(self, model, messages, *, max_tokens=4096):
        if model != self.model:
            raise ValueError('Google model substitution is forbidden')
        if self.budget['used'] >= self.budget['limit']:
            raise NvidiaAPIError('Harness logical-call budget exhausted',429)
        self.budget['used'] += 1
        payload = google_payload(messages,max_tokens)
        start = time.monotonic()
        transport, throttled = 0, 0
        while True:
            self._pace()
            request = urllib.request.Request(f'{BASE_URL}/models/{model}:generateContent',
                data=json.dumps(payload).encode(),method='POST',
                headers={'x-goog-api-key':self.api_key,'Content-Type':'application/json','User-Agent':'RAGLab-hard-harness/1.0'})
            self.calls += 1
            try:
                with self.opener.open(request,timeout=self.timeout) as response:
                    data=json.load(response)
                candidates=data.get('candidates') or []
                if not candidates:
                    raise NvidiaAPIError('Google returned no candidate (possibly safety blocked)',422)
                candidate=candidates[0]
                if candidate.get('finishReason') != 'STOP':
                    raise NvidiaAPIError('Incomplete Google final output: '+str(candidate.get('finishReason')),422)
                text=''.join(p.get('text','') for p in candidate.get('content',{}).get('parts',[]) if not p.get('thought'))
                if not text.strip():
                    raise NvidiaAPIError('Google returned no final text',422)
                reported=data.get('modelVersion')
                if reported and not reported.startswith(model):
                    raise NvidiaAPIError(f'Google reported {reported!r} for requested {model!r}; not attributed to this model',422)
                usage=data.get('usageMetadata') or {}
                _relax_interval(self.base_url,self.min_interval)
                return {'text':text,'requested_model':model,'served_model':data.get('modelVersion'),
                        'seconds':round(time.monotonic()-start,3),
                        'rate_limit_waits':self.rate_limit_events,
                        'usage':{'prompt_tokens':usage.get('promptTokenCount'),
                                 'completion_tokens':usage.get('candidatesTokenCount'),
                                 'reasoning_tokens':usage.get('thoughtsTokenCount'),
                                 'total_tokens':usage.get('totalTokenCount')}}
            except urllib.error.HTTPError as exc:
                body=exc.read().decode(errors='replace').replace(self.api_key,'[REDACTED]')
                error=NvidiaAPIError(f'Google HTTP {exc.code}: {safe_error(body)}',exc.code,
                                     retry_after_seconds(exc.headers.get('Retry-After')))
                wait = error.retry_after if error.retry_after is not None else advertised_retry_seconds(body)
            except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
                error=NvidiaAPIError('Google connection error: '+safe_error(exc))
                wait=None
            except NvidiaAPIError:
                raise
            # Auth/billing errors must prompt a user decision instead of spending
            # another project silently.
            if error.status_code in {401,402,403}:
                raise error
            if error.status_code == 429:
                # A minute-window free-tier limit is a scheduling fact: honour the
                # advertised wait and slow down. An unlabelled 429 or a long
                # advertised wait can be a daily quota, so it must stop and ask the
                # user instead of looping; no model or project is switched silently.
                throttled += 1
                if wait is None or wait > self.max_retry_delay or throttled > self.quota_attempts:
                    raise error
                self.rate_limit_events += 1
                delay = max(wait, _bump_interval(self.base_url,self.rate_limit_floor))
                time.sleep(min(delay,self.max_retry_delay))
                continue
            transport += 1
            if not error.retryable or transport >= self.attempts:
                raise error
            delay = wait if wait is not None else min(self.max_retry_delay,30*(2**(transport-1)))
            if delay > self.max_retry_delay:
                raise error
            time.sleep(delay)
