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

ALLOWED_FREE_TIER_MODELS = {'gemini-3.1-flash-lite'}
BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
_PACE_LOCK = threading.Lock()
_LAST_REQUEST = {}


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
    def __init__(self, model, key, *, free_project_confirmed, budget, opener=None):
        if model not in ALLOWED_FREE_TIER_MODELS or free_project_confirmed is not True:
            raise ValueError('Google fallback requires an explicitly approved free-tier model/project in the harness plan')
        if not key:
            raise NvidiaAPIError('The selected Google harness credential is missing',401)
        self.model, self.api_key = model, key
        self.base_url = BASE_URL
        self.timeout, self.attempts, self.min_interval, self.max_retry_delay = 120, 2, 6, 60
        self.calls = 0
        self.budget = budget
        self.opener = opener or urllib.request.build_opener(NoCredentialRedirects())
        self.last_call = 0
        self.model_metadata = {'capabilities':{'vision':True},
                               'pricing_policy':'Documented free-tier model; user-confirmed free-tier project, not independent billing verification.'}

    def chat(self, model, messages, *, max_tokens=4096):
        if model != self.model:
            raise ValueError('Google model substitution is forbidden')
        if self.budget['used'] >= self.budget['limit']:
            raise NvidiaAPIError('Harness logical-call budget exhausted',429)
        self.budget['used'] += 1
        payload = google_payload(messages,max_tokens)
        start = time.monotonic()
        for attempt in range(self.attempts):
            with _PACE_LOCK:
                delay = self.min_interval-(time.monotonic()-_LAST_REQUEST.get(self.base_url,0))
                if delay>0: time.sleep(delay)
                self.last_call = time.monotonic()
                _LAST_REQUEST[self.base_url] = self.last_call
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
                return {'text':text,'requested_model':model,'served_model':data.get('modelVersion'),
                        'seconds':round(time.monotonic()-start,3),
                        'usage':{'prompt_tokens':usage.get('promptTokenCount'),
                                 'completion_tokens':usage.get('candidatesTokenCount'),
                                 'reasoning_tokens':usage.get('thoughtsTokenCount'),
                                 'total_tokens':usage.get('totalTokenCount')}}
            except urllib.error.HTTPError as exc:
                body=exc.read().decode(errors='replace').replace(self.api_key,'[REDACTED]')
                error=NvidiaAPIError(f'Google HTTP {exc.code}: {safe_error(body)}',exc.code,
                                     retry_after_seconds(exc.headers.get('Retry-After')))
            except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
                error=NvidiaAPIError('Google connection error: '+safe_error(exc))
            except NvidiaAPIError:
                raise
            # Quota/auth errors prompt a switch instead of spending more of a
            # different project/account silently. Only transient transport gets
            # one bounded retry.
            if error.status_code in {401,402,403,429} or not error.retryable or attempt+1==self.attempts:
                raise error
            pause=error.retry_after if error.retry_after is not None else 2
            if pause>self.max_retry_delay:
                raise error
            time.sleep(pause)
